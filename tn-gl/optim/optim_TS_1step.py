"""Optimizer for one step of Trotter-Suzuki decomposition. The optimizer
maximizes the ratio of overlaps w.r.t the coefficients of the basic Cs tensors
and updates them."""

import sys
sys.path.append('/mnt/D484FDE484FDC94E/Documents/StageTN/TensorNetwork/peps-torch/')
import torch
import pickle
import numpy as np
from tqdm import tqdm #progress bars
import tensors.tensor_sum as ts
# peps-torch imports
import groups.su2 as su2
import models.j1j2 as j1j2
import config as cfg
from ipeps.ipeps_c4v import read_ipeps_c4v
from ctm.one_site_c4v import ctmrg_c4v
from ctm.one_site_c4v.rdm_c4v import rdm2x1_sl
from ctm.one_site_c4v.env_c4v import ENV_C4V, init_env
import logging
log = logging.getLogger(__name__)
path = '/mnt/D484FDE484FDC94E/Documents/StageTN/TensorNetwork/tn-gl/'

# Get parser from config
parser= cfg.get_args_parser()
parser.add_argument("--j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("--j2", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("--n", type=int, default=10, help="number of optimization steps")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
args, unknown_args = parser.parse_known_args()


def overlap(w0, w1, w2):
    return w0/torch.sqrt(w1*w2)


def build_gate(tau=1.0, H=j1j2.J1J2_C4V_BIPARTITE(j1=args.j1, j2=args.j2)):
    """Part of j1j2.py code.
    H is a class with j1=1.0 and j2=0 by default.
    Simple pi/2 rotation of the gate gives the Hamiltonian on another bond."""
    s2 = su2.SU2(H.phys_dim, dtype=H.dtype, device=H.device)
    expr_kron = 'ij,ab->iajb'
    
    # Spin-spin operator
    #   s1|   |s2
    #     |   |
    #   [  S.S  ]
    #     |   |
    #  s1'|   |s2'
    
    SS = torch.einsum(expr_kron,s2.SZ(),s2.SZ()) + 0.5*(torch.einsum(expr_kron,s2.SP(),s2.SM()) \
        + torch.einsum(expr_kron,s2.SM(),s2.SP()))
    SS = SS.view(4,4).contiguous()
    
    # Diagonalization of SS and creation of Hamiltonian Ha
    eig_va, eig_vec = np.linalg.eigh(SS)
    eig_va = np.exp(-tau/2*args.j1*eig_va)
    U = torch.tensor(eig_vec); D = torch.diag(torch.tensor(eig_va))
    # Ha = U D U^{\dagger}
    Ga = torch.einsum('ij,jk,lk->il', U, D, U)
    Ga = Ga.view(2,2,2,2).contiguous()
    return Ga


def compute_w0(B_tensor, env):
    """
    Be |phi'> the purified PEPS associated with A' tensor. 
    Compute the overlap <phi'|phi'>.

    Parameters
    ----------
    B_tensor : torch.tensor(4,4,4,4,4)

    Returns
    -------
    w0 : float
    """
    rdm = rdm2x1_sl((B_tensor, B_tensor), env)
    rdm = rdm.view((*[2]*8)).contiguous()
    # contract ancilla degrees of freedom
    rdm = torch.einsum('abcdafch', rdm)
    w0 = torch.einsum('abab', rdm) 
    return w0


def compute_w1(A_tensor, B_tensor, gate, env):
    """
    Compute the overlap <phi|G|phi'>.

    Parameters
    ----------
    A_tensor, B_tensor : torch.tensor(4,4,4,4,4)

    gate : torch.tensor

    Returns
    -------
    w1 : float
    """
    rdm = rdm2x1_sl((A_tensor, B_tensor), env)
    rdm = rdm.view((*[2]*8)).contiguous()
    # contract ancilla degrees of freedom
    rdm = torch.einsum('abcdafch', rdm)
    w1 = torch.einsum('abcd, cdab', rdm, gate)
    return w1


def compute_w2(A_tensor, gate, env):
    """
    Be |phi> the purified PEPS associated with A tensor. 
    Compute the overlap <phi|G*G|phi>.

    Parameters
    ----------
    A_tensor : torch.tensor(4,4,4,4,4)
    
    gate : torch.tensor

    Returns
    -------
    w2 : float
    """
    rdm = rdm2x1_sl((A_tensor, A_tensor), env)
    rdm = rdm.view((*[2]*8)).contiguous()
    # contract ancilla degrees of freedom
    rdm = torch.einsum('abcdafch', rdm)
    w2 = torch.einsum('cdij,ijkl->cdkl', gate, gate)
    w2 = torch.einsum('abcd, cdab', rdm, w2)
    return w2


def run_optimization(Ta_coef, Tb_coef, gate, env,
                     optimizer_class, n_iter, **optimizer_kwargs):
    """
    Run optimization to find the maximum of O = \frac{w0}{\sqrt{w1*w2}}. At
    each step of the optimization, 3 overlaps w0, w1 and w2 are computed using
    CTMRG. Then O is computed and Tb_coef is updated.
    
    Parameters
    ----------  
    Ta_coef : list of floats
        Initial coefficients. The size of the list corresponds to the number of
        tensors in the class of symmetry.
        Ta_ten and Ta_coef are used to build tensor A which is C4v symmetric.
    
    Tb_coef : list of floats
        Random initial coefficients. The size of the list corresponds to the
        number of tensors in the class of symmetry.
        Tb_ten and Tb_coef are used to build A' which is Cs symmetric.

    gate : torch.tensor
        G is the TS IT operator.

    optimizer_class : object
        Optimizer class.
        
    n_iter : int
        Number of iterations of the optimization.
        
    optimizer_kwargs : dict
        Additional parameters to be passed to the optimizer.
        
    Returns
    -------
    coef_opti : np.ndarray
        2D array of shape (n_iter, len(Tb_coef)). Where the rows represent the
        iteration and the columns represent the updated list.
    """
    Tb_coef_t = torch.tensor(Tb_coef, dtype=torch.float64, requires_grad=True)
    optimizer = optimizer_class([Tb_coef_t], **optimizer_kwargs)
    
    coef_opti = np.empty((n_iter, len(Tb_coef)))
    coef_opti[0,:] = Tb_coef
    
    for i in tqdm(range(n_iter)):
        optimizer.zero_grad()
        
        A_tensor = ts.build_A_tensor(Ta_coef)
        B_tensor = ts.build_B_tensor(Tb_coef_t)
        w0 = compute_w0(B_tensor, env)
        w1 = compute_w1(A_tensor, B_tensor, gate, env)
        w2 = compute_w2(A_tensor, gate, env)
        loss = -overlap(w0, w1, w2)

        # Compute gradient
        loss.backward()
        
        # Clip norm gradients to 1.0 to garantee they are not exploding
        torch.nn.utils.clip_grad_norm_(Tb_coef_t, 1.0)  
        
        # Update value of the coefficients
        optimizer.step()
        
        coef_opti[i, :] = Tb_coef_t.detach().numpy()
        
    file = open(path+"output/output", "wb")
    pickler = pickle.Pickler(file)
    pickler.dump(coef_opti)
    file.close()
        
def main():
    # 0) Parse command line arguments and configure simulation parameters
    cfg.configure(args)
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)
            
    # 1) Read IPEPS from .json file
    state_ini = read_ipeps_c4v(path+"tensors/input-states/tensor_Ta4.json")
    
    # 2) Initialize environment and convergence criterion 
    ctm_env_init = ENV_C4V(args.chi, state_ini)
    init_env(state_ini, ctm_env_init)
    def ctmrg_conv_rdm2x1(state_ini, env, history, ctm_args=cfg.ctm_args):
        with torch.no_grad():
            if not history:
                history = dict({"log": []})
            rdm2x1 = rdm2x1_sl(state_ini, env, force_cpu=ctm_args.conv_check_cpu)
            dist = float('inf')
            if len(history["log"]) > 1:
                dist = torch.dist(rdm2x1, history["rdm"], p=2).item()
            # update history
            history["rdm"] = rdm2x1
            history["log"].append(dist)
        return False, history
    
    # 3) Execute CTM algorithm
    ctm_env_init, *ctm_log = ctmrg_c4v.run(state_ini, ctm_env_init, conv_check=ctmrg_conv_rdm2x1)

    # 4) Initialize parameters
    gate = build_gate()
    Ta_coef = [0,0,0,0,1,0,0,0]
    Tb_coef = [*[1]*13]
    
    # 5) Run optimization
    run_optimization(Ta_coef=Ta_coef, Tb_coef=Tb_coef, 
                     gate=gate, env=ctm_env_init,
                     optimizer_class=torch.optim.SGD, 
                     n_iter=args.n, lr=args.lr)

    

if __name__ == '__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()
