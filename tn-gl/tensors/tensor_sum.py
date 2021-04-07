""""Build the Alpha tensor SU(2) and C4v symmetric from Mathieu's Mathematica
    library."""
import sys
sys.path.append('/mnt/D484FDE484FDC94E/Documents/StageTN/TensorNetwork/peps-torch/')
import torch
import tensors.tensors_D4_R3
import tensors.tensors_D4_R5
import ipeps.ipeps as ipeps
import ipeps.ipeps_c4v as ipepsc4v
path = '/mnt/D484FDE484FDC94E/Documents/StageTN/TensorNetwork/tn-gl/tensors/'

### WRITING BASIC TENSORS INTO IPEPS CLASS IN THE .JSON FORMAT ###

def contract_Ta(tensor1, tensor2):
    """ Contract spin-SU(2) symmetric tensors of rank-5 and rank-3 into a
    basic tensor T_{\alpha} 
    Indices: (ancilla, physical, u, l, d, r) """
    return torch.einsum('atp,auldr->tpuldr', tensor1, tensor2)


def contract_A_c4v():
    """Return the list of the 8 basic c4v Ta tensors.
    """
    tensor_Ta_list = []
    for tensor in tensors.tensors_D4_R5.list_S0:
        tensor_Ta_list.append(contract_Ta(tensors.tensors_D4_R3.T_2_B_1, tensor))
    for tensor in tensors.tensors_D4_R5.list_S1:
        tensor_Ta_list.append(contract_Ta(tensors.tensors_D4_R3.T_2_A_1, tensor))
    return tensor_Ta_list


def build_Ta_tensors():
    """Build the .json file of the 8 Ta tensors c4v-sym for the peps-torch CTM.
    """
    tensor_Ta_list = contract_A_c4v()
    for i in range(len(tensor_Ta_list)):
        tensor = tensor_Ta_list[i].view(4,4,4,4,4).contiguous()
        ipeps.write_ipeps(ipepsc4v.IPEPS_C4V(tensor), path+f"input-states/tensor_Ta{i}.json")


def build_A_tensor(coef_list):
    """Build the .json file associated with the A tensor from the Ta .json files.
    """
    A_tensor = torch.zeros([4,4,4,4,4], dtype=torch.float64)
    for i in range(len(coef_list)):
        tensor_Ta = ipepsc4v.read_ipeps_c4v(path+f"input-states/tensor_Ta{i}.json")
        A_tensor += coef_list[i]*tensor_Ta.site()
    ipeps.write_ipeps(ipepsc4v.IPEPS_C4V(A_tensor),path+"input-states/A_tensor.json")
    return A_tensor

def contract_B_cs():
    """Return the list of the 8 basic c4v Ta tensors.
    """
    tensor_Ta_list = []
    for tensor in tensors.tensors_D4_R5.list_S0:
        tensor_Ta_list.append(contract_Ta(tensors.tensors_D4_R3.T_2_B_1, tensor))
    for tensor in tensors.tensors_D4_R5.list_S1:
        tensor_Ta_list.append(contract_Ta(tensors.tensors_D4_R3.T_2_A_1, tensor))
    return tensor_Ta_list


def build_Tb_tensors():
    """Build the .json file of the 13 Tb tensors c4v-sym for the peps-torch CTM.
    """
    tensor_Tb_list = contract_B_cs()
    for i in range(len(tensor_Tb_list)):
        tensor = tensor_Tb_list[i].view(4,4,4,4,4).contiguous()
        ipeps.write_ipeps(ipepsc4v.IPEPS_C4V(tensor),path+f"input-states/tensor_Tb{i}.json")


def build_B_tensor(coef_list):
    """Build the .json file associated with the B tensor from the Ta2 .json files.
    """
    B_tensor = torch.zeros([4,4,4,4,4], dtype=torch.float64)
    for i in range(len(coef_list)):
        tensor_Tb = ipeps.read_ipeps(path+f"input-states/tensor_Tb{i}.json")
        B_tensor += coef_list[i]*tensor_Tb.site()
    ipeps.write_ipeps(ipepsc4v.IPEPS_C4V(B_tensor),path+"input-states/B_tensor.json")
    return B_tensor

### BUILDING IPEPO TENSORS

def build_A(coef_list):
    """Return the A tensor which is the sum of the basic c4v tensors.
    """
    # revoir les coef
    A_tensor = torch.zeros([2, 2, 4, 4, 4, 4], dtype=torch.float64)
    for tensor in contract_A_c4v():
        A_tensor += tensor*coef_list.pop()
    return A_tensor


def build_E(X_tensor):
    """ Contract two tensors of rank-6 along the ancilla degree of freedom and then
    merge the auxiliary dimensions to get a tensor of rank 6.
    
                 phys u
                    |/
                 l--X--r
                   /|
                  d | u
                    |/
                 l--X*--r
                   /|
                  d phys
                  
    """
    if len(X_tensor.size()) <= 6:
        X_tensor = X_tensor.view(2, 2, 4, 4, 4, 4).contiguous()
    E = torch.tensordot(X_tensor, X_tensor, dims=([0], [0]))
    length0 = len(E.size())
    length = length0
    while length > length0//2:
        E = E.permute(length0//2-1, length-1, *range(0,length0//2-1),
                  *range(length0//2,length-1))
        E = E.reshape(E.shape[0]**2, *E.shape[2:])
        length=len(E.size())
    return E

    
if __name__ == '__main__':
    build_Ta_tensors()