import pickle
import matplotlib.pyplot as plt
path = '/mnt/D484FDE484FDC94E/Documents/StageTN/TensorNetwork/tn-gl/'

def make_dico(file):
    depickler = pickle.Unpickler(file)
    result = depickler.load()
    dico = {}
    for i in range(len(result[0])):
        dico[f"coef_{i}"]=[]
    for optim_step, res in enumerate(result):
        for i, coef in enumerate(res):
            dico[f"coef_{i}"].append(coef)
    return dico


def read_results(dico):
    plt.figure(dpi=200)
    plt.ylabel("Coefficients of B_tensor")
    plt.xlabel('Optimization Steps')
    plt.title("Optimization of the fidelity O")
    for key in dico.keys():
        plt.plot(range(len(dico[key])), dico[key], markersize=3, label=f"{key}")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    file = open(path+"output/output", "rb")
    dico = make_dico(file=file)
    read_results(dico=dico)