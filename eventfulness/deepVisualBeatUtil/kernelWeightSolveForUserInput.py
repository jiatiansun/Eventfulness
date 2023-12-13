import numpy as np
from matplotlib import pyplot as plt
from gaussianFilterGenerator import GaussianKernelGenerator

class KernelWeightSolver():

    @staticmethod
    def kernelWeightSolve(fullbasis, targetIdx, targetValue):
        basis = fullbasis[:, targetIdx]
        A = np.transpose(basis)
        sol, res, _, _ = np.linalg.lstsq(A, targetValue)
        return sol, res

    @staticmethod
    def plot(fullbasis, weight, groundTruth, targetIdx, targetValue):
        target = groundTruth
        target[targetIdx] = targetValue
        weightedKernel = np.matmul(weight, fullbasis)

        x = np.arange(fullbasis.shape[1])
        plt.figure()
        plt.plot(x, weightedKernel, label="reconstructed from basis")
        plt.plot(x, target, label="target user input")
        plt.legend()
        plt.grid(True)
        plt.savefig("test_neighbor.png", dpi=300)
        plt.close("all")


