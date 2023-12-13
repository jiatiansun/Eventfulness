import numpy as np
import torch
from matplotlib import pyplot as plt

class GaussianKernelGenerator:
    def __init__(self, d, min_sig, max_sig, N, power=False, debug=False):
        assert(isinstance(d, int))
        assert(isinstance(N, int))
        self.N = N
        self.d = d
        self.min_sig = min_sig
        self.max_sig = max_sig
        self.debug = debug

        self.power = power
        if power:
            self.max_sig = self.min_sig * (2 ** (N-1))
            self.d = int(np.ceil(self.max_sig * 3)) * 2 + 1
            self.initPowerKernels()
        else:
            self.initKernel()

    def initKernel(self):
        allKernels = np.zeros((self.N, self.d))
        sigs = self.sigs()
        for i in range(self.N):
            curr_kernel = GaussianKernelGenerator.single1DGaussian(self.d, sigs[i]) # * self.weights[i]
            allKernels[i] = curr_kernel
        self.kernels = torch.from_numpy(allKernels).to(torch.float32)

    def initPowerKernels(self):
        allKernels = np.zeros((self.N, self.d))
        sigs = self.powerSigs()
        for i in range(self.N):
            curr_kernel = GaussianKernelGenerator.single1DGaussian(self.d, sigs[i]) # * self.weights[i]
            allKernels[i] = curr_kernel

        self.kernels = torch.from_numpy(allKernels).to(torch.float32)

    def powerSigs(self):
        return np.power(2, np.arange(self.N)) * self.min_sig

    def sigs(self):
        if self.power:
            return self.powerSigs()
        else:
            return np.linspace(self.min_sig, self.max_sig, self.N)

    def numpy_kernels(self):
        return self.kernels.numpy()

    def torch_kernels(self):
        return self.kernels

    def plotKernels(self):
        plt.figure()
        sigs = self.sigs()
        for i in range(len(self.kernels)):
            plt.plot(self.kernels[i], label=f"sig {sigs[i]}")

        plt.title(f"diameter {self.d}")
        print(f"saving kernels")
        plt.savefig("kernelPlot.png")

    @staticmethod
    def convolve(signal, kernels):
        assert(signal.dim() == 1 or (signal.dim() == 2 and (signal.size(0) == 1 or signal.size(1) == 1)))
        assert(kernels.dim() <= 2)
        assert(kernels.dim() >= 1)


        if kernels.dim() == 1:
            out_channel = 1
            padding = int(kernels.size(0) // 2)
        else:
            out_channel = kernels.size(0)
            padding = int(kernels.size(1) // 2)

        convolved = torch.nn.functional.conv1d(signal.view(1, 1, -1), kernels.view(out_channel, 1, -1),
                                               padding=padding)
        return torch.squeeze(convolved)

    @staticmethod
    def convolve_np(signal, kernels):
        convolved = np.zeros((kernels.shape[0], signal.size))
        for i in range(len(kernels)):
            kernel = kernels[i]
            convolved[i] = np.convolve(signal, kernel, 'same')
        return convolved

    @staticmethod
    def single1DGaussian(d, sig):
        sample = np.linspace(-(d - 1) / 2.0, (d-1)/2.0, d)
        gauss_prob = np.exp(- 0.5 * sample * sample / np.square(sig))
        gauss_prob = gauss_prob/ np.sum(gauss_prob)

        return gauss_prob

