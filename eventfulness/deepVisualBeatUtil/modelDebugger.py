from enum import Enum
import torch

class HookState(Enum):
    FORWARD = 1
    BACKWARD = 2
    ACTIVATION = 4

class Hook():
    def __init__(self, name, module, state):
        self.name = name
        if (state.value & HookState.FORWARD.value):
            self.hookF = module.register_forward_hook(self.hook_forward_fn)
        else:
            self.hookF = None

        if (state.value & HookState.BACKWARD.value):
            self.hookB = module.register_backward_hook(self.hook_backward_fn)
        else:
            self.hookB = None
        self.inputDict = {}
        self.outputDict = {}
        self.moduleDict = {}

    def updateGradientDict(self, module, input, output, state):
        self.inputDict[state] = input
        self.outputDict[state] = output
        self.moduleDict[state] = module

    def hook_forward_fn(self, module, input, output):
        self.updateGradientDict(module, input, output, 'forward')

    def hook_backward_fn(self, module, input, output):
        self.updateGradientDict(module, input, output, 'backward')

    def getGrad(self, state):
        moduleDict = None
        try:
            moduleDict = self.moduleDict[state]
        except:
            moduleDict = None

        try:
            inputDict = self.inputDict[state]
        except:
            inputDict = None

        try:
            outputDict = self.outputDict[state]
        except:
            outputDict = None

        return (moduleDict, inputDict, outputDict)

    def getActivation(self, layer_name):
        return self.activation[layer_name]

    def close(self):
        if self.hookF != None:
            self.hookF.remove()
        if self.hookB != None:
            self.hookB.remove()


class Debug(object):
    def __init__(self, model, state):
        self.model = model.getModel()
        self.hooks = [Hook(layer[0], layer[1], state) for layer in list(self.model._modules.items())]

    def printParameters(self):
        print('***' * 3 + '  Network Parameters  ' + '***' * 3)
        for name, param in self.model.named_parameters():
            if ('toProbVector' in name and 'weight' in name):
                print("\t", name)
                print(param.size())
                print(param[min([param.size(0) - 1, 256]), min([param.size(1) - 1, 256]), :, :, :])

            if 'transpose' in name and 'weight' in name:
                print("\t", name)
                print(param.size())
                print(param[min([param.size(0), 256]), min([param.size(1), 256]), :, :, :])

        print('***' * 3 + ' Network Parameters  ' + '***' * 3)

    def printModuleNames(self):
        for name, param in self.model.named_parameters():
            print(f"module Name : {name} param {param.shape}")

    def printGradient(self, state='backward'):
        print('---' * 3 + '{} propogation gradient'.format(state) + '---' * 3)
        for hook in self.hooks:
            print(f"hook name: {hook.name}")
            if (('transpose' in hook.name) or
                    ('toProbVector' in hook.name) or
                    ('avg' in hook.name) or
                    ('stem' in hook.name)):
                print('---' * 3 + hook.name + ' input' + '---' * 3)
                (_, input, output) = hook.getGrad(state)
                if input!= None:
                    for grad in input:
                        print(grad.size())
                        print(grad.norm())
                        print(torch.var(grad).size())
                        print(torch.var(grad))
                print('---' * 3 + hook.name + ' output' + '---' * 3)
                if output!= None:
                    for grad in output:
                        print(grad.size())
                        print(grad.norm())
                        print(torch.var(grad).size())
                        print(torch.var(grad))
        print('---' * 3 + '{} propogation gradient'.format(state) + '---' * 3)