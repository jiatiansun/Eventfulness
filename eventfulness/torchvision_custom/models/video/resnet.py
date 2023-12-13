import torch
import torch.nn as nn

from ..utils import load_state_dict_from_url
# from ...deepVisualBeatUtil import GaussianKernelGenerator

__all__ = ['r2plus1dw0tc_18', 'r2plus1dw0tc_9','r2plus1dw1tc_18',
           'r2plus1dw2tc_18', 'r2plus1dw3tc_18', 'r2plus1dw1tc_9',]

model_urls = {
    'r3d_18': 'https://download.pytorch.org/models/r3d_18-b3b3357e.pth',
    'mc3_18': 'https://download.pytorch.org/models/mc3_18-a90a0ba3.pth',
    'r2plus1d_18': 'https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth',
}


class Conv3DSimple(nn.Conv3d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class Conv2Plus1D(nn.Sequential):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 diff_stride = False,
                 temporal_stride=1,
                 padding=1):
        if not diff_stride:
            temporal_stride = stride
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(temporal_stride, 1, 1), padding=(padding, 0, 0),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride, temporal_stride=1):
        return (temporal_stride, stride, stride)


class Conv3DNoTemporal(nn.Conv3d):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DNoTemporal, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (1, stride, stride)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None, diff_stride=False, temporal_stride=1):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        if not diff_stride:
            temporal_stride = stride

        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride, diff_stride = diff_stride, temporal_stride=temporal_stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.temporal_stride = temporal_stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):

        super(Bottleneck, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self):
        super(BasicStem, self).__init__(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """
    def __init__(self, inplanes=64):
        s = inplanes // 64
        super(R2Plus1dStem, self).__init__(
            nn.Conv3d(3, 45 * s, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(45 * s),
            nn.ReLU(inplace=True),
            nn.Conv3d(45 * s, inplanes, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            nn.BatchNorm3d(inplanes),
            nn.ReLU(inplace=True))


class VideoResNet(nn.Module):

    def __init__(self, block, conv_makers, layers,
                 stem, num_frames=400,
                 inplanes=64,
                 num_labels=1,
                 zero_init_residual=False):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNet, self).__init__()
        self.inplanes = inplanes
        self.inplanes1 = inplanes

        self.stem = stem(inplanes=inplanes)

        self.layer1 = self._make_layer(block, conv_makers[0], inplanes, layers[0], stride=1)
        # self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)
        self.layer2 = self._make_layer(block, conv_makers[1], inplanes * 2, layers[1],
                                       stride=2,diff_stride = True, temporal_stride = 1)
        self.layer3 = self._make_layer(block, conv_makers[2], inplanes * 4, layers[2], stride=2,
                                       diff_stride = True, temporal_stride = 1)
        self.layer4 = self._make_layer(block, conv_makers[3], inplanes * 8, layers[3], stride=2,
                                       diff_stride = True, temporal_stride = 1)

        # self.transposeConv1 = self.transposeConv(512, 512)
        # self.transposeConv2 = self.transposeConv(512, 512)
        # self.transposeConv3 = self.transposeConv(512, 512)


        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avgpoolSpace = nn.AdaptiveAvgPool3d((None, 1, 1))

        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        # print(f"out channels {num_labels}")
        self.toProbVector = nn.Conv3d(in_channels=inplanes * 8, out_channels=num_labels, kernel_size=[1, 1, 1])
        # self.sig = nn.Sigmoid()


        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        input_shape = x.size()
        # print(f"input {input_shape}")
        # print("input size {}\n".format(x.size()))
        x = self.stem(x)

        # print("stem convolution {}\n".format(x.size()))
        x = self.layer1(x)
        # print("after one convolution {}\n".format(x.size()))
        x = self.layer2(x)
        # print("after 2nd convolution {}\n".format(x.size()))
        x = self.layer3(x)
        # print("after 3rd convolution {}\n".format(x.size()))
        x = self.layer4(x)
        # print("after 4th convolution {}\n".format(x.size()))

        # x = self.avgpool(x)
        # Flatten the layer to fc
        # # x = x.flatten(1)
        # x = self.fc(x)

        #M odified Version
        # print("after 4 convs {}\n".format(x.size()))
        x = self.avgpoolSpace(x) # downsample in spatial dimensions
        # print(f"inplanes {self.inplanes1} after average Pooling convs {x.size()}")
        # x = self.transposeConv1(x) # transConv to T/4 x 1 x 1
        # # print("after 1st transpose Convolution {}\n".format(x.size()))
        # x = self.transposeConv2(x) # transConv to T/2 x 1 x 1
        # # print("after 2nd transpose Convolution {}\n".format(x.size()))
        # x = self.transposeConv3(x) # transConv to T x 1 x 1
        # # print("after 3rd transpose Convolution {}\n".format(x.size()))
        #
        # x = nn.functional.interpolate(x, size=[input_shape[2],1,1])
        # print("after interpolation {} \n".format(x.size()))
        x = self.toProbVector(x) # make it into a probability string
        # print("after switching from feature map to hot vector label {}\n".format(x.size()))
        # x = self.sig(x)
        # print(f"output {x.size()}")
        # print(f"input shape {x.size()}")
        # x = torch.tanh(x)
        # print(f"output shape {x.size()}")
        return x

    # labels are, still know where the labels are.
    # output score for binary!
    # visualization
    def _make_layer(self, block, conv_builder, planes, blocks, stride=1, diff_stride = False, temporal_stride=1):
        downsample = None
        if not diff_stride:
            temporal_stride = stride
        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride, temporal_stride=temporal_stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample,
                            diff_stride = diff_stride, temporal_stride=temporal_stride))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)
    
    # larger kernel_size possibly
    def transposeConv(self, inplanes, outplanes, kernel_size =(3,1,1), stride = (2,1,1), padding = (1,0,0)):
        return nn.Sequential(nn.ConvTranspose3d(inplanes, outplanes, kernel_size = kernel_size, stride=stride,padding=padding),
                             nn.ReLU(inplace=True))


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class VideoResNetW1TemporalConv(nn.Module):

    def __init__(self, block, conv_makers, layers,
                 stem, num_frames=400,
                 inplanes=64,
                 num_labels=1,
                 zero_init_residual=False):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNetW1TemporalConv, self).__init__()
        self.inplanes = inplanes

        self.stem = stem(inplanes=inplanes)

        self.layer1 = self._make_layer(block, conv_makers[0], inplanes, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], inplanes * 2, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)
        # self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, conv_makers[2], inplanes * 4, layers[2], stride=2, diff_stride = True, temporal_stride = 1)
        self.layer4 = self._make_layer(block, conv_makers[3], inplanes * 8, layers[3], stride=2, diff_stride = True, temporal_stride = 1)

        self.transposeConv1 = self.transposeConv(inplanes * 8, inplanes * 8)
        # self.transposeConv2 = self.transposeConv(512, 512)
        # self.transposeConv3 = self.transposeConv(512, 512)


        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avgpoolSpace = nn.AdaptiveAvgPool3d((None, 1, 1))

        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.toProbVector = nn.Conv3d(in_channels=inplanes * 8, out_channels=num_labels, kernel_size=[1, 1, 1])
        #self.sig = nn.Sigmoid()


        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        input_shape = x.size()
        # print("input size {}\n".format(x.size()))
        x = self.stem(x)

        # print("stem convolution {}\n".format(x.size()))
        x = self.layer1(x)
        # print("after one convolution {}\n".format(x.size()))
        x = self.layer2(x)
        # print("after 2nd convolution {}\n".format(x.size()))
        x = self.layer3(x)
        # print("after 3rd convolution {}\n".format(x.size()))
        x = self.layer4(x)
        # print("after 4th convolution {}\n".format(x.size()))

        # x = self.avgpool(x)
        # Flatten the layer to fc
        # # x = x.flatten(1)
        # x = self.fc(x)

        #M odified Version
        # print("after 4 convs {}\n".format(x.size()))
        x = self.avgpoolSpace(x) # downsample in spatial dimensions
        print("after average Pooling convs {}\n".format(x.size()))
        x = self.transposeConv1(x) # transConv to T/4 x 1 x 1
        # # print("after 1st transpose Convolution {}\n".format(x.size()))
        # x = self.transposeConv2(x) # transConv to T/2 x 1 x 1
        # # print("after 2nd transpose Convolution {}\n".format(x.size()))
        # x = self.transposeConv3(x) # transConv to T x 1 x 1
        # # print("after 3rd transpose Convolution {}\n".format(x.size()))
        #
        x = nn.functional.interpolate(x, size=[input_shape[2],1,1])
        # print("after interpolation {} \n".format(x.size()))
        x = self.toProbVector(x) # make it into a probability string
        # print("after switching from feature map to hot vector label {}\n".format(x.size()))
        #x = self.sig(x)

        return x

    # labels are, still know where the labels are.
    # output score for binary!
    # visualization
    def _make_layer(self, block, conv_builder, planes, blocks, stride=1, diff_stride=False, temporal_stride=1):
        downsample = None
        if not diff_stride:
            temporal_stride = stride
        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride, temporal_stride=temporal_stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample,
                            diff_stride=diff_stride, temporal_stride=temporal_stride))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    # larger kernel_size possibly
    def transposeConv(self, inplanes, outplanes, kernel_size =(3,1,1), stride = (2,1,1), padding = (1,0,0)):
        return nn.Sequential(nn.ConvTranspose3d(inplanes, outplanes, kernel_size = kernel_size, stride=stride,padding=padding),
                             nn.ReLU(inplace=True))


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VideoResNetW2TemporalConv(nn.Module):

    def __init__(self, block, conv_makers, layers,
                 stem, num_frames=400,
                 inplanes=64,
                 num_labels = 1,
                 zero_init_residual=False):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNetW2TemporalConv, self).__init__()
        self.inplanes = inplanes

        self.stem = stem(inplanes=inplanes)

        self.layer1 = self._make_layer(block, conv_makers[0], inplanes, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], inplanes * 4, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)
        # self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=1)
        # self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2, diff_stride = True, temporal_stride = 1)
        self.layer4 = self._make_layer(block, conv_makers[3], inplanes * 8, layers[3], stride=2, diff_stride = True, temporal_stride = 1)

        self.transposeConv1 = self.transposeConv(inplanes * 8, inplanes * 8)
        self.transposeConv2 = self.transposeConv(inplanes * 8, inplanes * 8)
        # self.transposeConv3 = self.transposeConv(512, 512)


        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avgpoolSpace = nn.AdaptiveAvgPool3d((None, 1, 1))

        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.toProbVector = nn.Conv3d(in_channels=inplanes * 8, out_channels=num_labels, kernel_size=[1, 1, 1])
        #self.sig = nn.Sigmoid()


        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        input_shape = x.size()
        # print("input size {}\n".format(x.size()))
        x = self.stem(x)

        # print("stem convolution {}\n".format(x.size()))
        x = self.layer1(x)
        # print("after one convolution {}\n".format(x.size()))
        x = self.layer2(x)
        # print("after 2nd convolution {}\n".format(x.size()))
        x = self.layer3(x)
        # print("after 3rd convolution {}\n".format(x.size()))
        x = self.layer4(x)
        # print("after 4th convolution {}\n".format(x.size()))

        # x = self.avgpool(x)
        # Flatten the layer to fc
        # # x = x.flatten(1)
        # x = self.fc(x)

        #M odified Version
        # print("after 4 convs {}\n".format(x.size()))
        x = self.avgpoolSpace(x) # downsample in spatial dimensions
        # print("after average Pooling convs {}\n".format(x.size()))
        x = self.transposeConv1(x) # transConv to T/4 x 1 x 1
        # # print("after 1st transpose Convolution {}\n".format(x.size()))
        x = self.transposeConv2(x) # transConv to T/2 x 1 x 1
        # # print("after 2nd transpose Convolution {}\n".format(x.size()))
        # x = self.transposeConv3(x) # transConv to T x 1 x 1
        # # print("after 3rd transpose Convolution {}\n".format(x.size()))
        #
        x = nn.functional.interpolate(x, size=[input_shape[2],1,1])
        # print("after interpolation {} \n".format(x.size()))
        x = self.toProbVector(x) # make it into a probability string
        # print("after switching from feature map to hot vector label {}\n".format(x.size()))
        #x = self.sig(x)

        return x

    # labels are, still know where the labels are.
    # output score for binary!
    # visualization
    def _make_layer(self, block, conv_builder, planes, blocks, stride=1, diff_stride=False, temporal_stride=1):
        downsample = None
        if not diff_stride:
            temporal_stride = stride
        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride, temporal_stride=temporal_stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample,
                            diff_stride=diff_stride, temporal_stride=temporal_stride))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    # larger kernel_size possibly
    def transposeConv(self, inplanes, outplanes, kernel_size =(3,1,1), stride = (2,1,1), padding = (1,0,0)):
        return nn.Sequential(nn.ConvTranspose3d(inplanes, outplanes, kernel_size = kernel_size, stride=stride,padding=padding),
                             nn.ReLU(inplace=True))


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class VideoResNetW3TemporalConv(nn.Module):

    def __init__(self, block, conv_makers, layers,
                 stem, num_frames=400,
                 inplanes = 64,
                 num_labels = 1,
                 zero_init_residual=False):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNetW3TemporalConv, self).__init__()
        self.inplanes = inplanes

        self.stem = stem(inplanes=inplanes)

        self.layer1 = self._make_layer(block, conv_makers[0], inplanes, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], inplanes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], inplanes * 8, layers[3], stride=2)
        # self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=1)
        # self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2, diff_stride = True, temporal_stride = 1)
        # self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2, diff_stride = True, temporal_stride = 1)

        self.transposeConv1 = self.transposeConv(inplanes * 8, inplanes * 8)
        self.transposeConv2 = self.transposeConv(inplanes * 8, inplanes * 8)
        self.transposeConv3 = self.transposeConv(inplanes * 8, inplanes * 8)


        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avgpoolSpace = nn.AdaptiveAvgPool3d((None, 1, 1))

        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.toProbVector = nn.Conv3d(in_channels=inplanes * 8, out_channels=num_labels, kernel_size=[1, 1, 1])
        #self.sig = nn.Sigmoid()


        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        input_shape = x.size()
        # print("input size {}\n".format(x.size()))
        x = self.stem(x)

        # print("stem convolution {}\n".format(x.size()))
        x = self.layer1(x)
        # print("after one convolution {}\n".format(x.size()))
        x = self.layer2(x)
        # print("after 2nd convolution {}\n".format(x.size()))
        x = self.layer3(x)
        # print("after 3rd convolution {}\n".format(x.size()))
        x = self.layer4(x)
        # print("after 4th convolution {}\n".format(x.size()))

        # x = self.avgpool(x)
        # Flatten the layer to fc
        # # x = x.flatten(1)
        # x = self.fc(x)

        #M odified Version
        # print("after 4 convs {}\n".format(x.size()))
        x = self.avgpoolSpace(x) # downsample in spatial dimensions
        # print("after average Pooling convs {}\n".format(x.size()))
        x = self.transposeConv1(x) # transConv to T/4 x 1 x 1
        # # print("after 1st transpose Convolution {}\n".format(x.size()))
        x = self.transposeConv2(x) # transConv to T/2 x 1 x 1
        # # print("after 2nd transpose Convolution {}\n".format(x.size()))
        x = self.transposeConv3(x) # transConv to T x 1 x 1
        # # print("after 3rd transpose Convolution {}\n".format(x.size()))
        #
        x = nn.functional.interpolate(x, size=[input_shape[2],1,1])
        # print("after interpolation {} \n".format(x.size()))
        x = self.toProbVector(x) # make it into a probability string
        # print("after switching from feature map to hot vector label {}\n".format(x.size()))
        #x = self.sig(x)

        return x

    # labels are, still know where the labels are.
    # output score for binary!
    # visualization
    def _make_layer(self, block, conv_builder, planes, blocks, stride=1, diff_stride=False, temporal_stride=1):
        downsample = None
        if not diff_stride:
            temporal_stride = stride
        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride, temporal_stride=temporal_stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample,
                            diff_stride=diff_stride, temporal_stride=temporal_stride))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    # larger kernel_size possibly
    def transposeConv(self, inplanes, outplanes, kernel_size =(3,1,1), stride = (2,1,1), padding = (1,0,0)):
        return nn.Sequential(nn.ConvTranspose3d(inplanes, outplanes, kernel_size = kernel_size, stride=stride,padding=padding),
                             nn.ReLU(inplace=True))


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



def _video_resnet(arch, pretrained=False, progress=True, tc_num=0, num_labels = 1, **kwargs):
    model = None
    print(f"num labels in video_resnet {num_labels}")
    if tc_num == 1:
        model = VideoResNetW1TemporalConv(num_labels=num_labels,**kwargs)
    # elif tc_num == 1 and num_labels > 1:
    #     model = VideoResNetW1TemporalConvWFrequencyFilter(num_labels = num_labels,**kwargs)
    elif tc_num == 2:
        model = VideoResNetW2TemporalConv(num_labels=num_labels,**kwargs)
    elif tc_num == 3:
        model = VideoResNetW3TemporalConv(num_labels=num_labels,**kwargs)
    else:
        model = VideoResNet(num_labels=num_labels, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, False)
    return model


def r3d_18(pretrained=False, progress=True, **kwargs):
    """Construct 18 layer Resnet3D model as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R3D-18 network
    """

    return _video_resnet('r3d_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DSimple] * 4,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)


def mc3_18(pretrained=False, progress=True, **kwargs):
    """Constructor for 18 layer Mixed Convolution network as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: MC3 Network definition
    """
    return _video_resnet('mc3_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DSimple] + [Conv3DNoTemporal] * 3,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)


def r2plus1dw0tc_18(pretrained=False, progress=True, **kwargs):
    """Constructor for the 18 layer deep R(2+1)D network as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R(2+1)D-18 network
    """
    print(f"is pretrained {pretrained}")
    return _video_resnet('r2plus1d_18',
                         pretrained, progress,
                         tc_num=0,
                         block=BasicBlock,
                         conv_makers=[Conv2Plus1D] * 4,
                         layers=[2, 2, 2, 2],
                         stem=R2Plus1dStem, **kwargs)

def r2plus1dw0tc_9(pretrained=False, progress=True, **kwargs):
    """Constructor for the 9 layer deep R(2+1)D network as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R(2+1)D-9 network
    """
    """
    64 x 64
    reduce the number of layers, features per layer 
    """
    print(f"initial kwargs {kwargs}")
    return _video_resnet('r2plus1d_18',
                         pretrained, progress,
                         tc_num=0,
                         block=BasicBlock,
                         conv_makers=[Conv2Plus1D] * 4,
                         layers=[1, 1, 1, 1],
                         stem=R2Plus1dStem, **kwargs)

def r2plus1dw1tc_18(pretrained=False, progress=True, **kwargs):
    return _video_resnet('r2plus1d_18',
                  pretrained, progress,
                  tc_num=1,
                  block=BasicBlock,
                  conv_makers=[Conv2Plus1D] * 4,
                  layers=[2, 2, 2, 2],
                  stem=R2Plus1dStem, **kwargs)

def r2plus1dw2tc_18(pretrained=False, progress=True, **kwargs):
    return _video_resnet('r2plus1d_18',
                  pretrained, progress,
                  tc_num=2,
                  block=BasicBlock,
                  conv_makers=[Conv2Plus1D] * 4,
                  layers=[2, 2, 2, 2],
                  stem=R2Plus1dStem, **kwargs)

def r2plus1dw3tc_18(pretrained=False, progress=True, **kwargs):
    return _video_resnet('r2plus1d_18',
                  pretrained, progress,
                  tc_num=3,
                  block=BasicBlock,
                  conv_makers=[Conv2Plus1D] * 4,
                  layers=[2, 2, 2, 2],
                  stem=R2Plus1dStem, **kwargs)

def r2plus1dw1tc_9(pretrained=False, progress=True, **kwargs):
    return _video_resnet('r2plus1d_18',
                  pretrained, progress,
                  tc_num=1,
                  block=BasicBlock,
                  conv_makers=[Conv2Plus1D] * 4,
                  layers=[1, 1, 1, 1],
                  stem=R2Plus1dStem, **kwargs)
