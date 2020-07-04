import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple

from timm.models.layers.split_attn import SplitAttnConv2d


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv3x3_1d(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv1x1_1d(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input 
    planes with distinct spatial and time axes, by performing a 2D convolution over the 
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time 
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
        spatial_stride = [1, stride[1], stride[2]]
        spatial_padding = [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride = [stride[0], 1, 1]
        temporal_padding = [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula
        # from the paper section 3.5
        intermed_channels = int(
            math.floor(
                (
                    kernel_size[0]
                    * kernel_size[1]
                    * kernel_size[2]
                    * in_channels
                    * out_channels
                )
                / (
                    kernel_size[1] * kernel_size[2] * in_channels
                    + kernel_size[0] * out_channels
                )
            )
        )

        # the spatial conv is effectively a 2D conv due to the
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(
            in_channels,
            intermed_channels,
            spatial_kernel_size,
            stride=spatial_stride,
            padding=spatial_padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()

        # the temporal conv is effectively a 1D conv, but has batch norm
        # and ReLU added inside the model constructor, not here. This is an
        # intentional design choice, to allow this module to externally act
        # identical to a standard Conv3D, so it can be reused easily in any
        # other codebase
        self.temporal_conv = nn.Conv3d(
            intermed_channels,
            out_channels,
            temporal_kernel_size,
            stride=temporal_stride,
            padding=temporal_padding,
            bias=bias,
        )

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x


class BasicBlock2plus1D(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock2plus1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = SpatioTemporalConv(inplanes, planes, 3, stride, padding=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SpatioTemporalConv(planes, planes, 3, stride=1, padding=1)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3_1d(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_1d(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class SplitAttnConv1d(nn.Module):
    """Split-Attention Conv1d
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        radix=2,
        reduction_factor=4,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm1d,
        drop_block=None,
        **kwargs,
    ):
        super(SplitAttnConv1d, self).__init__()
        self.radix = radix
        self.drop_block = drop_block
        mid_chs = out_channels * radix
        attn_chs = max(in_channels * radix // reduction_factor, 32)

        self.conv = nn.Conv1d(
            in_channels,
            mid_chs,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=groups * radix,
            bias=bias,
            **kwargs,
        )
        self.bn0 = norm_layer(mid_chs) if norm_layer is not None else None
        self.act0 = act_layer(inplace=True)
        self.fc1 = nn.Conv1d(out_channels, attn_chs, 1, groups=groups)
        self.bn1 = norm_layer(attn_chs) if norm_layer is not None else None
        self.act1 = act_layer(inplace=True)
        self.fc2 = nn.Conv1d(attn_chs, mid_chs, 1, groups=groups)
        self.rsoftmax = RadixSoftmax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.bn0 is not None:
            x = self.bn0(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act0(x)

        B, RC, W = x.shape
        if self.radix > 1:
            x = x.reshape((B, self.radix, RC // self.radix, W))
            x_gap = x.sum(dim=1)
        else:
            x_gap = x
        x_gap = F.adaptive_avg_pool1d(x_gap, 1)
        x_gap = self.fc1(x_gap)
        if self.bn1 is not None:
            x_gap = self.bn1(x_gap)
        x_gap = self.act1(x_gap)
        x_attn = self.fc2(x_gap)

        x_attn = self.rsoftmax(x_attn).view(B, -1, 1)
        if self.radix > 1:
            out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1))).sum(dim=1)
        else:
            out = x * x_attn
        return out.contiguous()


def replace_batchnorm(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.BatchNorm2d):
            setattr(model, name, nn.BatchNorm1d(layer.num_features))
        elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.Module):
            replace_batchnorm(layer)
        else:
            pass


def replace_conv(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.Conv2d):
            setattr(
                model,
                name,
                nn.Conv1d(
                    in_channels=layer.in_channels,
                    out_channels=layer.out_channels,
                    kernel_size=layer.kernel_size[0],
                    stride=layer.stride[0],
                    padding=layer.padding[0],
                    bias=False,
                ),
            )
        elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.Module):
            replace_conv(layer)
        else:
            pass


def replace_ada_avg_pool(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.AdaptiveAvgPool2d):
            setattr(model, name, nn.AdaptiveAvgPool1d(layer.output_size))
        elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.Module):
            replace_ada_avg_pool(layer)
        else:
            pass


def replace_max_pool(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.MaxPool2d):
            setattr(
                model,
                name,
                nn.MaxPool1d(
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                ),
            )
        elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.Module):
            replace_max_pool(layer)
        else:
            pass


def replace_avg_pool(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.AvgPool2d):
            setattr(
                model,
                name,
                nn.AvgPool1d(
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                ),
            )
        elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.Module):
            replace_avg_pool(layer)
        else:
            pass


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool1d(x.clamp(min=self.eps).pow(self.p), (x.size(-1))).pow(
            1.0 / self.p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


def replace_split_attention2d_to_1d(model):
    for name, layer in model.named_children():
        if isinstance(layer, SplitAttnConv2d):
            setattr(
                model,
                name,
                SplitAttnConv1d(
                    layer.conv.in_channels,
                    layer.fc1.in_channels,
                    kernel_size=layer.conv.kernel_size
                    if isinstance(layer.conv.kernel_size, int)
                    else layer.conv.kernel_size[0],
                    stride=layer.conv.stride
                    if isinstance(layer.conv.stride, int)
                    else layer.conv.stride[0],
                    padding=layer.conv.padding
                    if isinstance(layer.conv.padding, int)
                    else layer.conv.padding[0],
                    dilation=layer.conv.dilation
                    if isinstance(layer.conv.dilation, int)
                    else layer.conv.padding[0],
                    groups=layer.fc1.groups,
                    bias=layer.conv.bias,
                    radix=layer.radix,
                    drop_block=layer.drop_block,
                ),
            )
            # bn = getattr(model, name)
            # bn.weight = layer.weight
            # bn.bias = layer.bias
        elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.Module):
            replace_split_attention2d_to_1d(layer)
        else:
            pass


def replace_batchnorm2d_to_1d(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.BatchNorm2d):
            setattr(model, name, nn.BatchNorm1d(layer.num_features))
            # bn = getattr(model, name)
            # bn.weight = layer.weight
            # bn.bias = layer.bias
        elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.Module):
            replace_batchnorm2d_to_1d(layer)
        else:
            pass


def replace_conv2d_to_1d(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.Conv2d):
            setattr(
                model,
                name,
                nn.Conv1d(
                    in_channels=layer.in_channels,
                    out_channels=layer.out_channels,
                    kernel_size=layer.kernel_size
                    if isinstance(layer.kernel_size, int)
                    else layer.kernel_size[0],
                    stride=layer.stride
                    if isinstance(layer.stride, int)
                    else layer.stride[0],
                    padding=layer.padding
                    if isinstance(layer.padding, int)
                    else layer.padding[0],
                    dilation=layer.dilation
                    if isinstance(layer.dilation, int)
                    else layer.dilation[0],
                    bias=False,
                ),
            )
            # cnn = getattr(model, name)
            # cnn.weight.data = layer.weight.data[:, :, 0]

        elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.Module):
            replace_conv2d_to_1d(layer)
        else:
            pass


def replace_conv2d_to_2plus1d(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.Conv2d):
            setattr(
                model,
                name,
                SpatioTemporalConv(
                    in_channels=layer.in_channels,
                    out_channels=layer.out_channels,
                    kernel_size=layer.kernel_size
                    if isinstance(layer.kernel_size, int)
                    else layer.kernel_size[0],
                    stride=layer.stride
                    if isinstance(layer.stride, int)
                    else layer.stride[0],
                    padding=layer.padding
                    if isinstance(layer.padding, int)
                    else layer.padding[0],
                    bias=False,
                ),
            )
            # cnn = getattr(model, name)
            # cnn.weight.data = layer.weight.data[:, :, 0]

        elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.Module):
            replace_conv2d_to_2plus1d(layer)
        else:
            pass


def replace_adaptive_avg_pool2d_to_1d(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.AdaptiveAvgPool2d):
            if isinstance(layer.output_size, int):
                setattr(model, name, nn.AdaptiveAvgPool1d(layer.output_size))
            else:
                setattr(model, name, nn.AdaptiveAvgPool1d(layer.output_size[0]))
        elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.Module):
            replace_adaptive_avg_pool2d_to_1d(layer)
        else:
            pass


def replace_max_pool2d_to_1d(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.MaxPool2d):
            setattr(
                model,
                name,
                nn.MaxPool1d(
                    kernel_size=layer.kernel_size
                    if isinstance(layer.kernel_size, int)
                    else layer.kernel_size[0],
                    stride=layer.stride
                    if isinstance(layer.stride, int)
                    else layer.stride[0],
                    padding=layer.padding
                    if isinstance(layer.padding, int)
                    else layer.padding[0],
                    dilation=layer.dilation
                    if isinstance(layer.dilation, int)
                    else layer.dilation[0],
                ),
            )
        elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.Module):
            replace_max_pool2d_to_1d(layer)
        else:
            pass


def replace_avg_pool2d_to_1d(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.AvgPool2d):
            setattr(
                model,
                name,
                nn.AvgPool1d(
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                ),
            )
        elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.Module):
            replace_avg_pool2d_to_1d(layer)
        else:
            pass


def replace_batchnorm2d_to_3d(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.BatchNorm2d):
            setattr(model, name, nn.BatchNorm3d(layer.num_features))
            # bn = getattr(model, name)
            # bn.weight = layer.weight
            # bn.bias = layer.bias
        elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.Module):
            replace_batchnorm2d_to_3d(layer)
        else:
            pass


def replace_conv2d_to_3d(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.Conv2d):
            setattr(
                model,
                name,
                nn.Conv3d(
                    in_channels=layer.in_channels,
                    out_channels=layer.out_channels,
                    kernel_size=layer.kernel_size
                    if isinstance(layer.kernel_size, int)
                    else layer.kernel_size[0],
                    stride=layer.stride
                    if isinstance(layer.stride, int)
                    else layer.stride[0],
                    padding=layer.padding
                    if isinstance(layer.padding, int)
                    else layer.padding[0],
                    dilation=layer.dilation
                    if isinstance(layer.dilation, int)
                    else layer.dilation[0],
                    bias=False,
                ),
            )
            # cnn = getattr(model, name)
            # cnn.weight.data = layer.weight.data[:, :, 0]

        elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.Module):
            replace_conv2d_to_3d(layer)
        else:
            pass


def replace_adaptive_avg_pool2d_to_3d(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.AdaptiveAvgPool2d):
            if isinstance(layer.output_size, int):
                setattr(model, name, nn.AdaptiveAvgPool3d(layer.output_size))
            else:
                setattr(model, name, nn.AdaptiveAvgPool3d(layer.output_size[0]))
        elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.Module):
            replace_adaptive_avg_pool2d_to_3d(layer)
        else:
            pass


def replace_max_pool2d_to_3d(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.MaxPool2d):
            setattr(
                model,
                name,
                nn.MaxPool3d(
                    kernel_size=layer.kernel_size
                    if isinstance(layer.kernel_size, int)
                    else layer.kernel_size[0],
                    stride=layer.stride
                    if isinstance(layer.stride, int)
                    else layer.stride[0],
                    padding=layer.padding
                    if isinstance(layer.padding, int)
                    else layer.padding[0],
                    dilation=layer.dilation
                    if isinstance(layer.dilation, int)
                    else layer.dilation[0],
                ),
            )
        elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.Module):
            replace_max_pool2d_to_3d(layer)
        else:
            pass


def replace_avg_pool2d_to_3d(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.AvgPool2d):
            setattr(
                model,
                name,
                nn.AvgPool3d(
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                ),
            )
        elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.Module):
            replace_avg_pool2d_to_3d(layer)
        else:
            pass

