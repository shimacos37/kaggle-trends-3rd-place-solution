import copy
from typing import Optional

import dgl
import numpy as np
import pretrainedmodels
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv
from dgl import function as fn
from dgl.utils import expand_as_pair

from efficientnet_pytorch import EfficientNet

from .module import (
    SpatioTemporalConv,
    BasicBlock2plus1D,
    replace_adaptive_avg_pool2d_to_1d,
    replace_avg_pool2d_to_1d,
    replace_batchnorm2d_to_1d,
    replace_conv2d_to_1d,
    replace_max_pool2d_to_1d,
    replace_split_attention2d_to_1d,
    replace_adaptive_avg_pool2d_to_3d,
    replace_avg_pool2d_to_3d,
    replace_batchnorm2d_to_3d,
    replace_conv2d_to_3d,
    replace_max_pool2d_to_3d,
)


class CNN1D(nn.Module):
    def __init__(self, model_config):
        super(CNN1D, self).__init__()
        self.model_config = model_config
        if "resnest" not in model_config.backbone:
            self.cnn = pretrainedmodels.__dict__[model_config.backbone](
                num_classes=1000, pretrained=None
            )
            if "inception" in self.model_config.backbone:
                conv0 = self.cnn.conv2d_1a.conv
                self.cnn.conv2d_1a.conv = nn.Conv2d(
                    in_channels=model_config.in_channels,
                    out_channels=conv0.out_channels,
                    kernel_size=conv0.kernel_size,
                    stride=conv0.stride,
                    padding=conv0.padding,
                    bias=conv0.bias,
                )

            elif "vgg" in self.model_config.backbone:
                conv0 = self.cnn._features
                self.cnn.conv1 = nn.Conv2d(
                    in_channels=model_config.in_channels,
                    out_channels=conv0.out_channels,
                    kernel_size=conv0.kernel_size,
                    stride=conv0.stride,
                    padding=conv0.padding,
                    bias=conv0.bias,
                )
            elif "resnet" in self.model_config.backbone:
                conv0 = self.cnn.conv1
                self.cnn.conv1 = nn.Conv2d(
                    in_channels=model_config.in_channels,
                    out_channels=conv0.out_channels,
                    kernel_size=conv0.kernel_size,
                    stride=conv0.stride,
                    padding=conv0.padding,
                    bias=conv0.bias,
                )

            elif "se_" in self.model_config.backbone:
                conv0 = self.cnn.layer0.conv1
                self.cnn.layer0.conv1 = nn.Conv2d(
                    in_channels=model_config.in_channels,
                    out_channels=conv0.out_channels,
                    kernel_size=conv0.kernel_size,
                    stride=conv0.stride,
                    padding=conv0.padding,
                    bias=conv0.bias,
                )

            elif "densenet" in self.model_config.backbone:

                conv0 = self.cnn.features.conv0
                self.cnn.features.conv0 = nn.Conv2d(
                    in_channels=model_config.in_channels,
                    out_channels=conv0.out_channels,
                    kernel_size=conv0.kernel_size,
                    stride=conv0.stride,
                    padding=conv0.padding,
                    bias=conv0.bias,
                )

            self.in_features = self.cnn.last_linear.in_features

        else:
            self.cnn = timm.create_model(
                model_config.backbone,
                pretrained=False,
                num_classes=model_config.num_classes,
                in_chans=53,
            )
            self.in_features = self.cnn.fc.in_features // 2
            # assert in_channels % 32 == 0
            replace_split_attention2d_to_1d(self.cnn)

        # For DDP Error
        self.cnn.logits = self.logits
        self.last_linear = nn.Linear(self.in_features, model_config.num_classes)

        replace_batchnorm2d_to_1d(self.cnn)
        replace_conv2d_to_1d(self.cnn)
        replace_adaptive_avg_pool2d_to_1d(self.cnn)
        replace_avg_pool2d_to_1d(self.cnn)
        replace_max_pool2d_to_1d(self.cnn)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def logits(self, x):
        feature = F.adaptive_avg_pool1d(x, 1)
        feature = feature.view(feature.size(0), -1)

        return feature

    def forward(self, **batch):
        """
        Input:
            data (torch.Tensor): shape [bs, n_components, n_parcellation]
        """
        if self.model_config.in_channels == 400:
            imgs = batch["data"].permute(0, 2, 1)
        else:
            imgs = batch["data"]
        if "resnest" not in self.model_config.backbone:
            feature = self.cnn.features(imgs)
            self.feature = self.logits(feature)
        else:
            # feature = self.cnn.forward_features(imgs)
            x = self.cnn.conv1(imgs)
            x = self.cnn.bn1(x)
            x = self.cnn.act1(x)
            x = self.cnn.maxpool(x)

            x = self.cnn.layer1(x)
            x = self.cnn.layer2(x)
            x = self.cnn.layer3(x)
            self.feature = self.logits(x)
        out = self.last_linear(self.feature)

        return out

    def get_feature(self):
        return self.feature


class CNN2D(nn.Module):
    def __init__(self, model_config):
        super(CNN2D, self).__init__()
        self.model_config = model_config
        if "efficientnet" not in model_config.backbone:
            self.cnn = pretrainedmodels.__dict__[model_config.backbone](
                num_classes=1000, pretrained="imagenet"
            )
            if "inception" in self.model_config.backbone:
                conv0 = self.cnn.conv2d_1a.conv
                self.cnn.conv2d_1a.conv = nn.Conv2d(
                    in_channels=48,
                    out_channels=conv0.out_channels,
                    kernel_size=conv0.kernel_size,
                    stride=conv0.stride,
                    padding=conv0.padding,
                    bias=conv0.bias,
                )

            elif "vgg" in self.model_config.backbone:
                conv0 = self.cnn._features
                self.cnn.conv1 = nn.Conv2d(
                    in_channels=48,
                    out_channels=conv0.out_channels,
                    kernel_size=conv0.kernel_size,
                    stride=conv0.stride,
                    padding=conv0.padding,
                    bias=conv0.bias,
                )
            elif "resnet" in self.model_config.backbone:
                conv0 = self.cnn.conv1
                self.cnn.conv1 = nn.Conv2d(
                    in_channels=48,
                    out_channels=conv0.out_channels,
                    kernel_size=conv0.kernel_size,
                    stride=conv0.stride,
                    padding=conv0.padding,
                    bias=conv0.bias,
                )

            elif "se_" in self.model_config.backbone:
                conv0 = self.cnn.layer0.conv1
                self.cnn.layer0.conv1 = nn.Conv2d(
                    in_channels=48,
                    out_channels=conv0.out_channels,
                    kernel_size=conv0.kernel_size,
                    stride=conv0.stride,
                    padding=conv0.padding,
                    bias=conv0.bias,
                )

            elif "densenet" in self.model_config.backbone:

                conv0 = self.cnn.features.conv0
                self.cnn.features.conv0 = nn.Conv2d(
                    in_channels=48,
                    out_channels=conv0.out_channels,
                    kernel_size=conv0.kernel_size,
                    stride=conv0.stride,
                    padding=conv0.padding,
                    bias=conv0.bias,
                )

            self.in_features = self.cnn.last_linear.in_features

        else:
            self.cnn = EfficientNet.from_pretrained(model_config.backbone)
            conv0 = self.cnn._conv_stem
            self.cnn._conv_stem = nn.Conv2d(
                in_channels=48,
                out_channels=conv0.out_channels,
                kernel_size=conv0.kernel_size,
                stride=conv0.stride,
                padding=conv0.padding,
                bias=conv0.bias,
            )
            self.in_features = self.cnn._fc.in_features
        # For DDP Error
        self.cnn.logits = self.logits

        self.cnn.last_linear = nn.Linear(self.in_features, model_config.num_classes)

    def features(self, input):
        if "efficientnet" in self.model_config.backbone:
            x = self.cnn.extract_features(input)
        else:
            x = self.cnn.features(input)
        return x

    def logits(self, x):
        feature = F.adaptive_avg_pool2d(x, 1)
        feature = feature.view(feature.size(0), -1)

        return feature

    def forward(self, **batch):
        """
        Input:
            data (torch.Tensor): shape [bs, height, width, depth]
        """
        imgs = batch["data"]
        # spatial feature
        feature = self.features(imgs)
        feature = self.logits(feature)
        out = self.cnn.last_linear(feature)

        return out

    def get_feature(self):
        return self.feature


class CNN3D(nn.Module):
    def __init__(self, model_config):
        super(CNN3D, self).__init__()
        self.model_config = model_config
        self.cnn = pretrainedmodels.__dict__[model_config.backbone](
            num_classes=1000, pretrained=None
        )
        if "inception" in self.model_config.backbone:
            conv0 = self.cnn.conv2d_1a.conv
            self.cnn.conv2d_1a.conv = nn.Conv2d(
                in_channels=model_config.in_channels,
                out_channels=conv0.out_channels,
                kernel_size=conv0.kernel_size,
                stride=conv0.stride,
                padding=conv0.padding,
                bias=conv0.bias,
            )

        elif "vgg" in self.model_config.backbone:
            conv0 = self.cnn._features
            self.cnn.conv1 = nn.Conv2d(
                in_channels=model_config.in_channels,
                out_channels=conv0.out_channels,
                kernel_size=conv0.kernel_size,
                stride=conv0.stride,
                padding=conv0.padding,
                bias=conv0.bias,
            )
        elif "resnet" in self.model_config.backbone:
            conv0 = self.cnn.conv1
            self.cnn.conv1 = nn.Conv2d(
                in_channels=model_config.in_channels,
                out_channels=conv0.out_channels,
                kernel_size=conv0.kernel_size,
                stride=conv0.stride,
                padding=conv0.padding,
                bias=conv0.bias,
            )

        elif "se_" in self.model_config.backbone:
            conv0 = self.cnn.layer0.conv1
            self.cnn.layer0.conv1 = nn.Conv2d(
                in_channels=model_config.in_channels,
                out_channels=conv0.out_channels,
                kernel_size=conv0.kernel_size,
                stride=conv0.stride,
                padding=conv0.padding,
                bias=conv0.bias,
            )

        elif "densenet" in self.model_config.backbone:

            conv0 = self.cnn.features.conv0
            self.cnn.features.conv0 = nn.Conv2d(
                in_channels=model_config.in_channels,
                out_channels=conv0.out_channels,
                kernel_size=conv0.kernel_size,
                stride=conv0.stride,
                padding=conv0.padding,
                bias=conv0.bias,
            )
        self.in_features = self.cnn.last_linear.in_features
        self.cnn.last_linear = nn.Linear(self.in_features, model_config.num_classes)
        # For DDP Error
        self.cnn.logits = self.logits

        replace_batchnorm2d_to_3d(self.cnn)
        replace_conv2d_to_3d(self.cnn)
        replace_adaptive_avg_pool2d_to_3d(self.cnn)
        replace_avg_pool2d_to_3d(self.cnn)
        replace_max_pool2d_to_3d(self.cnn)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def logits(self, x):
        feature = F.adaptive_avg_pool3d(x, 1)
        feature = feature.view(feature.size(0), -1)

        return feature

    def forward(self, **batch):
        """
        Input:
            data (torch.Tensor): shape [bs, n_components, height, width, depth]
        """
        imgs = batch["data"]
        feature = self.cnn.features(imgs)
        self.feature = self.logits(feature)
        out = self.cnn.last_linear(self.feature)

        return out

    def get_feature(self):
        return self.feature


class SimpleCNN3D(nn.Module):
    def __init__(self, model_config):
        super(SimpleCNN3D, self).__init__()
        norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer
        self.dilation = 1
        self.inplanes = 32
        self.base_width = 64

        self.conv1 = nn.Conv3d(53, self.inplanes, 5, 2, padding=0, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=1)
        self.conv2 = nn.Conv3d(
            self.inplanes, self.inplanes, 3, 1, padding=0, bias=False
        )
        self.bn2 = norm_layer(self.inplanes)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=1)
        self.conv3 = nn.Conv3d(
            self.inplanes, self.inplanes, 3, 1, padding=0, bias=False
        )
        self.bn3 = norm_layer(self.inplanes)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=1)

        self.last_linear = nn.Linear(576, model_config.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, **batch):
        """
        Input:
            frames (torch.Tensor): shape [bs, timestep, height, width, n_components]
        """
        imgs = batch["data"]
        out = self.conv1(imgs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.maxpool3(out)
        self.feature = out.view(out.shape[0], -1)

        out = self.last_linear(self.feature)

        return out

    def get_feature(self):
        return self.feature


class CNN2plus1D(nn.Module):
    def __init__(self, model_config):
        super(CNN2plus1D, self).__init__()
        norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer
        self.dilation = 1
        self.inplanes = 64
        self.base_width = 64
        block = BasicBlock2plus1D

        self.conv1 = SpatioTemporalConv(53, self.inplanes, 7, 2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.inplanes, 1)
        self.layer2 = self._make_layer(block, self.inplanes * 2, 1, stride=2)
        self.layer3 = self._make_layer(
            block, self.inplanes * 2, 1, stride=2, dilate=True
        )
        self.layer4 = self._make_layer(
            block, self.inplanes * 2, 1, stride=2, dilate=True
        )

        self.last_linear = nn.Linear(self.inplanes, model_config.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SpatioTemporalConv(self.inplanes, planes * block.expansion, 1, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def logits(self, x):
        feature = F.adaptive_avg_pool3d(x, (1, 1, 1))
        feature = feature.view(feature.size(0), -1)

        return feature

    def features(self):
        return self.feature

    def forward(self, **batch):
        """
        Input:
            data (torch.Tensor): shape [bs, n_components, height, width, depth]
        """
        imgs = batch["data"]
        out = self.conv1(imgs)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        self.feature = self.logits(out)

        out = self.last_linear(self.feature)

        return out

    def get_feature(self):
        return self.feature


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: nn.Module,
        dropout_rate: float = 0.0,
        use_norm=False,
    ):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation()
        self.dropout = nn.Dropout(dropout_rate)
        if use_norm:
            self.norm_layer = nn.BatchNorm1d(out_features)
        else:
            self.norm_layer = None

    def forward(self, x, is_residual=False):
        out = self.linear(x)
        if self.norm_layer:
            out = self.norm_layer(out)
        out = self.dropout(self.activation(out))
        if is_residual:
            out = torch.cat([x, out], dim=-1)
        return out


class FNCTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(FNCTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.Linear(d_model, d_model)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu

    def forward(
        self,
        src: torch.Tensor,
        fnc_mat: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        fnc_mat : [bs, 53, 53]
        """
        attention_weight = F.softmax(fnc_mat, dim=-1)
        src2 = self.self_attn(src)
        src2 = torch.bmm(attention_weight, src2)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Transformer(nn.Module):
    def __init__(self, model_config):
        super(Transformer, self).__init__()
        self.model_config = model_config
        self.reduction = nn.Sequential(nn.Linear(400, 256), nn.ReLU())
        encoder_layer = nn.TransformerEncoderLayer(
            256,
            nhead=8,
            dim_feedforward=512,
            dropout=model_config.dropout_rate,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, model_config.num_layers)
        self.last_linear = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, model_config.num_classes),
        )

    def forward(self, **batch):
        """
        Input:
            x (torch.Tensor): shape [bs, n_components, n_parcellation]
        """
        time_feature = batch["data"]
        bs, n_components, n_parcellation = time_feature.shape
        time_feature = self.reduction(time_feature.view(-1, n_parcellation))
        time_feature = time_feature.view(
            bs, n_components, time_feature.shape[-1]
        ).permute(1, 0, 2)
        time_feature = self.transformer(time_feature)
        avg = torch.mean(time_feature, dim=0)
        max, _ = torch.max(time_feature, dim=0)
        feature = torch.cat([avg, max], dim=-1)
        out = self.last_linear(feature)

        return out

    def get_feature(self):
        return self.feature


class GINConv(nn.Module):
    r"""Graph Isomorphism Network layer from paper `How Powerful are Graph
    Neural Networks? <https://arxiv.org/pdf/1810.00826.pdf>`__.

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{aggregate}\left(\left\{h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)

    Parameters
    ----------
    apply_func : callable activation function/layer or None
        If not None, apply this function to the updated node feature,
        the :math:`f_\Theta` in the formula.
    aggregator_type : str
        Aggregator type to use (``sum``, ``max`` or ``mean``).
    init_eps : float, optional
        Initial :math:`\epsilon` value, default: ``0``.
    learn_eps : bool, optional
        If True, :math:`\epsilon` will be a learnable parameter.
    """

    def __init__(self, apply_func, aggregator_type, init_eps=0, learn_eps=False):
        super(GINConv, self).__init__()
        self.apply_func = apply_func
        if aggregator_type == "sum":
            self._reducer = fn.sum
        elif aggregator_type == "max":
            self._reducer = fn.max
        elif aggregator_type == "mean":
            self._reducer = fn.mean
        else:
            raise KeyError("Aggregator type {} not recognized.".format(aggregator_type))
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([init_eps]))

    def forward(self, graph, feat):
        r"""Compute Graph Isomorphism Network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.
            If ``apply_func`` is not None, :math:`D_{in}` should
            fit the input dimensionality requirement of ``apply_func``.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is the output dimensionality of ``apply_func``.
            If ``apply_func`` is None, :math:`D_{out}` should be the same
            as input dimensionality.
        """
        graph = graph.local_var()
        feat_src, feat_dst = expand_as_pair(feat)
        graph.srcdata["h"] = feat_src
        graph.update_all(fn.copy_u("h", "m"), self._reducer("m", "neigh"))
        rst = (1 + self.eps) * feat_dst + graph.dstdata["neigh"]
        if self.apply_func is not None:
            rst = rst.reshape(-1, rst.shape[-1])
            rst = self.apply_func(rst)
            rst = rst.reshape(53, -1, rst.shape[-1])
        return rst


class GIN(nn.Module):
    def __init__(self, model_config):
        super(GIN, self).__init__()

        adj = np.load("/root/workdir/input/fnc_adj.npy")
        self.G = dgl.DGLGraph(np.where(adj >= np.quantile(adj.reshape(-1), 0.9)))

        self.gin_conv1 = GINConv(
            apply_func=nn.Linear(400, 256), aggregator_type="sum", learn_eps=True,
        )

        self.gin_conv2 = GINConv(
            apply_func=nn.Linear(256, 256), aggregator_type="sum", learn_eps=True,
        )

        self.gin_conv3 = GINConv(
            apply_func=nn.Linear(256, 256), aggregator_type="sum", learn_eps=True,
        )

        self.loading_mlp = nn.Sequential(
            nn.Linear(26, 128), nn.LeakyReLU(), nn.Dropout(model_config.dropout_rate),
        )
        self.fnc_mlp = nn.Sequential(
            nn.Linear(1378, 512), nn.LeakyReLU(), nn.Dropout(model_config.dropout_rate),
        )

        self.last_linear = nn.Linear(256 * 3 + 128, model_config.num_classes)

    def features(self, x):
        out = self.gin_conv1(self.G, x)
        out = torch.relu(out)
        feat1 = torch.sum(out, 0)

        out = self.gin_conv2(self.G, out)
        out = torch.relu(out)
        feat2 = torch.sum(out, 0)

        out = self.gin_conv3(self.G, out)
        out = torch.relu(out)
        feat3 = torch.sum(out, 0)

        feature = torch.cat([feat1, feat2, feat3], dim=-1)
        return feature

    def forward(self, **batch):
        """
        inputs shape: [bs, n_nodes, n_feature]
        """
        inputs = batch["data"]
        loading_feature = self.loading_mlp(batch["loading"].float())
        # fnc_feature = self.fnc_mlp(batch["fnc"].float())
        feature = self.features(inputs.permute(1, 0, 2))
        self.feature = torch.cat([feature, loading_feature], dim=-1)
        out = self.last_linear(self.feature)
        return out

    def get_feature(self):
        return self.feature


class GAT(nn.Module):
    def __init__(self, model_config):
        super(GAT, self).__init__()

        # adj = np.load("/root/workdir/input/fnc_adj.npy")
        self.G = dgl.DGLGraph(np.where(np.ones([53, 53]) == 1))
        self.linear1 = nn.Linear(400, 256)

        self.gat_conv1 = GATConv(
            256, 32, num_heads=8, feat_drop=model_config.dropout_rate, residual=True
        )
        self.bn1 = nn.BatchNorm1d(64)

        self.gat_conv2 = GATConv(
            256, 32, num_heads=8, feat_drop=model_config.dropout_rate, residual=True
        )
        self.bn2 = nn.BatchNorm1d(64)

        self.gat_conv3 = GATConv(
            256, 32, num_heads=8, feat_drop=model_config.dropout_rate, residual=True
        )
        self.bn3 = nn.BatchNorm1d(64)

        self.loading_mlp = nn.Sequential(
            nn.Linear(26, 128), nn.LeakyReLU(), nn.Dropout(model_config.dropout_rate),
        )

        self.last_linear = nn.Linear(256 * 3 + 128, model_config.num_classes)

    def _sample_forward(self, x):
        out = torch.relu(self.linear1(x))
        out = self.gat_conv1(self.G, out)
        out = out.view(53, -1)
        out = torch.relu(out)
        feat1 = torch.sum(out, 0, keepdim=True)

        out = self.gat_conv2(self.G, out)
        out = out.view(53, -1)
        out = torch.relu(out)
        feat2 = torch.sum(out, 0, keepdim=True)

        out = self.gat_conv3(self.G, out)
        out = out.view(53, -1)
        out = torch.relu(out)
        feat3 = torch.sum(out, 0, keepdim=True)

        feature = torch.cat([feat1, feat2, feat3], dim=-1)
        return feature

    def forward(self, **batch):
        """
        inputs shape: [bs, n_nodes, n_feature]
        """
        inputs = batch["data"]
        self.feature = torch.cat([self._sample_forward(x) for x in inputs], dim=0)
        loading_feature = self.loading_mlp(batch["loading"].float())
        # fnc_feature = self.fnc_mlp(batch["fnc"].float())
        out = self.last_linear(torch.cat([self.feature, loading_feature], dim=-1))

        return out

    def get_feature(self):
        return self.feature


class AutoEncoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        # Encoder specification
        self.enc_cnn_1 = nn.Conv3d(53, 32, kernel_size=3)
        self.enc_cnn_2 = nn.Conv3d(32, 16, kernel_size=3)
        self.enc_cnn_3 = nn.Conv3d(16, 8, kernel_size=3)

        self.enc_pool = nn.MaxPool3d(2, stride=2, return_indices=True)

        self.enc_linear_1 = nn.Linear(480, 256)
        self.enc_linear_2 = nn.Linear(256, 256)

        self.dec_cnn_1 = nn.Conv3d(8, 16, kernel_size=3, padding=[2, 2, 2])
        self.dec_cnn_2 = nn.Conv3d(16, 32, kernel_size=3, padding=[2, 2, 2])
        self.dec_cnn_3 = nn.Conv3d(32, 53, kernel_size=3, padding=[2, 2, 2])

        self.dec_unpool1 = nn.MaxUnpool3d(kernel_size=[2, 2, 3], stride=2)
        self.dec_unpool2 = nn.MaxUnpool3d(kernel_size=[3, 3, 3], stride=2)

        self.dec_linear_1 = nn.Linear(256, 256)
        self.dec_linear_2 = nn.Linear(256, 480)

        self.debug_model = True

    def encode(self, images):

        code = self.enc_cnn_1(images)

        pool1, indices1 = self.enc_pool(code)
        code = F.selu(pool1)

        code = self.enc_cnn_2(code)

        pool2, indices2 = self.enc_pool(code)
        code = F.selu(pool2)

        code = self.enc_cnn_3(code)
        pool3, indices3 = self.enc_pool(code)
        code = F.selu(pool3)

        code = code.view([images.size(0), -1])

        code = F.selu(self.enc_linear_1(code))

        code = self.enc_linear_2(code)

        # required for unpool
        pool_par = {"P1": [indices1], "P2": [indices2], "P3": [indices3]}

        return code, pool_par

    def decode(self, code, pool_par):

        out = self.dec_linear_1(code)

        out = F.selu(self.dec_linear_2(out))
        out = out.view([out.size(0), 8, 4, 5, 3])
        out = self.dec_unpool1(out, pool_par["P3"][0])
        out = F.selu(out)
        out = self.dec_cnn_1(out)

        out = self.dec_unpool2(out, pool_par["P2"][0])
        out = F.selu(out)

        out = self.dec_cnn_2(out)
        out = F.selu(out)

        out = self.dec_unpool1(out, pool_par["P1"][0])
        out = F.selu(out)

        out = self.dec_cnn_3(out)
        out = F.selu(out)

        return out

    def forward(self, **batch):
        imgs = batch["data"].float()
        self.feature, pool_par = self.encode(imgs)
        out = self.decode(self.feature, pool_par)
        return out

    def get_feature(self):
        return self.feature


def get_1dcnn(model_config):
    model = CNN1D(model_config)
    return model


def get_2dcnn(model_config):
    model = CNN2D(model_config)
    return model


def get_simple_3dcnn(model_config):
    model = SimpleCNN3D(model_config)
    return model


def get_3dcnn(model_config):
    model = CNN3D(model_config)
    return model


def get_2plus1dcnn(model_config):
    model = CNN2plus1D(model_config)
    return model


def get_autoencoder(model_config):
    model = AutoEncoder(model_config)
    return model


def get_transformer(model_config):
    model = Transformer(model_config)
    return model


def get_auto_encoder(model_config):
    model = AutoEncoder(model_config)
    return model


def get_gin(model_config):
    model = GIN(model_config)
    return model


def get_gat(model_config):
    model = GAT(model_config)
    return model


def get_model(model_config):
    print("model name:", model_config.model_name)
    print("backbone name:", model_config.backbone)
    if "simple_resnet" in model_config.model_name:
        f = get_3dcnn
    else:
        f = globals().get("get_" + model_config.model_name)
    return f(model_config)
