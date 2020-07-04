import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable


def ohem_loss(rate, cls_pred, cls_target):
    batch_size = cls_pred.size(0)
    ohem_cls_loss = F.cross_entropy(
        cls_pred, cls_target, reduction="none", ignore_index=-1
    )

    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    keep_num = min(sorted_ohem_loss.size()[0], int(batch_size * rate))
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, in_features=2048, s=10.0, m=0.5, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = 2
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(self.out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, label=None):

        logit = F.linear(input, self.weight)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)).float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # phi = cos(theta + theta_m)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        if label is not None:
            # --------------------------- convert label to one-hot ---------------------------
            # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
            one_hot = torch.zeros(cosine.size(), device="cuda")
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
            output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine
            )  # you can use torch.where if your torch.__version__ is 0.4
            output *= self.s
            # print(output)
            loss = self.loss(output, label)
            return loss, logit
        else:
            return logit


class AdaCos(nn.Module):
    def __init__(self, num_features=2048, num_classes=2, m=0.50):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(2)
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(
                one_hot < 1,
                torch.exp(self.s * logits.float()),
                torch.zeros_like(logits).float(),
            )
            B_avg = torch.sum(B_avg) / input.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(
                torch.min(math.pi / 4 * torch.ones_like(theta_med), theta_med)
            )
        print(self.s)
        output = self.s * logits
        print(output)
        loss = self.loss(output, label.long())

        return loss, output


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class NormalizedMAE(nn.Module):
    def __init__(self, weights=[0.3, 0.175, 0.175, 0.175, 0.175]):
        super(NormalizedMAE, self).__init__()
        self.weights = torch.tensor(weights, requires_grad=False)
        # 訓練データから算出した平均値
        # self.norm = torch.tensor(
        #     [
        #         50.034068314838336,
        #         51.47469215992403,
        #         59.244131909485276,
        #         47.32512986153187,
        #         51.90565840967048,
        #     ],
        #     requires_grad=False,
        # )

    def forward(self, inputs, targets):
        inputs = inputs.double()
        is_nan = torch.isnan(targets)
        inputs[is_nan] = 0
        targets[is_nan] = 0
        diff = torch.abs(inputs - targets).sum(0)
        # norm = (targets != 0).sum(0) * self.norm.to(inputs.device)

        norm = targets.sum(0)
        loss = (diff / norm) * self.weights.to(inputs.device)
        return loss.sum() / self.weights.to(inputs.device).sum()


class AuxNormalizedMAE(nn.Module):
    def __init__(self):
        super(AuxNormalizedMAE, self).__init__()
        self.nmae = NormalizedMAE()

    def forward(self, inputs_list, targets):
        auxs = inputs_list[:-1]
        out = inputs_list[-1]
        aux_loss = 0
        for aux in auxs:
            aux_loss += self.nmae(aux, targets) * 0.1
        main_loss = self.nmae(out, targets)
        loss = main_loss + aux_loss

        return main_loss


def get_weighted_binary_cross_entropy():
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([0.2]))
    return loss_fn


def get_focalloss():
    loss_fn = FocalLoss()
    return loss_fn


def get_adacos():
    loss_fn = AdaCos()
    return loss_fn


def get_arcface(in_features):
    loss_fn = ArcMarginProduct(in_features=in_features)
    return loss_fn


def get_binary_cross_entropy():
    loss_fn = nn.BCEWithLogitsLoss()
    return loss_fn


def get_any_binary_cross_entropy():
    loss_fn = nn.BCEWithLogitsLoss()
    return loss_fn


def get_weighted_cross_entropy():
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    return loss_fn


def get_normilized_mae(weights=None):
    loss_fn = NormalizedMAE(weights)
    return loss_fn


def get_mse(weights=None):
    loss_fn = nn.MSELoss()
    return loss_fn


def get_aux_normilized_mae():
    loss_fn = AuxNormalizedMAE()
    return loss_fn


def get_loss(loss_name, **params):
    print("loss name:", loss_name)
    f = globals().get("get_" + loss_name)
    return f(**params)
