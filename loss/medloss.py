from __future__ import annotations

import numpy as np
from skimage.morphology import erosion, dilation
from skimage.measure import label
import torch
import warnings

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot
from monai.utils import DiceCEReduction, LossReduction, Weight, look_up_option, pytorch_after


def cal_tumor_weight_map(tumor):
    cnt_map = np.zeros_like(tumor, dtype=np.float16)
    tumor_tmp = tumor.copy()
    cnt = 0
    while np.count_nonzero(tumor) != 0:
        cnt += 1
        tumor = erosion(tumor)
        cnt_map[tumor_tmp != tumor] = cnt
        tumor_tmp = tumor
    cnt_map /= cnt
    res_map = np.power(cnt_map, 0.2)
    res_map = res_map / res_map[(res_map>0)].mean()
    return res_map


def cal_weight_map(mask):
    weight_map = np.zeros_like(mask, dtype=np.float16)
    labeled, tumor_num = label(mask, return_num=True)
    for i in range(tumor_num):
        idx = i + 1
        tumor = labeled == idx
        tumor_map = cal_tumor_weight_map(tumor)
        weight_map += tumor_map
    return weight_map


class MapDiceLoss(_Loss):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: LossReduction | str = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
    ) -> None:
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch

    def forward(self, input: torch.Tensor, target: torch.Tensor, map: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis
        if map is not None:
            intersection = torch.sum(map * target * input, dim=reduce_axis)
        else:
            intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            ground_o = torch.sum(target**2, dim=reduce_axis)
            pred_o = torch.sum(input**2, dim=reduce_axis)
        else:
            ground_o = torch.sum(target, dim=reduce_axis)
            pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f
    

class MedLoss(_Loss):
    """
    Compute both Dice loss and Cross Entropy Loss, and return the weighted sum of these two losses.
    The details of Dice loss is shown in ``monai.losses.DiceLoss``.
    The details of Cross Entropy Loss is shown in ``torch.nn.CrossEntropyLoss``. In this implementation,
    two deprecated parameters ``size_average`` and ``reduce``, and the parameter ``ignore_index`` are
    not supported.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: torch.Tensor | None = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ) -> None:
        super().__init__()
        reduction = look_up_option(reduction, DiceCEReduction).value
        self.dice = MapDiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.old_pt_ver = not pytorch_after(1, 10)

    def ce(self, input: torch.Tensor, target: torch.Tensor, map=None) -> torch.Tensor:
        """
        Compute CrossEntropy loss for the input and target.
        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?#torch.nn.CrossEntropyLoss.

        """
        num_classes = input.shape[1]
        target = target.squeeze(1).long()
        target = F.one_hot(target, num_classes=num_classes)
        target = target.movedim(-1,1).float() # (batch, channel, height, width, depth)
        if map is not None:
            target = target*map
        
        log_softmax = F.log_softmax(input, dim=1)
        loss = -torch.sum(target * log_softmax, dim=1)
        
        return loss.mean()

    def forward(self, input: torch.Tensor, target: torch.Tensor, map: torch.Tensor) -> torch.Tensor:
        # input: b, c, h, w, d; target, weight: b, 1, h, w, d
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )

        dice_loss = self.dice(input, target, map)
        ce_loss = self.ce(input, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss

        return total_loss


def custom_cross_entropy_loss(inputs, targets, weight=None):
    """
    针对五维输入和目标的自定义交叉熵损失函数，修正版。
    
    :param inputs: 预测的类别得分，形状为 (batch, channel, height, width, depth)。
    :param targets: 目标类别，形状为 (batch, 1, height, width, depth)。
    :param weight: 像素权重，形状为 (batch, 1, height, width, depth)。
    :param num_classes: 类别总数。
    :return: 交叉熵损失的平均值。
    """
    # 移除 targets 的额外维度并转换为 one-hot 编码
    num_classes = inputs.shape[1]
    targets = targets.squeeze(1)
    targets = F.one_hot(targets, num_classes=num_classes)
    targets = targets.movedim(-1,1).float() # (batch, channel, height, width, depth)
    if weight is not None:
        targets = targets*weight
    
    # 计算 log softmax
    log_softmax = F.log_softmax(inputs, dim=1)

    # 应用 one-hot 编码的 targets 选择 log softmax 值
    loss = -torch.sum(targets * log_softmax, dim=1)

    # 计算平均损失
    return loss.mean()

