#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Sep 30 21:42:33 2021

@author: mibook

metrics
"""
import cv2
import numpy as np
import torch, pdb
# from torchvision.ops import sigmoid_focal_loss
from scipy.ndimage import gaussian_filter
from timeit import default_timer as timer
from numba import jit, njit
from nnMorpho import operations
#from torchvision.transforms.functional import gaussian_blur

# https://github.com/mlyg/boundary-uncertainty/blob/main/loss_functions.py
# def border_uncertainty(seg: np.ndarray, alpha=0.9, beta=0.1):
#     res = np.zeros_like(seg)
#     check_seg = seg.astype(np.bool)
#     if check_seg.any():
#         kernel = np.ones((3,3),np.uint8)
#         im_erode = cv2.erode(seg[:, 1],kernel,iterations = 1)
#         im_dilate = cv2.dilate(seg[:, 1],kernel,iterations = 1)
#         # compute inner border and adjust certainty with alpha parameter
#         inner = seg[:, 1] - im_erode
#         inner = alpha * inner
#         # compute outer border and adjust certainty with beta parameter
#         outer = im_dilate - seg[:,1]
#         outer = beta * outer
#         # combine adjusted borders together with unadjusted image
#         combined = inner + outer + im_erode
#         combined = np.expand_dims(combined,axis=-1)

#         res = np.concatenate([1-combined, combined],axis=-1)

#         return res
#     else:
#         return res

# # @torch.jit.script
# def border_uncertainty_batch(y_true):
#     return np.array([border_uncertainty(y.cpu().numpy()) for y in y_true]).astype(np.float32)

# @njit(parallel=True)
@torch.jit.script
def compute_res(seg, im_erode, im_dilate, alpha, beta):
    # compute inner border and adjust certainty with alpha parameter
    inner = seg - im_erode
    inner = alpha * inner
    # compute outer border and adjust certainty with beta parameter
    outer = im_dilate - seg
    outer = beta * outer
    # combine adjusted borders together with unadjusted image
    res = inner + outer + im_erode
    return res

accumulated_res = 0
accumulated_res_idx = 0
# Function to calculate boundary uncertainty
def border_uncertainty_sigmoid(seg: torch.Tensor, alpha = 0.9, beta = 0.9):
    global accumulated_res, accumulated_res_idx
    """
    Parameters
    ----------
    alpha : float, optional
        controls certainty of ground truth inner borders, by default 0.9.
        Higher values more appropriate when over-segmentation is a concern
    beta : float, optional
        controls certainty of ground truth outer borders, by default 0.1
        Higher values more appropriate when under-segmentation is a concern
    """
    # start_time = timer()
    # res = np.zeros_like(seg)
    # check_seg = seg.astype(np.bool)
    # seg = np.squeeze(seg)

    res = torch.zeros_like(seg)
    check_seg = seg.to(torch.bool)
    seg = seg.squeeze()

    if check_seg.any():
        # kernel = np.ones((3,3),np.uint8)
        kernel = torch.ones((3, 3), torch.uint8)
        # im_erode = cv2.erode(seg,kernel,iterations = 1)
        # im_dilate = cv2.dilate(seg,kernel,iterations = 1)
        start_time = timer()
        im_erode = operations.erosion(seg, kernel)
        im_dilate = operations.dilation(seg, kernel)

        accumulated_res += timer()-start_time
        accumulated_res_idx += 1
        
        res = compute_res(seg, im_erode, im_dilate, alpha, beta)

        # print(f'{timer()-start_time:.5f}s for p1')
        # res = np.expand_dims(res,axis=-1)

        return res
    else:
        return res

# Enables batch processing of boundary uncertainty
# def border_uncertainty_sigmoid_batch(y_true):
#     return torch.from_numpy(np.array([border_uncertainty_sigmoid(y.cpu().numpy()) for y in y_true]).astype(np.float32))

# https://github.com/mlyg/unified-focal-loss/blob/main/loss_functions.py
# For later https://github.com/LIVIAETS/boundary-loss
class unifiedloss(torch.nn.Module):
    def __init__(self, weight=1, delta=0.6, gamma=0.5, act=torch.nn.Sigmoid(), label_smoothing=0.1, outchannels=2, boundary=True):
        super().__init__()
        '''
        weight: float, optional represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
        delta : float, optional controls weight given to each class, by default 0.6
        gamma : float, optional focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
        '''
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.act = act
        self.label_smoothing = label_smoothing
        self.outchannels = outchannels
        self.boundary = boundary

    def forward(self, pred, target):
        if self.boundary:
            global accumulated_res_idx, accumulated_res
            device = target.device
            # Convert batch to np
            accumulated_res_idx = 0
            accumulated_res = 0

            start_time = timer()
            target_np = target.permute(0, 2, 3, 1)
            print(f'\t\t{timer()-start_time:.5f}s for boundary')

            # start_time = timer()
            # target_np = target_np.cpu().numpy()
            # print(f'\t\t{timer()-start_time:.5f}s for cpu numpy')
            
            start_time = timer()
            target = [border_uncertainty_sigmoid(y_true) for y_true in target_np]
            print(f'\t\t{timer()-start_time:.5f}s for list comprehension')

            start_time = timer()
            target = np.array(target)
            print(f'\t\t{timer()-start_time:.5f}s for np.array(target)')

            start_time = timer()
            target = torch.from_numpy(target).to(device).permute(0, 3, 1, 2)
            print(f'\t\t{timer()-start_time:.5f}s for last step')
            print(f'\t\t\t{accumulated_res/accumulated_res_idx:.5f}s for compute_res')

        # Label Smoothing
        start_time = timer()
        target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels
        print(f'\t\t{timer()-start_time:.5f}s for smoothing')

        # Masking
        start_time = timer()
        mask = torch.sum(target, dim=1) == 1
        pred = self.act(pred).permute(0,2,3,1)
        target = target.permute(0,2,3,1)
        pred = pred[mask]
        target = target[mask]
        print(f'\t\t{timer()-start_time:.5f}s for masking')
        
        # Losses
        start_time = timer()
        asymmetric_ftl = self.asymmetric_focal_tversky_loss(pred, target)
        print(f'\t\t{timer()-start_time:.5f}s for tversky')

        start_time = timer()
        asymmetric_fl = self.asymmetric_focal_loss(pred, target)
        print(f'\t\t{timer()-start_time:.5f}s for focal')

        if self.weight is not None:
            return (self.weight * asymmetric_ftl) + ((1-self.weight) * asymmetric_fl)  
        else:
            return asymmetric_ftl + asymmetric_fl

    def asymmetric_focal_tversky_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        epsilon = 1e-07
        y_pred = y_pred.clip(epsilon, 1.0 - epsilon)

        tp = (y_true * y_pred).sum(dim=0)
        fn = (y_true * (1-y_pred)).sum(dim=0)
        fp = ((1-y_true) * y_pred).sum(dim=0)
        dice_class = (tp + epsilon)/(tp + self.delta*fn + (1-self.delta)*fp + epsilon)
        back_dice = (1-dice_class[0]) 
        fore_dice = (1-dice_class[1]) * torch.pow(1-dice_class[1], -self.gamma) 
        return torch.stack([back_dice, fore_dice], -1).mean()

    def asymmetric_focal_loss(self, y_pred, y_true):
        epsilon = 1e-07
        y_pred = y_pred.clip(epsilon, 1.0 - epsilon)

        cross_entropy = -y_true * torch.log(y_pred)
        back_ce = torch.pow(1 - y_pred[:,0], self.gamma) * cross_entropy[:,0]
        back_ce =  (1 - self.delta) * back_ce

        fore_ce = cross_entropy[:,1]
        fore_ce = self.delta * fore_ce
        return torch.stack([back_ce, fore_ce], -1).sum(-1).mean()

class diceloss(torch.nn.Module):
    def __init__(self, act=torch.nn.Sigmoid(), smooth=1.0, w=[1.0], outchannels=1, label_smoothing=0, masked = False):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.w = torch.tensor(w)
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked

    def forward(self, pred, target):
        if len(self.w) != self.outchannels:
            raise ValueError("Loss weights should be equal to the output channels.")
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones((target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool)
        target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels
        pred = self.act(pred).permute(0,2,3,1)
        target = target.permute(0,2,3,1)
        intersection = (pred * target)[mask].sum(dim=0)
        A_sum = pred[mask].sum(dim=0)
        B_sum = target[mask].sum(dim=0)
        total_sum = A_sum + B_sum
        dice = 1 - ((2.0 * intersection + self.smooth) / (total_sum + self.smooth))

        dice = dice * self.w.to(device=dice.device)
        return dice.sum()

class iouloss(torch.nn.Module):
    def __init__(self, act=torch.nn.Sigmoid(), smooth=1.0, w=[1.0], outchannels=1, label_smoothing=0, masked = False):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.w = torch.tensor(w)
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked

    def forward(self, pred, target):
        if len(self.w) != self.outchannels:
            raise ValueError("Loss weights should be equal to the output channels.")
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones((target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool)
        target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels

        pred = self.act(pred).permute(0,2,3,1)
        target = target.permute(0,2,3,1)
        intersection = (pred * target)[mask].sum(dim=0)
        A_sum = pred[mask].sum(dim=0)
        B_sum = target[mask].sum(dim=0)
        union = A_sum + B_sum - intersection
        iou =  1 - ((intersection + self.smooth) / (union + self.smooth))
        iou = iou * self.w.to(device=iou.device)
        return iou.sum()

class celoss(torch.nn.Module):
    def __init__(self, act=torch.nn.Sigmoid(), smooth=1.0, w=[1.0], outchannels=1, label_smoothing=0, masked = False):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.w = torch.tensor(w)
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked

    def forward(self, pred, target):
        if len(self.w) != self.outchannels:
            raise ValueError("Loss weights should be equal to the output channels.")
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones((target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool)
        target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels
        pred = self.act(pred)
        ce = torch.nn.CrossEntropyLoss(weight=self.w.to(device=pred.device))(pred, torch.argmax(target, dim=1).long())
        return ce

class nllloss(torch.nn.Module):
    def __init__(self, act=torch.nn.Sigmoid(), smooth=1.0, w=[1.0], outchannels=1, label_smoothing=0, masked = False):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.w = torch.tensor(w)
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked

    def forward(self, pred, target):
        if len(self.w) != self.outchannels:
            raise ValueError("Loss weights should be equal to the output channels.")
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones((target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool)
        target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels

        pred = self.act(pred)
        nll = torch.nn.NLLLoss(weight=self.w.to(device=pred.device))(pred, torch.argmax(target, dim=1).long())
        return nll


class senseloss(torch.nn.Module):
    def __init__(self, act=torch.nn.Sigmoid(), smooth=1.0, w=[1.0], outchannels=1, label_smoothing=0, masked = False):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.w = torch.tensor(w)
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked

    def forward(self, pred, target):
        if len(self.w) != self.outchannels:
            raise ValueError("Loss weights should be equal to the output channels.")
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones((target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool)
        target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels

        pred = self.act(pred).permute(0,2,3,1)
        target = target.permute(0,2,3,1)
        intersection = (pred * target)[mask].sum(dim=0)
        A_sum = pred[mask].sum(dim=0)
        B_sum = target[mask].sum(dim=0)
        union = A_sum + B_sum - intersection
        iou =  -(intersection + self.smooth) / (union + self.smooth)
        iou = iou * self.w.to(device=iou.device)
        return iou.sum()


# class focalloss(torch.nn.modules.loss._WeightedLoss):
#     def __init__(self, act=torch.nn.Sigmoid(), smooth=1.0, w=[1.0], outchannels=1, label_smoothing=0, masked = False, gamma=2):
#         super().__init__()

#     def forward(self, pred, target):
#         focal_loss = sigmoid_focal_loss(pred, target, alpha = -1, gamma = 2, reduction = "mean")
#         return focal_loss


class customloss(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, act=torch.nn.Sigmoid(), smooth=1.0, w=[1.0], outchannels=1, label_smoothing=0, masked = False, gamma=2, sigma=0):
        super().__init__()
        self.act = act
        self.smooth = smooth
        self.w = torch.tensor(w)
        self.outchannels = outchannels
        self.label_smoothing = label_smoothing
        self.masked = masked
        self.sigma = sigma

    def forward(self, pred, target):
        if len(self.w) != self.outchannels:
            raise ValueError("Loss weights should be equal to the output channels.")
        if self.masked:
            mask = torch.sum(target, dim=1) == 1
        else:
            mask = torch.ones((target.size()[0], target.size()[2], target.size()[3]), dtype=torch.bool)
        focal_loss = sigmoid_focal_loss(pred, target, alpha = -1, gamma = 2, reduction = "mean")
        target = target * (1 - self.label_smoothing) + self.label_smoothing / self.outchannels
        if self.sigma > 0:
            for i in range(target.shape[0]):
                for j in range(target.shape[1]):
                    target[i,j,:,:] = torch.from_numpy(gaussian_filter(target[i,j,:,:].cpu() ,self.sigma))
        #print(pred.max(), pred.shape)
        _pred = self.act(pred)
        #print(_pred.max(), pred.shape)
        #pdb.set_trace()
        ce = torch.nn.CrossEntropyLoss(weight=self.w.to(device=_pred.device))(_pred, torch.argmax(target, dim=1).long())

        pred = self.act(pred).permute(0,2,3,1)
        target = target.permute(0,2,3,1)
        intersection = (pred * target)[mask].sum(dim=0)
        A_sum = pred[mask].sum(dim=0)
        B_sum = target[mask].sum(dim=0)
        total_sum = A_sum + B_sum
        dice = 1 - ((2.0 * intersection + self.smooth) / (total_sum + self.smooth))
        dice = dice * self.w.to(device=dice.device)

        return 0.67*dice.sum() + 0.33*ce
