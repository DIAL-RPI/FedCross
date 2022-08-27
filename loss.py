import torch
import torch.nn as nn
import numpy as np

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target, flag):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        intersection = self.smooth
        union = self.smooth
        if flag is None:
            pd = predict
            gt = target
            intersection += torch.sum(pd*gt)*2
            union += torch.sum(pd.pow(self.p) + gt.pow(self.p))
        else:
            for i in range(target.shape[0]):
                if flag[i,0] > 0:
                    pd = predict[i:i+1,:]
                    gt = target[i:i+1,:]
                    intersection += torch.sum(pd*gt)*2
                    union += torch.sum(pd.pow(self.p) + gt.pow(self.p))
        dice = intersection / union

        loss = 1 - dice
        return loss
        
class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=[], **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        if weight is not None:
            self.weight = weight / weight.sum()
        else:
            self.weight = None
        self.ignore_index = ignore_index

    def forward(self, predict, target, flag=None):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        total_loss_num = 0

        for c in range(target.shape[1]):
            if c not in self.ignore_index:
                dice_loss = dice(predict[:, c], target[:, c], flag)
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weight[c]
                total_loss += dice_loss
                total_loss_num += 1

        if self.weight is not None:
            return total_loss
        elif total_loss_num > 0:
            return total_loss/total_loss_num
        else:
            return 0

def make_onehot(input, cls):
    oh_list = []
    for c in range(cls):
        tmp = torch.zeros_like(input)
        tmp[input==c] = 1        
        oh_list.append(tmp)
    oh = torch.cat(oh_list, dim=1)
    return oh

def dice_and_ce_loss(prob, logit, target):
    cls_num = np.sum(logit.shape[1])
    target_oh = make_onehot(target, cls=cls_num)
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    l_ce = ce_loss(logit, target.squeeze(dim=1))
    l_dice = dice_loss(prob, target_oh)

    return l_ce, l_dice