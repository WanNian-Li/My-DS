__author__ = 'Muhammed Patel'
__contributor__ = 'Xinwwei chen, Fernando Pena Cantu,Javier Turnes, Eddie Park'
__copyright__ = ['university of waterloo']
__contact__ = ['m32patel@uwaterloo.ca', 'xinweic@uwaterloo.ca']
__version__ = '1.0.0'
__date__ = '2024-04-05'

import torch
from torch import nn
import torch.nn.functional as F


class OrderedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(OrderedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, output: torch.Tensor, target: torch.Tensor):

        criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)
        loss = criterion(output, target)
        # calculate the hard predictions by using softmax followed by an argmax
        softmax = torch.nn.functional.softmax(output, dim=1)
        hard_prediction = torch.argmax(softmax, dim=1)
        # set the mask according to ignore index
        mask = target == self.ignore_index
        hard_prediction = hard_prediction[~mask]
        target = target[~mask]
        # calculate the absolute difference between target and prediction
        weights = torch.abs(hard_prediction-target) + 1
        # remove ignored index losses
        loss = loss[~mask]
        # if done normalization with weights the loss becomes of the order 1e-5
        # loss = (loss * weights)/weights.sum()
        loss = (loss * weights)
        loss = loss.mean()

        return loss


class MSELossFromLogits(nn.Module):
    def __init__(self, chart, ignore_index=-100):
        super(MSELossFromLogits, self).__init__()
        self.ignore_index = ignore_index
        self.chart = chart
        if self.chart == 'SIC':
            self.replace_value = 11
            self.num_classes = 12
        elif self.chart == 'SOD':
            self.replace_value = 4
            self.num_classes = 5
        elif self.chart == 'FLOE':
            self.replace_value = 7
            self.num_classes = 8
        else:
            raise NameError(f'The chart {self.chart} is not recognized')
        
        # Create class weights for expectation calculation: [0, 1, 2, ..., N-1]
        self.register_buffer('class_weights', torch.arange(self.num_classes).float())

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        """
        Calculate MSE/Distance loss treating the classification problem as an ordinal regression.
        Instead of One-Hot MSE, we calculate the expected class value from the softmax distribution
        and compare it to the ground truth class index.
        """
        # Create a mask for valid pixels (not ignore_index)
        valid_mask = (target != self.ignore_index)  # (B, H, W)

        # If no valid pixels, return zero loss
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=output.device, requires_grad=True)

        # Select valid pixels
        # output shape: (B, C, H, W) -> permute to (B, H, W, C) -> mask to (N_valid, C)
        output_valid = output.permute(0, 2, 3, 1)[valid_mask] # (N_valid, num_classes)
        target_valid = target[valid_mask].float()             # (N_valid,)

        # Calculate Softmax probabilities
        probs = F.softmax(output_valid, dim=1)  # (N_valid, num_classes)

        # Calculate Expected Value (Soft Prediction)
        # E[x] = sum(p_i * i)
        # self.class_weights shape: (num_classes,)
        # Ensure class_weights is on the same device
        pred_expected = torch.sum(probs * self.class_weights.to(output.device), dim=1) # (N_valid,)

        # Calculate MSE between Expected Value and True Class Index
        loss = F.mse_loss(pred_expected, target_valid)

        return loss

class FocalLoss(nn.Module):
    """Alpha-weighted Focal Loss for multi-class segmentation.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: focusing parameter >= 0. gamma=0 degenerates to weighted CE.
        weight: per-class alpha weights, list/tuple or 1-D FloatTensor [C].
        ignore_index: pixels with this label are excluded from loss.
    """

    def __init__(self, gamma: float = 2.0, weight=None, ignore_index: int = 255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if weight is not None:
            self.register_buffer('weight', torch.FloatTensor(weight))
        else:
            self.weight = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs : [B, C, H, W] logits
        # targets: [B, H, W]   class indices
        valid_mask = targets != self.ignore_index          # [B, H, W]
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0                      # prevent gather OOB

        log_prob = F.log_softmax(inputs, dim=1)            # [B, C, H, W]
        log_pt = log_prob.gather(1, targets_safe.unsqueeze(1)).squeeze(1)  # [B, H, W]
        pt = log_pt.exp()

        if self.weight is not None:
            alpha_t = self.weight[targets_safe]            # [B, H, W]
        else:
            alpha_t = inputs.new_ones(targets.shape)

        loss = -alpha_t * (1.0 - pt) ** self.gamma * log_pt   # [B, H, W]
        loss = loss * valid_mask.float()
        return loss.sum() / valid_mask.float().sum().clamp(min=1.0)


class WaterConsistencyLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.keys = ['SIC', 'SOD', 'FLOE']
        self.activation = nn.Softmax(dim=1)
    
    def forward(self, output):
        sic = self.activation(output[self.keys[0]])[:, 0, :, :]
        sod = self.activation(output[self.keys[1]])[:, 0, :, :]
        floe = self.activation(output[self.keys[2]])[:, 0, :, :]
        return torch.mean((sic-sod)**2 + (sod-floe)**2 + (floe-sic)**2)

# only applicable to regression outputs
class MSELossWithIgnoreIndex(nn.MSELoss):
    def __init__(self, ignore_index=255, reduction='mean'):
        super(MSELossWithIgnoreIndex, self).__init__(reduction=reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = (target != self.ignore_index).type_as(input)
        diff = input.squeeze(-1) - target
        diff = diff * mask
        loss = torch.sum(diff ** 2) / mask.sum()
        return loss

# only applicable to regression outputs
class MSELossWithIgnoreIndex(nn.MSELoss):
    def __init__(self, ignore_index=255, reduction='mean'):
        super(MSELossWithIgnoreIndex, self).__init__(reduction=reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        mask = (target != self.ignore_index).type_as(input)
        diff = input.squeeze(-1) - target
        diff = diff * mask
        loss = torch.sum(diff ** 2) / mask.sum()
        return loss
