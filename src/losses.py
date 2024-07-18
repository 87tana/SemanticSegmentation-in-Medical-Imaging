import torch
import torch.nn.functional as F


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # Apply sigmoid to get probabilities if logits are provided
        y_pred = torch.sigmoid(y_pred)

        # Flatten label and prediction tensors
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)

        # Compute the intersection and the union
        intersection = (y_pred_flat * y_true_flat).sum()
        union = y_pred_flat.sum() + y_true_flat.sum()

        # Compute Dice score
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)

        # Dice loss is 1 - Dice score
        dice_loss = 1 - dice_score

        return dice_loss
