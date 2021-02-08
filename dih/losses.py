import torch
import torch.nn as nn


class DIHLoss(nn.Module):

    def __init__(
        self,
        recon_loss_weight: float,
        pixel_loss_weight: float
    ):
        super().__init__()
        self.recon_loss = nn.MSELoss()
        self.pixel_loss = nn.CrossEntropyLoss()
        self.recon_loss_weight = recon_loss_weight
        self.pixel_loss_weight = pixel_loss_weight

    def forward(
        self,
        y_pred: torch.Tensor,
        y_pred_mask: torch.Tensor,
        y_true: torch.Tensor,
        y_true_mask: torch.Tensor
    ) -> torch.Tensor:

        # Reconstruction loss
        recon_loss = self.recon_loss(y_pred, y_true)
        recon_loss *= self.recon_loss_weight

        # Segmentation loss
        pixel_loss = self.pixel_loss(y_pred_mask, y_true_mask)
        pixel_loss *= self.pixel_loss_weight

        return recon_loss + pixel_loss
