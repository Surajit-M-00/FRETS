import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------
# Negative Pearson Correlation (same class name kept)
# ----------------------------------------------------
class NegCorrLoss(nn.Module):
    def __init__(self):
        super(NegCorrLoss, self).__init__()

    def forward(self, x, y):
        # x, y: [batch, time]
        x = x - x.mean(dim=-1, keepdim=True)
        y = y - y.mean(dim=-1, keepdim=True)
        x_norm = F.normalize(x, p=2, dim=-1)
        y_norm = F.normalize(y, p=2, dim=-1)
        corr = (x_norm * y_norm).sum(dim=-1)
        return -corr.mean()

# ----------------------------------------------------
# Combined_loss (BUT NOW ONLY MSE + NegCorr)
# ----------------------------------------------------
class Combined_loss(nn.Module):
    """
    L_SpO2 = MSE(Y, Y_GT) + NegCorr(Y, Y_GT)
    Class name is preserved exactly as requested.
    """
    def __init__(self):
        super(Combined_loss, self).__init__()
        self.mse = nn.MSELoss()
        self.neg_corr = NegCorrLoss()

    def forward(self, y_pred, y_true, *args, **kwargs):
        # Only SpO2 supervision — ignore X_DC_pred, X_AC_pred etc.
        l_mse = self.mse(y_pred, y_true)
        l_corr = self.neg_corr(y_pred, y_true)
        loss = l_mse + l_corr
        return loss

# Export EXACT same name
SpO2Loss = Combined_loss

