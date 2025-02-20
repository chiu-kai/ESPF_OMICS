import torch
import torch.nn as nn

class MAE_MSEmean(nn.Module):
    def __init__(self):
        super(MAE_MSE, self).__init__()
        self.mae_loss = nn.L1Loss()  # MAE
        self.mse_loss = nn.MSELoss()  # MSE

    def forward(self, prediction, target):
        mae = self.mae_loss(prediction, target)
        mse = self.mse_loss(prediction, target)
        combined_loss = (mae + mse) / 2
        return combined_loss
    
class MAE_RMSEmean(nn.Module):
    def __init__(self):
        super(MAE_RMSEmean, self).__init__()
        self.mae_loss = nn.L1Loss()  # MAE
        self.mse_loss = nn.MSELoss()  # MSE (used for RMSE)

    def forward(self, prediction, target):
        mae = self.mae_loss(prediction, target)
        mse = self.mse_loss(prediction, target)
        combined_loss = (mae + torch.sqrt(mse) ) / 2
        return combined_loss
    
    def __repr__(self):  # Customize how the object is represented
            return "MAE_RMSEmean"
# Example usage
# criterion = MAE_RMSEmean()