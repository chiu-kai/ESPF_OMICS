import torch
import torch.nn as nn

import torch
import torch.nn as nn

class FocalMSELoss(nn.Module):
    def __init__(self, alpha=8.0, gamma=1.0, regular_type=None, regular_lambda=1e-05):
        super(FocalMSELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.regular_type = regular_type
        self.regular_lambda = regular_lambda
        self.penalty_value = None
        self.loss_type="FocalMSELoss"

    def forward(self, y_pred, y_true, model=None, weights=None):
        error = torch.abs(y_true - y_pred)
        weight = (1 - torch.exp(-self.alpha * error)) ** self.gamma  # Weight function
        self.loss = (weight * (error ** 2)).mean()  # Weighted MSE
        
        loss_with_penalty= None
        # Add regularization penalty if specified
        if self.regular_type and model:
            if self.regular_type == "L1":
                reg_penalty = sum(p.abs().sum() for p in model.parameters())
            elif self.regular_type == "L2":
                reg_penalty = sum(p.pow(2).sum() for p in model.parameters())
            elif self.regular_type == "L1+L2":
                reg_penalty = sum(p.abs().sum() + p.pow(2).sum() for p in model.parameters())
            else:
                raise ValueError(f"Unsupported regularization type: {self.regular_type}")
            loss_with_penalty = self.loss + self.regular_lambda * reg_penalty
            self.penalty_value = self.regular_lambda * reg_penalty
            return loss_with_penalty
        return self.loss

    def __repr__(self):
        return (f"FocalMSELoss(alpha={self.alpha},gamma={self.gamma}),"
                f"regular_type={self.regular_type}, "
                f"regular_lambda={self.regular_lambda})"
                f"penalty_value={self.penalty_value}\n")
    
class FocalMAELoss(nn.Module):
    def __init__(self, alpha=8.0, gamma=1.0, regular_type=None, regular_lambda=1e-05):
        super(FocalMAELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.regular_type = regular_type
        self.regular_lambda = regular_lambda
        self.penalty_value = None
        self.loss_type="FocalMAELoss"

    def forward(self, y_pred, y_true, model=None, weights=None):
        error = torch.abs(y_true - y_pred)
        weight = (1 - torch.exp(-self.alpha * error)) ** self.gamma  # Weight function
        self.loss = (weight * error).mean()  # Weighted MAE
        
        loss_with_penalty= None
        # Add regularization penalty if specified
        if self.regular_type and model:
            if self.regular_type == "L1":
                reg_penalty = sum(p.abs().sum() for p in model.parameters())
            elif self.regular_type == "L2":
                reg_penalty = sum(p.pow(2).sum() for p in model.parameters())
            elif self.regular_type == "L1+L2":
                reg_penalty = sum(p.abs().sum() + p.pow(2).sum() for p in model.parameters())
            else:
                raise ValueError(f"Unsupported regularization type: {self.regular_type}")
            loss_with_penalty = self.loss + self.regular_lambda * reg_penalty
            self.penalty_value = self.regular_lambda * reg_penalty
            return loss_with_penalty
        return self.loss

    def __repr__(self):
        return (f"FocalMAELoss(alpha={self.alpha},gamma={self.gamma}),"
                f"regular_type={self.regular_type}, "
                f"regular_lambda={self.regular_lambda})"
                f"penalty_value={self.penalty_value}\n")

# Example usage:
# criterion = FocalHuberLoss(delta=0.2, alpha=0.3, gamma=2.0, regular_type=None, regular_lambda=1e-05)
class FocalHuberLoss(nn.Module):
    def __init__(self, delta=0.2, alpha=1.0, gamma=2.0, regular_type=None, regular_lambda=1e-05):
        """
        Focal Huber Loss for regression.

        Args:
        - delta: Huber threshold.
        - alpha: Controls how fast weight decays for easy samples.
        - gamma: Controls focus on hard samples.
        """
        super(FocalHuberLoss, self).__init__()
        self.delta = delta
        self.alpha = alpha
        self.gamma = gamma
        self.regular_type = regular_type
        self.regular_lambda = regular_lambda
        self.penalty_value = None
        self.loss_type="FocalHuberLoss"

    def forward(self, y_pred, y_true, model=None, weights=None):
        error = torch.abs(y_true - y_pred)
        weight = (1 - torch.exp(-self.alpha * error)) ** self.gamma  # Focal weighting

        # Huber loss: quadratic for small errors, linear for large errors
        quadratic = 0.5 * (error ** 2)
        linear = self.delta * (error - 0.5 * self.delta)
        huber_loss = torch.where(error < self.delta, quadratic, linear)
        self.loss = (weight * huber_loss).mean()

        loss_with_penalty=None
        # Add regularization penalty if specified
        if self.regular_type and model:
            if self.regular_type == "L1":
                reg_penalty = sum(p.abs().sum() for p in model.parameters())
            elif self.regular_type == "L2":
                reg_penalty = sum(p.pow(2).sum() for p in model.parameters())
            elif self.regular_type == "L1+L2":
                reg_penalty = sum(p.abs().sum() + p.pow(2).sum() for p in model.parameters())
            else:
                raise ValueError(f"Unsupported regularization type: {self.regular_type}")
            loss_with_penalty = self.loss + self.regular_lambda * reg_penalty
            self.penalty_value = self.regular_lambda * reg_penalty
            return loss_with_penalty
        
        return self.loss
    def __repr__(self):
        return (f"FocalHuberLoss(delta={self.delta},alpha={self.alpha},gamma={self.gamma}),"
                f"regular_type={self.regular_type}, "
                f"regular_lambda={self.regular_lambda}"
                f"penalty_value={self.penalty_value}\n") 

# Example usage:
# criterion = Custom_LossFunction(loss_type="RMSE", loss_lambda=1.0, regular_type=None, regular_lambda=1e-05)
class Custom_Weighted_LossFunction(nn.Module):
    def __init__(self, loss_type="weighted_RMSE", loss_lambda=1.0, regular_type=None, regular_lambda=1e-05):
        super(Custom_Weighted_LossFunction, self).__init__()
        self.loss_type = loss_type
        self.loss_lambda = loss_lambda
        self.regular_type = regular_type
        self.regular_lambda = regular_lambda
        self.penalty_value = None

        self.mse_loss = nn.MSELoss(reduction="none")  # Keep per-sample loss # 不要取平均 #輸出sample loss list
        self.mae_loss = nn.L1Loss(reduction="none")  # Keep per-sample loss # 不要取平均 #輸出sample loss list
        
    def forward(self, prediction, target, model=None, weights=None):
        """
        Compute the loss based on the specified loss type, regularization, and weights.
        Args:
            prediction (Tensor): The model's predictions.
            target (Tensor): The ground truth values.
            model (nn.Module, optional): The model for calculating regularization penalties.
            weights (Tensor, optional): Weights for each sample in the batch.
        Returns:
            Tensor: The computed loss.
        """
        # Compute base loss
        if self.loss_type == "weighted_RMSE":
            batch_sample_loss = torch.sqrt(self.mse_loss(prediction, target))
        elif self.loss_type == "weighted_MSE":
            batch_sample_loss = self.mse_loss(prediction, target)
        elif self.loss_type == "weighted_MAE":
            batch_sample_loss = self.mae_loss(prediction, target)
        elif self.loss_type == "weighted_MAE+MSE":
            mae = self.mae_loss(prediction, target)
            mse = self.mse_loss(prediction, target)
            batch_sample_loss = mae + self.loss_lambda * mse
        elif self.loss_type == "weighted_MAE+RMSE":
            mae = self.mae_loss(prediction, target)
            mse = self.mse_loss(prediction, target)
            batch_sample_loss = mae + self.loss_lambda * torch.sqrt(mse)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        # batch_sample_loss.shape : torch.Size([sample's number in a batch])

        # Apply weight to each sample loss 
        if weights is not None:
            batch_sample_loss *= weights 

        # Aggregate loss (mean over batch)
        self.loss = batch_sample_loss.mean()

        loss_with_penalty= None
        # Add regularization penalty if specified
        if self.regular_type and model:
            if self.regular_type == "L1":
                reg_penalty = sum(p.abs().sum() for p in model.parameters())
            elif self.regular_type == "L2":
                reg_penalty = sum(p.pow(2).sum() for p in model.parameters())
            elif self.regular_type == "L1+L2":
                reg_penalty = sum(p.abs().sum() + p.pow(2).sum() for p in model.parameters())
            else:
                raise ValueError(f"Unsupported regularization type: {self.regular_type}")
            loss_with_penalty = self.loss + self.regular_lambda * reg_penalty
            self.penalty_value = self.regular_lambda * reg_penalty
            return loss_with_penalty
        return self.loss

    def __repr__(self):
        return (f"Custom_Weighted_LossFunction(loss_type={self.loss_type}, "
                f"loss_lambda={self.loss_lambda}, "
                f"regular_type={self.regular_type}, "
                f"regular_lambda={self.regular_lambda})"
                f"penalty_value={self.penalty_value}\n")



# Example usage:
# criterion = Custom_LossFunction(loss_type="RMSE", loss_lambda=1.0, regular_type=None, regular_lambda=0.001)
class Custom_LossFunction(nn.Module):
    def __init__(self, loss_type="RMSE", loss_lambda=1.0, regular_type=None, regular_lambda=0.001):
        """
        Args:
            loss_type (str): The type of loss to use ("RMSE", "MSE", "MAE", "MAE+MSE", "MAE+RMSE").
            loss_lambda (float): The lambda weight for the additional loss (MSE or RMSE) if applicable.
            regular_type (str): The type of regularization to use ("L1", "L2", "L1+L2"), or None for no regularization.
            regular_lambda (float): The lambda weight for regularization.
        """
        super(Custom_LossFunction, self).__init__()
        self.loss_type = loss_type
        self.loss_lambda = loss_lambda
        self.regular_type = regular_type
        self.regular_lambda = regular_lambda
        self.penalty_value = None

        self.mse_loss = nn.MSELoss() # reduction (default 'mean')
        self.mae_loss = nn.L1Loss() # reduction (default 'mean')
    def forward(self, prediction, target, model=None, weights=None):
        """
        Compute the loss based on the specified loss type and regularization.
        Args:
            prediction (Tensor): The model's predictions.
            target (Tensor): The ground truth values.
            model (nn.Module, optional): The model for calculating regularization penalties.
        Returns:
            Tensor: The computed loss.
        """
        if self.loss_type == "RMSE":
            self.loss = torch.sqrt(self.mse_loss(prediction, target))
        elif self.loss_type == "MSE":
            self.loss = self.mse_loss(prediction, target)
        elif self.loss_type == "MAE":
            self.loss = self.mae_loss(prediction, target)
        elif self.loss_type == "MAE+MSE":
            mae = self.mae_loss(prediction, target)
            mse = self.mse_loss(prediction, target)
            self.loss = mae + self.loss_lambda * mse
        elif self.loss_type == "MAE+RMSE":
            mae = self.mae_loss(prediction, target)
            mse = self.mse_loss(prediction, target)
            self.loss = mae + self.loss_lambda * torch.sqrt(mse)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        loss_with_penalty= None
        # Add regularization penalty if specified
        if self.regular_type and model:
            if self.regular_type == "L1":
                reg_penalty = sum(p.abs().sum() for p in model.parameters())
            elif self.regular_type == "L2":
                reg_penalty = sum(p.pow(2).sum() for p in model.parameters())
            elif self.regular_type == "L1+L2":
                reg_penalty = sum(p.abs().sum() + p.pow(2).sum() for p in model.parameters())
            else:
                raise ValueError(f"Unsupported regularization type: {self.regular_type}")
            loss_with_penalty = self.loss + self.regular_lambda * reg_penalty
            self.penalty_value = self.regular_lambda * reg_penalty
            return loss_with_penalty
        return self.loss

    def __repr__(self):
        return (f"Custom_LossFunction(loss_type={self.loss_type}, "
                f"loss_lambda={self.loss_lambda}, "
                f"regular_type={self.regular_type}, "
                f"regular_lambda={self.regular_lambda})")

'''
class RMSEplusRegular(nn.Module):
    def __init__(self, regular_type,regular_lambda=0.001):
        super(RMSEplusRegular, self).__init__()
        self.mse_loss = nn.MSELoss()  # MSE
        self.reg_lambda = regular_lambda
        self.reg_type = regular_type

    def forward(self, prediction, target, model):
        mse = self.mse_loss(prediction, target)
        if self.reg_type == 'L1':
            reg_penalty= sum(p.abs().sum() for p in model.parameters())
        elif self.reg_type == 'L2':
            reg_penalty= sum(p.pow(2).sum() for p in model.parameters())
        elif self.reg_type == 'L1+L2':
            reg_penalty= sum(p.abs().sum() + p.pow(2).sum() for p in model.parameters())
        combined_loss = torch.sqrt(mse)+ self.reg_lambda* reg_penalty  
        return combined_loss
    def __repr__(self):  # Customize how the object is represented
        return "RMSE_plus_Regularization"


class MAEplusMSE(nn.Module):
    def __init__(self,MSE_lambda=1): # MSE_lambda: contril the weight of MSE
        super(MAEplusMSE, self).__init__()
        self.mae_loss = nn.L1Loss()  # MAE
        self.mse_loss = nn.MSELoss()  # MSE
        self.MSE_lambda = MSE_lambda

    def forward(self, prediction, target):
        mae = self.mae_loss(prediction, target)
        mse = self.mse_loss(prediction, target)
        combined_loss = mae+ self.MSE_lambda* mse
        return combined_loss
    def __repr__(self):  # Customize how the object is represented
        return "MAEplusMSE"
    
class MAEplusRMSE(nn.Module):
    def __init__(self,RMSE_lambda=1): # default: 1
        super(MAEplusRMSE, self).__init__()
        self.mae_loss = nn.L1Loss()  # MAE
        self.mse_loss = nn.MSELoss()  # MSE (used for RMSE)
        self.RMSE_lambda = RMSE_lambda

    def forward(self, prediction, target):
        mae = self.mae_loss(prediction, target)
        mse = self.mse_loss(prediction, target)
        combined_loss = mae+ self.RMSE_lambda* torch.sqrt(mse)
        return combined_loss
    def __repr__(self):  # Customize how the object is represented
        return "MAEplusRMSE"
    
# Example usage
# criterion = MAE_RMSEmean()
'''

# https://github.com/NYCUciflab/PAGAN_PXR/blob/504c6737d2d5ec759d3da5dba906dd3a9e2d16a1/utils/loss.py
def BCEloss(pred, label):
    
    loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn_2 = nn.MSELoss()
    label = torch.unsqueeze(label, 1).float()
    
    loss_normal = loss_fn(pred, label) # + loss_fn_2(pred, label)
    
    return loss_normal

# Metrics
def get_results(preds, labels, threshold=0.5):
    
    preds = [int(pred>=threshold) for pred in preds]
    
    ε  = 1e-8
    TP = [(pred+label) for (pred, label) in zip(preds, labels)].count(2)
    TN = [(pred+label) for (pred, label) in zip(preds, labels)].count(0)
    FP = [(pred-label) for (pred, label) in zip(preds, labels)].count(1)
    FN = [(pred-label) for (pred, label) in zip(preds, labels)].count(-1)

    acc = round((TP+TN) / (TP+TN+FP+FN+ε), 4)
    sen = round((TP)    / (TP+FN+ε), 4)
    spc = round((TN)    / (TN+FP+ε), 4)
    ydn = round(sen+spc-1, 4)
    
    return acc, sen, spc, ydn