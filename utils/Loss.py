import torch
import torch.nn as nn

import torch.nn.functional as F

class BCE_FocalLoss(nn.Module):
    def __init__(self, loss_type="BCE_Focal", alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Initializes the FocalLoss module for inputs that are already probabilities (after sigmoid).
        Args:
            alpha (float): Weighting factor for positive and negative samples.
                           A common value is 0.25 for positive samples (1-alpha for negative).
            gamma (float): Focusing parameter. Higher gamma reduces the loss for easy examples.
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. 'mean' is the default.
        """
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets, model=None, weights=None):
        """
        Args:
            inputs (torch.Tensor): Predicted probabilities (output of sigmoid). Shape: (N, 1) or (N, ...)
            targets (torch.Tensor): Ground truth labels (0 or 1). Shape: (N, 1) or (N, ...)
        Returns:
            torch.Tensor: The computed focal loss.
        """
        # Ensure inputs and targets are floats for calculations
        inputs = inputs.float()
        targets = targets.float()

        # Clamp inputs to avoid log(0) errors (though F.binary_cross_entropy handles this internally too)
        inputs = torch.clamp(inputs, min=1e-8, max=1-1e-8)

        # Step 1: Compute binary cross-entropy (BCE)
        # Use F.binary_cross_entropy since inputs are already probabilities
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        # Step 2: Calculate p_t (probability of the true class)
        # If target is 1, p_t is 'inputs'. If target is 0, p_t is '1 - inputs'.
        # This can be elegantly written as:
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        # Another way to get p_t from BCE_loss (if BCE_loss = -log(p_t)):
        # p_t = torch.exp(-BCE_loss)
        # However, the direct calculation using inputs and targets is more intuitive here.
        # Step 3: Compute the modulating factor (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        # Step 4: Compute alpha term
        # alpha_t is alpha for positive class (target=1) and (1-alpha) for negative class (target=0)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        # Step 5: Combine everything
        loss= alpha_t * focal_term * BCE_loss
        
        # Step 6: Apply reduction
        if self.reduction == 'mean':
            self.loss_WO_penalty = loss.mean()
            return self.loss_WO_penalty 
        elif self.reduction == 'sum':
            self.loss_WO_penalty = loss.sum()
            return self.loss_WO_penalty
        elif self.reduction == 'none':
            return self.loss_WO_penalty
        else:
            raise ValueError("Reduction must be 'none', 'mean', or 'sum'.")
    def __repr__(self):
        return (f"loss_type={self.loss_type}, "
                f"self.alpha={self.alpha}, "
                f"self.gamma={self.gamma}, "
                f"self.reduction={self.reduction})")
    
class FocalLoss(nn.Module):
    def __init__(self, loss_type="", alpha=8.0, gamma=1.0, regular_type=None, regular_lambda=1e-05):
        super(FocalLoss, self).__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.gamma = gamma
        self.regular_type = regular_type
        self.regular_lambda = regular_lambda
        self.penalty_value = None
    
    def forward(self, y_pred, y_true, model=None, weights=None):
        error = torch.abs(y_true - y_pred)
        weight = (1 - torch.exp(-self.alpha * error)) ** self.gamma  # Weight function
        
        if self.loss_type == "MSE":
            loss_wo_penalty = (weight * (error ** 2)).mean()  # Weighted MSE
        elif self.loss_type == "MAE":
            loss_wo_penalty = (weight * error).mean()  # Weighted MAE
        else:
            raise ValueError(f"Unsupported Loss type: {self.loss_type}")

        loss = loss_wo_penalty
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
            self.penalty_value = self.regular_lambda * reg_penalty
            loss += self.penalty_value
        
        return loss
    
    def __repr__(self):
        return (f"FocalLoss(loss_type={self.loss_type}, alpha={self.alpha}, gamma={self.gamma}, "
                f"regular_type={self.regular_type}, regular_lambda={self.regular_lambda}, "
                f"penalty_value={self.penalty_value})")

# Example usage:
# criterion = FocalHuberLoss(loss_type="FocalHuberLoss",delta=0.2, alpha=0.3, gamma=2.0, regular_type=None, regular_lambda=1e-05)
class FocalHuberLoss(nn.Module):
    def __init__(self,loss_type="FocalHuberLoss", delta=0.2, alpha=1.0, gamma=2.0, regular_type=None, regular_lambda=1e-05):
        """Focal Huber Loss for regression.
                Args:
                - delta: Huber threshold.
                - alpha: Controls how fast weight decays for easy samples.
                - gamma: Controls focus on hard samples."""
        super(FocalHuberLoss, self).__init__()
        self.loss_type=loss_type
        self.delta = delta
        self.alpha = alpha
        self.gamma = gamma
        self.regular_type = regular_type
        self.regular_lambda = regular_lambda
        self.penalty_value = None

    def forward(self, y_pred, y_true, model=None, weights=None):
        error = torch.abs(y_true - y_pred)
        weight = (1 - torch.exp(-self.alpha * error)) ** self.gamma  # Focal weighting

        # Huber loss: quadratic for small errors, linear for large errors
        quadratic = 0.5 * (error ** 2)
        linear = self.delta * (error - 0.5 * self.delta)
        huber_loss = torch.where(error < self.delta, quadratic, linear)
        self.loss_WO_penalty = (weight * huber_loss).mean()

        self.loss = self.loss_WO_penalty
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
            self.penalty_value = self.regular_lambda * reg_penalty 
            self.loss += self.penalty_value
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
        self.bce_loss = nn.BCELoss(reduction="none") # reduction (default 'mean')
        
    def forward(self, prediction, target, model=None, weights=None):
        """
        Compute the loss based on the specified loss type, regularization, and weights.
        Args:
            prediction (Tensor): The model's predictions.
            target (Tensor): The ground truth values.
            model (nn.Module, optional): The model for calculating regularization penalties.
            weights (Tensor, optional): Weights for each sample in the batch. Give more weight to few samples.
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
        elif self.loss_type == "weighted_BCE": # sigmoid is already done in model
            batch_sample_loss = self.bce_loss(prediction, target)
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
        self.loss_WO_penalty = batch_sample_loss.mean()

        self.loss = self.loss_WO_penalty
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
            self.penalty_value = self.regular_lambda * reg_penalty 
            self.loss += self.penalty_value
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
        self.bce_loss = nn.BCELoss() # reduction (default 'mean')

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
            self.loss_WO_penalty = torch.sqrt(self.mse_loss(prediction, target))
        elif self.loss_type == "MSE":
            self.loss_WO_penalty = self.mse_loss(prediction, target)
        elif self.loss_type == "MAE":
            self.loss_WO_penalty = self.mae_loss(prediction, target)
        elif self.loss_type == "BCE": # sigmoid is already done in model
            self.loss_WO_penalty = self.bce_loss(prediction, target)
        
        elif self.loss_type == "MAE+BCE":
            mae = self.mae_loss(prediction, target)
            bce = self.bce_loss(prediction, target)
            self.loss_WO_penalty = mae + self.loss_lambda * bce
        elif self.loss_type == "MAE+MSE":
            mae = self.mae_loss(prediction, target)
            mse = self.mse_loss(prediction, target)
            self.loss_WO_penalty = mae + self.loss_lambda * mse
        elif self.loss_type == "MAE+RMSE":
            mae = self.mae_loss(prediction, target)
            mse = self.mse_loss(prediction, target)
            self.loss_WO_penalty = mae + self.loss_lambda * torch.sqrt(mse)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        self.loss = self.loss_WO_penalty
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
            self.penalty_value = self.regular_lambda * reg_penalty 
            self.loss += self.penalty_value
        return self.loss

    def __repr__(self):
        return (f"Custom_LossFunction(loss_type={self.loss_type}, "
                f"loss_lambda={self.loss_lambda}, "
                f"regular_type={self.regular_type}, "
                f"regular_lambda={self.regular_lambda})")





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