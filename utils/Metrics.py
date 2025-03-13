import numpy as np
import torch
import torch.nn as nn
class MetricsCalculator_nntorch(nn.Module):
    def __init__(self,types= ["MSE", "MAE", "R^2"]): #
        super(MetricsCalculator_nntorch, self).__init__() # Call nn.Module's init first!
        
        self.types = types
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.mae_loss = nn.L1Loss(reduction="mean") # reduction (default 'mean')

    def forward(self, y_true, y_pred): 
        self.results = {}
        if "MSE" in self.types:
            self.results["MSE"] = self.mse_loss(y_true, y_pred) 

        if "RMSE" in self.types:
            self.results["RMSE"] = torch.sqrt(self.mse_loss(y_true, y_pred) )

        if "MAE" in self.types:
            self.results["MAE"] = self.mae_loss(y_true, y_pred)

        if "R^2" in self.types:
            y_mean = torch.mean(y_true)  # Compute mean once
            ss_total = torch.sum((y_true - y_mean) ** 2)
            ss_residual = torch.sum((y_true - y_pred) ** 2)
            self.results["R^2"] = 1 - (ss_residual / ss_total)
        
        return self.results




class MetricsCalculator_numpy:
    def __init__(self): #
        self.results = {}
        
    def calculate_rmse(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2)) 

    def calculate_mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def calculate_mae(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred)) 

    def calculate_r2(self, y_true, y_pred):
        y_true_mean = np.mean(y_true)
        ss_total = np.sum((y_true - y_true_mean) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

    def compute_all_metrics(self, y_true, y_pred,set_name): 
        mse = self.calculate_mse(y_true, y_pred)
        mae = self.calculate_mae(y_true, y_pred)
        r2 = self.calculate_r2(y_true, y_pred)
        
        self.results = {
            "Evaluation":set_name,
            "MSE": mse,
            "MAE": mae,
            "R^2": r2 }
        return self.results

    def print_results(self,set_name):
        print(f"Evaluation {set_name}")
        print(self.results)

