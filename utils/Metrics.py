import numpy as np

class MetricsCalculator:
    def __init__(self): #
        self.results = {}
        

    def calculate_rmse(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2)) 

    def calculate_mae(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred)) 

    def calculate_r2(self, y_true, y_pred):
        y_true_mean = np.mean(y_true)
        ss_total = np.sum((y_true - y_true_mean) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

    def compute_all_metrics(self, y_true, y_pred,set_name): 
        rmse = self.calculate_rmse(y_true, y_pred)
        mae = self.calculate_mae(y_true, y_pred)
        r2 = self.calculate_r2(y_true, y_pred)
        
        self.results = {
            "Evaluation":set_name,
            "RMSE": rmse,
            "MAE": mae,
            "R^2": r2
        }
        return self.results

    def print_results(self,set_name):
        print(f"Evaluation {set_name}")
        print(self.results)