import numpy as np
import torch
import torch.nn as nn
import torchmetrics
class MetricsCalculator_nntorch(nn.Module):
    def __init__(self,types= []): #
        super(MetricsCalculator_nntorch, self).__init__() # Call nn.Module's init first!
        
        self.types = types
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.mae_loss = nn.L1Loss(reduction="mean") # reduction (default 'mean')
    def _find_best_threshold(self, y_true, y_pred, metric="0.5", thresholds=torch.linspace(0.0, 1.0, steps=500)):
        """
        Find the best threshold that maximizes the given metric (default: 0.5).
        Supported metrics: "F1", "Accuracy", "Precision", "Sensitivity"
        """
        y_true = y_true.detach()
        y_pred = y_pred.detach()
        best_thresh = 0.5
        best_score = -1.0
        metric_class = { "F1": torchmetrics.classification.F1Score(task="binary"),
                         "Accuracy": torchmetrics.classification.Accuracy(task="binary"),
                         "Precision": torchmetrics.classification.Precision(task="binary"),
                         "Sensitivity": torchmetrics.classification.Recall(task="binary"),
                         "Specificity": torchmetrics.classification.Specificity(task="binary")}
        if metric not in metric_class:
            return best_thresh
        else:
            metric_func = metric_class[metric].to(y_true.device)
            for thresh in thresholds:
                pred_bi = (y_pred > thresh).int()
                GT = (y_true > 0.5).int()  # assuming original labels are probabilities or continuous
                score = metric_func(pred_bi, GT)
                if score > best_score:
                    best_score = score
                    best_thresh = thresh.item()
            return best_thresh
    def forward(self, y_true, y_pred, best_prob_threshold, metric, dataset=""): 
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
        if "Accuracy" in self.types:
            if dataset == "val":
                best_prob_threshold = self._find_best_threshold(y_true, y_pred, metric=metric, thresholds=torch.linspace(0.0, 1.0, steps=500))
            print(f"Best threshold for {metric} on {dataset} set: {best_prob_threshold}")
            device=y_true.device
            best_prob_threshold = torch.tensor(best_prob_threshold, dtype=torch.float32, device=device)
            # Binarize labels and predictions based on best_prob_threshold
            GT = (y_true > 0.5).int() # in order to match the type of y_pred # GT already binarized
            pred_bi = (y_pred > best_prob_threshold).int()
            # Compute metrics using torchmetrics
            accuracy = torchmetrics.classification.Accuracy(task="binary").to(device)(pred_bi, GT)
            auroc = torchmetrics.classification.AUROC(task="binary").to(device)(y_pred, GT)  # Use raw scores
            auprc = torchmetrics.classification.AveragePrecision(task="binary").to(device)(y_pred, GT) # Use raw scores
            f1 = torchmetrics.classification.F1Score(task="binary").to(device)(pred_bi, GT)
            sensitivity = torchmetrics.classification.Recall(task="binary").to(device)(pred_bi, GT)
            specificity = torchmetrics.classification.Specificity(task="binary").to(device)(pred_bi, GT)
            precision = torchmetrics.classification.Precision(task="binary").to(device)(pred_bi, GT)
            self.results = {"Accuracy": accuracy,
                            "AUROC": auroc,
                            "AUPRC": auprc,
                            "Sensitivity": sensitivity,
                            "Specificity": specificity,
                            "Precision": precision,
                            "F1": f1}
        return self.results, best_prob_threshold
        
    def confusion_matrix(self, y_true, y_pred, best_prob_threshold):
        # Binarize labels and predictions based on prob_threshold
        GT = (y_true > best_prob_threshold).int()
        pred_bi = (y_pred > best_prob_threshold).int()
        # Count occurrences
        GT_0_count = torch.sum(GT == 0)
        GT_1_count = torch.sum(GT == 1)
        pred_binary_0_count = torch.sum(pred_bi == 0)
        pred_binary_1_count = torch.sum(pred_bi == 1)
        # Compute confusion matrix
        cm = torchmetrics.classification.ConfusionMatrix(task="binary", num_classes=2).to(y_true.device)(pred_bi, GT)
        # self.results["TP"] = cm[1, 1]  # True Positives
        # self.results["TN"] = cm[0, 0]  # True Negatives
        # self.results["FP"] = cm[0, 1]  # False Positives
        # self.results["FN"] = cm[1, 0]  # False Negatives   
        return (cm.cpu().numpy(), GT_0_count.item(),GT_1_count.item(), pred_binary_0_count.item(), pred_binary_1_count.item())

class MetricsCalculator_numpy(nn.Module):
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
        self.results = {"Evaluation":set_name, "MSE": mse, "MAE": mae, "R^2": r2 }
        return self.results
    def print_results(self,set_name):
        print(f"Evaluation {set_name}")
        print(self.results)


