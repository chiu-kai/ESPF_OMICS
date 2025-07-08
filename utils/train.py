# train.py
from tracemalloc import start
import torch
import copy
import torch.nn as nn
from torch_geometric.data import Batch
import numpy as np
import matplotlib.pyplot as plt
import os
import time 
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import importlib.util
# import config.py dynamically
# 設定命令列引數
parser = argparse.ArgumentParser(description="import config to main")
parser.add_argument("--config", required=True, help="Path to the config.py file")
args = parser.parse_args()
# 動態載入 config.py
spec = importlib.util.spec_from_file_location("config", args.config)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
# 將 config 模組中的變數導入當前命名空間
for key, value in vars(config).items():
    if not key.startswith("_"):  # 過濾內部變數，例如 __builtins__
        globals()[key] = value

class GradientNormTracker:
    def __init__(self, batch_size,check_frequency=10, enable_plot=True):
        """
        Initializes the gradient norm tracker.
        Args:
            check_frequency (int): Frequency of gradient norm checks (in terms of steps or epochs).
            plot_interval (int): Frequency of plotting gradient norms (in terms of steps or epochs).
            enable_plot (bool): Whether to enable plotting of gradient norms.
        """
        self.check_frequency = check_frequency
        self.enable_plot = enable_plot
        self.gradient_norms = []  # Store gradient norms for plotting
        self.steps = 0  # Track the number of steps or epochs
        self.gradient_fig = None
        self.batch_size = batch_size
    def check_and_log(self, model):
        """
        Calculates and logs the gradient norms for the model parameters.
        Args:
            model (torch.nn.Module): The model to track gradient norms for.
        """
        total_norm = 0.0
        self.steps += 1
        if self.steps % self.check_frequency == 0:
            total_norm = torch.sqrt(torch.sum(torch.stack([param.grad.data.norm(2) ** 2 for param in model.parameters() if param.grad is not None])))
            self.gradient_norms.append(total_norm)
            # print(f"Step {self.steps}: Total Gradient Norm = {total_norm:.4f}")
        return self.gradient_norms

    def plot_gradient_norms(self):
        """
        Plots the gradient norms over time, selecting every nth item for clarity.
        """
        step_interval = self.batch_size//2  # Set the interval for plotting
        if self.enable_plot:    
            self.gradient_fig = plt.figure(figsize=(16, 6))
            gradient_norms_cpu = [gn.cpu().item() if isinstance(gn, torch.Tensor) else gn 
                          for gn in self.gradient_norms[::step_interval]]
            plt.plot(range(1, len(self.gradient_norms) + 1, step_interval), gradient_norms_cpu, marker='o', linestyle='-')
            plt.title("Gradient Norms Over Time")
            plt.xlabel("Step")
            plt.ylabel("Gradient Norm")
            plt.grid()
        elif self.enable_plot is False:
            self.gradient_fig = None   
        return self.gradient_fig

    
# Usage 
# Grad_tracker = GradientNormTracker(check_frequency=1, enable_plot=True)  # Enable or disable plotting
# for epoch in range
#    loss.backward()
#    gradient_norms_list = tracker.check_and_log(model)  # Check and log gradient norms
# gradient_fig = tracker.plot_gradient_norms()
def warmup_lr_scheduler(optimizer, decrese_epoch, Decrease_percent,continuous=True):
    def f(epoch):
        if epoch >= decrese_epoch:
            if continuous is True:
                return Decrease_percent ** (epoch-decrese_epoch+1)
            elif continuous is not True:
                return Decrease_percent
        return 1
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def log_gradient_norms(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:  # Skip parameters without gradients
            param_norm = param.grad.data.norm(2)  # L2 norm (Euclidean distance)# sum(ei^2)^0.5
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f"Total Gradient Norm: {total_norm:.4f}")
    return total_norm

kwargs = {"ESPF":ESPF,"Drug_SelfAttention":Drug_SelfAttention,"DCSA":DCSA}

def evaluation(model, eval_epoch_loss_W_penalty_ls, eval_epoch_loss_WO_penalty_ls, 
               criterion, eval_loader, device,ESPF,Drug_SelfAttention, 
               weighted_threshold, few_weight, more_weight, 
               outputcontrol='' ):
    
    torch.manual_seed(42)
    eval_outputs = [] # for correlation
    eval_targets = [] # for correlation
    eval_outputs_before_final_activation_list = []
    predAUCwithUnknownGT = []
    mean_batch_eval_loss_W_penalty = None #np.float32(0.0)
    mean_batch_eval_loss_WO_penalty= None #np.float32(0.0)
    model.eval()
    model.requires_grad = False
    weight_loss_mask = None
    with torch.no_grad():
        for inputs in eval_loader:
            # omics_tensor_dict,drug, target = inputs[0],inputs[1], inputs[-1]#.to(device=device)
            gene_list, drug_data_list, target_list = inputs # batch
            omics_tensor_dict = {omic: torch.stack([d[omic] for d in gene_list], dim=0) for omic in include_omics}
            target = torch.stack(target_list) # torch.Size([bsz, 1]) # bsz: batch size
            drug = Batch.from_data_list(drug_data_list)
            model_output = model(omics_tensor_dict, drug, device, **kwargs) #drug.to(torch.float32)
            outputs = model_output[0]  # model_output[1] # model_output[2] # output.shape(n_sample, 1)
            mask = ~torch.isnan(target)# Create a mask for non-NaN values in tensor # 去除nan的項 # mask.shape(n_sample, 1)
            target = target[mask]# Apply the mask to filter out NaN values from both tensors # target.shape(n_sample, 1)->(n_sample-nan, 1)
            
            predAUCwithUnknownGT.append(outputs.detach().cpu().numpy().reshape(-1))# for unknown GroundTruth
            outputs = outputs[mask] #dtype = 'float32'
            
            eval_outputs.append(outputs.detach().reshape(-1)) #dtype = 'float32' # [tensor]
            eval_targets.append(target.detach().reshape(-1)) # [tensor]
            if outputcontrol != 'plotLossCurve':
                eval_outputs_before_final_activation_list.append((model_output[3])[mask].detach().cpu().numpy().reshape(-1))

            if 'weighted' in criterion.loss_type :    
                if 'BCE' in criterion.loss_type :
                    weight_loss_mask = torch.where(torch.cat(eval_targets) == 1, few_weight, more_weight) # 手動對正樣本給 few_weight 倍權重，負樣本給 more_weight 倍                        
                else:
                    weight_loss_mask = torch.where(torch.cat(eval_targets) > weighted_threshold, few_weight, more_weight)
            
        mean_batch_eval_loss_W_penalty = criterion(torch.cat(eval_outputs),torch.cat(eval_targets), model, weight_loss_mask)# with weighted loss # without batch effect the loss
        mean_batch_eval_loss_WO_penalty = criterion.loss_WO_penalty.cpu().detach().numpy()

        # just for evaluation in train epoch loop , and plot the epochs loss, not for correlation
        if outputcontrol =='plotLossCurve': 
            # print(f'Epoch [{epoch + 1}/{num_epoch}] - mean_batch Validation Loss: {mean_batch_eval_loss:.8f}')
            eval_epoch_loss_W_penalty_ls.append(mean_batch_eval_loss_W_penalty.cpu().detach().numpy() )# 
            eval_epoch_loss_WO_penalty_ls.append(mean_batch_eval_loss_WO_penalty )
            return (eval_targets, eval_outputs,
                    eval_epoch_loss_W_penalty_ls,  eval_epoch_loss_WO_penalty_ls,  
                    mean_batch_eval_loss_WO_penalty)
        # for inference after train epoch loop, and store output for correlation
        elif outputcontrol == 'correlation':
            # print(f'Evaluation {outputcontrol} Loss: {mean_batch_eval_loss:.8f}')
            return (eval_targets, eval_outputs,mean_batch_eval_loss_WO_penalty,
                    eval_outputs_before_final_activation_list)
        elif outputcontrol =='inference':
            AttenScorMat_DrugSelf = model_output[1]
            AttenScorMat_DrugCellSelf = model_output[2]
            return (eval_targets, eval_outputs,predAUCwithUnknownGT, 
                    AttenScorMat_DrugSelf,AttenScorMat_DrugCellSelf,
                    eval_outputs_before_final_activation_list, mean_batch_eval_loss_WO_penalty)
        else:
            print('error occur when outputcontrol argument is not correct')
            return 'error occur when outputcontrol argument is not correct'



def train(model, optimizer, 
          criterion, train_loader, val_loader, device,
          ESPF,Drug_SelfAttention,seed,
          weighted_threshold, few_weight, more_weight, TrackGradient=False):
    
    # Training with early stopping (assuming you've defined the EarlyStopping logic)
    if warmup_lr is True:
        lr_scheduler = warmup_lr_scheduler(optimizer, decrese_epoch, Decrease_percent,continuous)
    if CosineAnnealing_LR is True:
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    if TrackGradient is True:
        Grad_tracker = GradientNormTracker(batch_size,check_frequency=10, enable_plot=True)  # Enable or disable plotting

    BE_val_loss_WO_penalty = float('inf')
    BE_val_train_loss_WO_penalty = None
    best_weight=None
    counter = 0
    val_epoch_loss_W_penalty_ls = []
    val_epoch_loss_WO_penalty_ls = [] # for validation every epoch loss plot
    train_epoch_loss_W_penalty_ls = []
    train_epoch_loss_WO_penalty_ls = []
    torch.manual_seed(seed)
    model.train()
    model.requires_grad = True
    for epoch in range(num_epoch):
        batch_idx_without_nan_count=0 # if a batch has [] empty list than don't count
        weight_loss_mask = None
        for batch_idx,inputs in enumerate(train_loader):
            optimizer.zero_grad()
            
            gene_list, drug_data_list, target_list = inputs # batch
            omics_tensor_dict = {omic: torch.stack([d[omic] for d in gene_list], dim=0) for omic in include_omics}
            target = torch.stack(target_list) # torch.Size([bsz, 1]) # bsz: batch size
            drug = Batch.from_data_list(drug_data_list)

            model_output = model(omics_tensor_dict, drug, device,**kwargs) #drug.to(torch.float32)
            outputs =model_output[0]
            # attention_score_matrix torch.Size([bsz, 8, 50, 50])# softmax(without dropout)
            mask = ~torch.isnan(target)# Create a mask for non-NaN values in tensor # 0:nan, 1:non-nan
            target = target[mask]# Apply the mask to filter out NaN values from both tensors # 去除nan的項 [nan, 0.7908]->[0.7908]
            outputs = outputs[mask]

            if target.numel() != 0: # 確保batch中target去除掉nan後還有數值 (count)
                batch_idx_without_nan_count+=1# if 這個batch有數值batch才累加 
                if 'weighted' in criterion.loss_type :    
                    if 'BCE' in criterion.loss_type :
                        weight_loss_mask = torch.where(target == 1, few_weight, more_weight) # 手動對正樣本給 few_weight 倍權重，負樣本給 more_weight 倍
                    else:
                        weight_loss_mask = torch.where(target > weighted_threshold, few_weight, more_weight)
                loss = criterion(outputs.reshape(-1), target.to(torch.float32).reshape(-1), model, weight_loss_mask)
                # assert loss.requires_grad == True  # Ensure gradients are being computed
                loss.backward()  # Compute gradients
                if TrackGradient is True:
                    gradient_norms_list = Grad_tracker.check_and_log(model)  # Check and log gradient norms
                else:
                    gradient_norms_list = None
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Apply gradient clipping
                optimizer.step()  # Update weights


        (val_targets, val_outputs,
         val_epoch_loss_W_penalty_ls,  val_epoch_loss_WO_penalty_ls, 
         mean_batch_val_loss_WO_penalty) = evaluation(model, 
                                                    val_epoch_loss_W_penalty_ls, val_epoch_loss_WO_penalty_ls, 
                                                    criterion, val_loader, device, ESPF, Drug_SelfAttention, 
                                                    weighted_threshold, few_weight, more_weight, 
                                                    outputcontrol='plotLossCurve') 
        (train_targets, train_outputs,
         train_epoch_loss_W_penalty_ls,  train_epoch_loss_WO_penalty_ls, 
         mean_batch_train_loss_WO_penalty) = evaluation(model, 
                                                        train_epoch_loss_W_penalty_ls, train_epoch_loss_WO_penalty_ls, 
                                                        criterion, train_loader, device, ESPF, Drug_SelfAttention, 
                                                        weighted_threshold, few_weight, more_weight, 
                                                        outputcontrol='plotLossCurve') 
        
        
        if decrese_epoch is not None:
            # print("lr of epoch", epoch + 1, "=>", lr_scheduler.get_lr()) 
            lr_scheduler.step()

        if mean_batch_val_loss_WO_penalty < BE_val_loss_WO_penalty: # BEpo
            BE_val_loss_WO_penalty = mean_batch_val_loss_WO_penalty # BEpo
            BE_val_train_loss_WO_penalty = mean_batch_train_loss_WO_penalty
            best_weight = copy.deepcopy(model.state_dict()) # best epoch_weight
            best_epoch = epoch+1 # BEpo
            counter = 0
            BE_val_targets, BE_val_outputs  = val_targets, val_outputs
            BE_train_targets , BE_train_outputs = train_targets , train_outputs
            BEpo_valLoss_W_penalty_ls = val_epoch_loss_W_penalty_ls
            BEpo_valLoss_WO_penalty_ls = val_epoch_loss_WO_penalty_ls
            BEpo_trainloss_W_penalty_ls = train_epoch_loss_W_penalty_ls
            BEpo_trainLoss_WO_penalty_ls = train_epoch_loss_WO_penalty_ls


        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping after {patience} epochs of no improvement.')
                break
    if TrackGradient is True:    
        gradient_fig = Grad_tracker.plot_gradient_norms()
    else:
        gradient_fig = None
        
    # print("BE_val_outputs",BE_val_outputs[:10],"\n","\n")
    # print("val_outputs",val_outputs[:10])
    
    return (best_epoch, best_weight, BE_val_loss_WO_penalty, BE_val_train_loss_WO_penalty,
            BEpo_trainloss_W_penalty_ls, BEpo_trainLoss_WO_penalty_ls,
            BEpo_valLoss_W_penalty_ls, BEpo_valLoss_WO_penalty_ls,  
            BE_val_targets, BE_val_outputs, BE_train_targets , BE_train_outputs,
            gradient_fig, gradient_norms_list
            )

