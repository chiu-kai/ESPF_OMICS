# train.py
from tracemalloc import start
import torch
import copy
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time 

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
def warmup_lr_scheduler(optimizer, warmup_iters, Decrease_percent,continuous=True):
    def f(epoch):
        if epoch >= warmup_iters:
            if continuous is True:
                return Decrease_percent ** (epoch-warmup_iters+1)
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

def evaluation(model, val_epoch_loss_list, criterion, eval_loader, device,ESPF,Drug_SelfAttention, weighted_threshold, few_weight, more_weight, outputcontrol='' ):
    torch.manual_seed(42)
    eval_outputs = [] # for correlation
    eval_targets = [] # for correlation
    eval_outputs_before_final_activation_list = []
    model.eval()
    model.requires_grad = False
    total_eval_loss = np.float32(0.0)
    total_eval_lossWOpenalty = np.float32(0.0)
    batch_idx_without_nan_count=0 # if a batch has [] empty list than don't count
    weight_loss_mask = None
    with torch.no_grad():
        for batch_idx,inputs in enumerate(eval_loader):
            omics_tensor_dict,drug, target = inputs[0],inputs[1], inputs[-1]#.to(device=device)
            model_output = model(omics_tensor_dict, drug, device, **{"ESPF":ESPF,"Drug_SelfAttention":Drug_SelfAttention}) #drug.to(torch.float32)
            outputs = model_output[0]  # model_output[1] # model_output[2] # output.shape(n_sample, 1)
            mask = ~torch.isnan(target)# Create a mask for non-NaN values in tensor # 去除nan的項 # mask.shape(n_sample, 1)
            target = target[mask]# Apply the mask to filter out NaN values from both tensors # target.shape(n_sample, 1)->(n_sample-nan, 1)
            
            predAUCwithUnknownGT = outputs # for unknown GroundTruth
            outputs = outputs[mask] #dtype = 'float32'
            # if isinstance(activation_func_final, nn.Sigmoid): # if ReLU(), no need
            #     outputs = outputs*valueMultiply
            eval_outputs.append(outputs.detach().cpu().numpy().reshape(-1)) #dtype = 'float32'
            eval_targets.append(target.detach().cpu().numpy().reshape(-1))

            if outputcontrol != 'plotLossCurve':
                eval_outputs_before_final_activation_list.append((model_output[3])[mask].detach().cpu().numpy().reshape(-1))
            if target.numel() != 0: # check if a batch do not has [] empty list 
                batch_idx_without_nan_count+=1
                # if isinstance(criterion, (nn.MSELoss, nn.L1Loss)):
                #     batch_val_loss = criterion(outputs.reshape(-1), target.to(torch.float32).reshape(-1))
                #     assert batch_val_loss.requires_grad == False  # Ensure no gradients are computed
                #     total_eval_loss += (batch_val_loss.cpu().detach().numpy())/ (valueMultiply**2 if isinstance(criterion, nn.MSELoss) else valueMultiply)
                # else:  # Custom_LossFunction
                if weighted_threshold is not None:
                    weight_loss_mask = torch.where(target > weighted_threshold, few_weight, more_weight)# Returns few_weight where condition is True.
                batch_val_loss = criterion(outputs.reshape(-1), target.to(torch.float32).reshape(-1), model, weight_loss_mask)
                # assert batch_val_loss.requires_grad == False  # Ensure no gradients are computed
                total_eval_loss += (batch_val_loss.cpu().detach().numpy())
                total_eval_lossWOpenalty += ((criterion.loss).cpu().detach().numpy())#lossWOpenalty
                    
        if batch_idx_without_nan_count > 0:
            mean_batch_eval_loss = (total_eval_loss/(batch_idx_without_nan_count)).astype('float32') # batches' average loss as one epoch's loss
            mean_batch_eval_lossWOpenalty = (total_eval_lossWOpenalty/(batch_idx_without_nan_count)).astype('float32')

        # just for evaluation in train epoch loop , and plot the epochs loss, not for correlation
        if outputcontrol =='plotLossCurve': 
            # print(f'Epoch [{epoch + 1}/{num_epoch}] - mean_batch Validation Loss: {mean_batch_eval_loss:.8f}')
            # val_epoch_loss_list.append(mean_batch_eval_loss)
            val_epoch_loss_list.append(mean_batch_eval_lossWOpenalty)
            return mean_batch_eval_loss, val_epoch_loss_list,mean_batch_eval_lossWOpenalty
        # for inference after train epoch loop, and store output for correlation
        elif outputcontrol == 'correlation':
            # print(f'Evaluation {outputcontrol} Loss: {mean_batch_eval_loss:.8f}')
            return mean_batch_eval_loss, eval_targets, eval_outputs, predAUCwithUnknownGT,mean_batch_eval_lossWOpenalty,eval_outputs_before_final_activation_list
        elif outputcontrol =='inference':
            AttenScorMat_DrugCellSelf = model_output[2]
            return eval_targets, eval_outputs,predAUCwithUnknownGT, AttenScorMat_DrugCellSelf,eval_outputs_before_final_activation_list
        else:
            print('error occur when outputcontrol argument is not correct')
            return 'error occur when outputcontrol argument is not correct'




def train(model, optimizer, batch_size, num_epoch,patience, warmup_iters, Decrease_percent, continuous, learning_rate, criterion, train_loader, val_loader, device,ESPF,Drug_SelfAttention,seed, kfoldCV,weighted_threshold, few_weight, more_weight, TrackGradient=False):
    # Training with early stopping (assuming you've defined the EarlyStopping logic)
    if warmup_iters is not None:
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, Decrease_percent,continuous)
    best_val_loss = float('inf')
    best_val_epoch_train_loss = None
    best_weight=None
    counter = 0
    train_epoch_loss_list = []#  for train every epoch loss plot
    val_epoch_loss_list=[]#  for validation every epoch loss plot
    if TrackGradient is True:
        Grad_tracker = GradientNormTracker(batch_size,check_frequency=10, enable_plot=True)  # Enable or disable plotting

    torch.manual_seed(seed)
    model.train()
    model.requires_grad = True
    for epoch in range(num_epoch):
        total_train_loss = np.float32(0.0)
        total_train_lossWOpenalty = np.float32(0.0)
        batch_idx_without_nan_count=0 # if a batch has [] empty list than don't count
        weight_loss_mask = None
        for batch_idx,inputs in enumerate(train_loader):
            optimizer.zero_grad()
            omics_tensor_dict,drug = inputs[0],inputs[1]
            target = inputs[2]#.to(device=device)
            
            model_output = model(omics_tensor_dict, drug, device,**{"ESPF":ESPF,"Drug_SelfAttention":Drug_SelfAttention}) #drug.to(torch.float32)
            outputs =model_output[0]
            # attention_score_matrix torch.Size([bsz, 8, 50, 50])# softmax(without dropout)
            mask = ~torch.isnan(target)# Create a mask for non-NaN values in tensor # 0:nan, 1:non-nan

            target = target[mask]# Apply the mask to filter out NaN values from both tensors # 去除nan的項 [nan, 0.7908]->[0.7908]
            outputs = outputs[mask]
            # if isinstance(activation_func_final, nn.Sigmoid): # ReLU就不用， Sigmoid要因為只有0~1 所以要跟著targetAUC一起變大
            #     outputs = outputs*valueMultiply
                
            if target.numel() != 0: # 確保batch中target去除掉nan後還有數值 (count)
                batch_idx_without_nan_count+=1# if 這個batch有數值batch才累加 
                # if isinstance(criterion, (nn.MSELoss, nn.L1Loss)):
                #     loss = criterion(outputs.reshape(-1), target.to(torch.float32).reshape(-1))
                # else:  # Custom_LossFunction
                if weighted_threshold is not None:
                    weight_loss_mask = torch.where(target > weighted_threshold, few_weight, more_weight)
                loss = criterion(outputs.reshape(-1), target.to(torch.float32).reshape(-1), model, weight_loss_mask)
                lossWOpenalty = criterion.loss
                # assert loss.requires_grad == True  # Ensure gradients are being computed
                loss.backward()  # Compute gradients
                if TrackGradient is True:
                    gradient_norms_list = Grad_tracker.check_and_log(model)  # Check and log gradient norms
                else:
                    gradient_norms_list = None
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Apply gradient clipping
                optimizer.step()  # Update weights
                total_train_loss += (loss.cpu().detach().numpy()) #/ (valueMultiply**2 if isinstance(criterion, nn.MSELoss) else (valueMultiply if isinstance(criterion, nn.L1Loss) else 1))
                total_train_lossWOpenalty += (lossWOpenalty.cpu().detach().numpy())
                
        if batch_idx_without_nan_count > 0:
            mean_batch_train_loss = (total_train_loss/(batch_idx_without_nan_count)).astype('float32')
            # train_epoch_loss_list.append(mean_batch_train_loss) # mean_batch_train_loss = epoch_train_loss
            mean_batch_train_lossWOpenalty = (total_train_lossWOpenalty/(batch_idx_without_nan_count)).astype('float32')
            train_epoch_loss_list.append(mean_batch_train_lossWOpenalty) # mean_batch_train_loss = epoch_train_loss
            # print(f'Epoch [{epoch + 1}/{num_epoch}] - mean_batch Training Loss: {mean_batch_train_loss:.8f}')  
        
        mean_batch_val_loss, val_epoch_loss_list, mean_batch_eval_lossWOpenalty = evaluation(model, val_epoch_loss_list, criterion, val_loader, device,ESPF,Drug_SelfAttention, weighted_threshold, few_weight, more_weight, outputcontrol='plotLossCurve') # input arg kfoldCV must be None (1)
                                               
        if warmup_iters is not None:
            # print("lr of epoch", epoch + 1, "=>", lr_scheduler.get_lr()) 
            lr_scheduler.step()

        if mean_batch_eval_lossWOpenalty < best_val_loss: # bestepoch
            best_val_loss = mean_batch_eval_lossWOpenalty # bestepoch
            best_weight = copy.deepcopy(model.state_dict()) # best epoch_weight
            best_epoch = epoch+1 # bestepoch
            counter = 0
            best_val_epoch_train_loss = mean_batch_train_loss
            best_val_epoch_train_lossWOpenalty = mean_batch_train_lossWOpenalty
            # best_val_lossWOpenalty = mean_batch_eval_lossWOpenalty
            best_epoch_AttenScorMat_DrugSelf = model_output[1] # torch.Size([bsz, 8, 50, 50])(without dropout)
            if len(model_output) == 3:
                best_epoch_AttenScorMat_DrugCellSelf = model_output[2]
            else:
                best_epoch_AttenScorMat_DrugCellSelf = None
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping after {patience} epochs of no improvement.')
                break
    if TrackGradient is True:    
        gradient_fig = Grad_tracker.plot_gradient_norms()
    else:
        gradient_fig = None
        
    return best_epoch, best_weight, best_val_loss, train_epoch_loss_list, val_epoch_loss_list, best_val_epoch_train_loss,best_val_epoch_train_lossWOpenalty, best_epoch_AttenScorMat_DrugSelf,best_epoch_AttenScorMat_DrugCellSelf, gradient_fig, gradient_norms_list
