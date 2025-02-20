# train.py
import torch
import copy
import torch.nn as nn
import matplotlib.pyplot as plt
import os

class GradientNormTracker:
    def __init__(self, batch_size,check_frequency=1, enable_plot=True):
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

def evaluation(model, activation_func_final , epoch, num_epoch, val_epoch_loss_list, criterion, valueMultiply, eval_loader, device,ESPF,Transformer, correlation='', kfoldCV = None):
    torch.manual_seed(42)
    eval_outputs = [] # for correlation
    eval_targets = [] # for correlation
    model.eval()
    model.requires_grad = False
    total_eval_loss = 0.0
    batch_idx_without_nan_count=0 # if a batch has [] empty list than don't count
    with torch.no_grad():
        for batch_idx,inputs in enumerate(eval_loader):
            omics_tensor_dict,drug, target = inputs[0],inputs[1], inputs[-1].to(device=device)
            outputs,_ = model(omics_tensor_dict, drug, device,ESPF,Transformer) #drug.to(torch.float32)
            
            mask = ~torch.isnan(target)# Create a mask for non-NaN values in tensor # 去除nan的項
            target = target[mask]# Apply the mask to filter out NaN values from both tensors
            outputs = outputs[mask]
            if isinstance(activation_func_final, nn.Sigmoid): # if ReLU(), no need
                outputs = outputs*valueMultiply
            eval_outputs.append(outputs.detach().cpu().numpy().reshape(-1))
            eval_targets.append(target.detach().cpu().numpy().reshape(-1))
            
            if target.numel() != 0: # check if a batch do not has [] empty list 
                batch_idx_without_nan_count+=1
                batch_val_loss = criterion(outputs.reshape(-1), target.to(torch.float32).reshape(-1))
                
                assert batch_val_loss.requires_grad ==False# check requires_grad off
                if isinstance(criterion, nn.MSELoss):
                    total_eval_loss += (batch_val_loss.cpu().detach().numpy())/ (valueMultiply**2)
                elif isinstance(criterion, nn.L1Loss):
                    total_eval_loss += (batch_val_loss.cpu().detach().numpy())/ (valueMultiply)
        # print("outputs",outputs)
        if batch_idx_without_nan_count > 0:
            # print("batch_val_loss",batch_val_loss)  
            # print("total_eval_loss",total_eval_loss,"batch_idx_without_nan_count",batch_idx_without_nan_count)
            mean_batch_eval_loss = total_eval_loss/(batch_idx_without_nan_count) # batches' average loss as one epoch's loss
    # just for evaluation in train epoch loop , and plot the epochs loss, not for correlation
    if correlation=='plotLossCurve': 
        if kfoldCV is True: # for kfold testset evaluation to pick the best fold on 
            # print(f'Evaluation Test Loss: {mean_batch_eval_loss:.8f}')
            return mean_batch_eval_loss
        else :
            # print(f'Epoch [{epoch + 1}/{num_epoch}] - mean_batch Validation Loss: {mean_batch_eval_loss:.8f}')
            val_epoch_loss_list.append(mean_batch_eval_loss)
            return mean_batch_eval_loss, val_epoch_loss_list
    # for inference after train epoch loop, and store output for correlation
    elif correlation in ['train', 'val', 'test']:
        # print(f'Evaluation {correlation} Loss: {mean_batch_eval_loss:.8f}')
        return mean_batch_eval_loss, eval_targets, eval_outputs
    else:
        print('error occur when correlation argument is not correct')
        return 'error occur when correlation argument is not correct'




def train(model, activation_func_final, optimizer, batch_size, num_epoch,patience, warmup_iters, Decrease_percent, continuous, learning_rate, criterion, valueMultiply, train_loader, val_loader, device,ESPF,Transformer,seed=42, kfoldCV = None):
    # Training with early stopping (assuming you've defined the EarlyStopping logic)
    if warmup_iters is not None:
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, Decrease_percent,continuous)
    best_val_loss = float('inf')
    best_weight=None
    counter = 0
    train_epoch_loss_list = []#  for train every epoch loss plot
    val_epoch_loss_list=[]#  for validation every epoch loss plot
    Grad_tracker = GradientNormTracker(batch_size,check_frequency=1, enable_plot=True)  # Enable or disable plotting

    torch.manual_seed(seed)
    for epoch in range(num_epoch):
        model.train()
        model.requires_grad = True
        total_train_loss = 0.0
        batch_idx_without_nan_count=0 # if a batch has [] empty list than don't count
        for batch_idx,inputs in enumerate(train_loader):
            optimizer.zero_grad()
            omics_tensor_dict,drug, target = inputs[0],inputs[1], inputs[-1].to(device=device)
            outputs,attention_score_matrix = model(omics_tensor_dict, drug, device,ESPF,Transformer) #drug.to(torch.float32)
            
            mask = ~torch.isnan(target)# Create a mask for non-NaN values in tensor # 去除nan的項
            target = target[mask]# Apply the mask to filter out NaN values from both tensors
            outputs = outputs[mask]
            
            if isinstance(activation_func_final, nn.Sigmoid): # ReLU就不用， Sigmoid要因為只有0~1
                outputs = outputs*valueMultiply
                
            if target.numel() != 0: # 確保batch中traget去除掉nan後還有數值 (count)
                batch_idx_without_nan_count+=1# if 這個batch有數值batch才累加 
                loss = criterion(outputs.reshape(-1), target.to(torch.float32).reshape(-1)) # 
                 
                loss.backward()#計算 grad (partial Loss/partial weight)
                gradient_norms_list = Grad_tracker.check_and_log(model)  # Check and log gradient norms
                optimizer.step()# W'= W - (lr*delta W)
                if isinstance(criterion, nn.MSELoss):
                    total_train_loss += (loss.cpu().detach().numpy())/ (valueMultiply**2)
                elif isinstance(criterion, nn.L1Loss):
                    total_train_loss += (loss.cpu().detach().numpy())/ (valueMultiply)
        # print("outputs",outputs)
        # print("train_loss",loss) 
        # print("total_train_loss",total_train_loss,"batch_idx_without_nan_count",batch_idx_without_nan_count) 
        mean_batch_train_loss = total_train_loss/(batch_idx_without_nan_count)
        train_epoch_loss_list.append(mean_batch_train_loss) # mean_batch_train_loss = epoch_train_loss
        # print(f'Epoch [{epoch + 1}/{num_epoch}] - mean_batch Training Loss: {mean_batch_train_loss:.8f}')  
        
        mean_batch_val_loss, val_epoch_loss_list = evaluation(model, activation_func_final, epoch, num_epoch, val_epoch_loss_list, criterion, valueMultiply, val_loader, device,ESPF,Transformer, correlation='plotLossCurve',kfoldCV = None)

        if warmup_iters is not None:
            # print("lr of epoch", epoch + 1, "=>", lr_scheduler.get_lr()) 
            lr_scheduler.step()

        if mean_batch_val_loss < best_val_loss: # bestepoch
            best_val_loss = mean_batch_val_loss # bestepoch
            best_weight = copy.deepcopy(model.state_dict()) # bestepoch_weight
            best_epoch = epoch+1 # bestepoch
            counter = 0
            if kfoldCV is True:
                best_val_loss_mean_batch_train_loss = mean_batch_train_loss
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping after {patience} epochs of no improvement.')
                break
    if kfoldCV is not True:
        best_val_loss_mean_batch_train_loss = None

    gradient_fig = Grad_tracker.plot_gradient_norms()

    return best_epoch, best_weight, best_val_loss, train_epoch_loss_list, val_epoch_loss_list, best_val_loss_mean_batch_train_loss,attention_score_matrix, gradient_fig, gradient_norms_list

