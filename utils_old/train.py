# train.py
import torch
import copy
import torch.nn as nn

def warmup_lr_scheduler(optimizer, warmup_iters, Decrease_percent,continuous=True):
    def f(epoch):
        if epoch >= warmup_iters:
            if continuous is True:
                return Decrease_percent ** (epoch-warmup_iters+1)
            elif continuous is not True:
                return Decrease_percent
        return 1
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def evaluation(model, activation_func_final , epoch, num_epoch, val_epoch_loss_list, criterion, valueMultiply, eval_loader, device,Transformer=None, correlation='', kfoldCV = None):
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
            outputs,_ = model(omics_tensor_dict, drug, device,Transformer) #drug.to(torch.float32)
            
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
            print(f'Evaluation Test Loss: {mean_batch_eval_loss:.10f}')
            return mean_batch_eval_loss
        else :
            print(f'Epoch [{epoch + 1}/{num_epoch}] - mean_batch Validation Loss: {mean_batch_eval_loss:.10f}')
            val_epoch_loss_list.append(mean_batch_eval_loss)
            return mean_batch_eval_loss, val_epoch_loss_list
    # for inference after train epoch loop, and store output for correlation
    elif correlation in ['train', 'val', 'test']:
        print(f'Evaluation {correlation} Loss: {mean_batch_eval_loss:.10f}')
        return mean_batch_eval_loss, eval_targets, eval_outputs
    else:
        print('error occur when correlation argument is not correct')
        return 'error occur when correlation argument is not correct'




def train(model, activation_func_final, optimizer, batch_size, num_epoch,patience, warmup_iters, Decrease_percent, continuous, learning_rate, criterion, valueMultiply, train_loader, val_loader, device,Transformer=None,seed=42, kfoldCV = None):
    # Training with early stopping (assuming you've defined the EarlyStopping logic)
    if warmup_iters is not None:
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, Decrease_percent,continuous)
    best_val_loss = float('inf')
    best_weight=None
    counter = 0
    train_epoch_loss_list = []#  for train every epoch loss plot
    val_epoch_loss_list=[]#  for validation every epoch loss plot
    torch.manual_seed(seed)
    for epoch in range(num_epoch):
        model.train()
        model.requires_grad = True
        total_train_loss = 0.0
        batch_idx_without_nan_count=0 # if a batch has [] empty list than don't count
        for batch_idx,inputs in enumerate(train_loader):
            optimizer.zero_grad()
            omics_tensor_dict,drug, target = inputs[0],inputs[1], inputs[-1].to(device=device)
            outputs,attention_score_matrix = model(omics_tensor_dict, drug, device,Transformer) #drug.to(torch.float32)
            
            mask = ~torch.isnan(target)# Create a mask for non-NaN values in tensor # 去除nan的項
            target = target[mask]# Apply the mask to filter out NaN values from both tensors
            outputs = outputs[mask]
            
            if isinstance(activation_func_final, nn.Sigmoid): # ReLU就不用， Sigmoid要因為只有0~1
                outputs = outputs*valueMultiply
                
            if target.numel() != 0: # 確保batch中traget去除掉nan後還有數值 (count)
                batch_idx_without_nan_count+=1# if 這個batch有數值batch才累加 
                loss = criterion(outputs.reshape(-1), target.to(torch.float32).reshape(-1)) # 
                 
                loss.backward()#計算 grad (partial Loss/partial weight)
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
        print(f'Epoch [{epoch + 1}/{num_epoch}] - mean_batch Training Loss: {mean_batch_train_loss:.10f}')  
        
        mean_batch_val_loss, val_epoch_loss_list = evaluation(model, activation_func_final, epoch, num_epoch, val_epoch_loss_list, criterion, valueMultiply, val_loader, device,Transformer=Transformer, correlation='plotLossCurve',kfoldCV = None)

        if warmup_iters is not None:
            print("lr of epoch", epoch + 1, "=>", lr_scheduler.get_lr()) 
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
    
    return best_epoch, best_weight, best_val_loss, train_epoch_loss_list, val_epoch_loss_list, best_val_loss_mean_batch_train_loss,attention_score_matrix

