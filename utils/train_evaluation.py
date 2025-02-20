
def train_loop():
    train_outputs = []
    validation_outputs = []
    batch_loss_values_train = []# plot train batch loss
    loss_values_train = []#  for train epoch loss plot
    loss_values_validation=[]#  for validation epoch loss plot
    
    for epoch in range(num_epoch):
        model.train()
        model.requires_grad = True
        total_train_loss = 0.0
        for batch_idx,inputs in enumerate(train_loader):
            optimizer.zero_grad()
            mut, target = inputs[0], inputs[-1]
            mut, target = mut.to(device=device),target.to(device=device)
            outputs = model(mut)
            loss = criterion(outputs, target.unsqueeze(1))
            loss.backward()#計算 grad (partial Loss/partial weight)
            optimizer.step()# W'= W - (lr*delta W)
            
            train_outputs.append(outputs.cpu().detach().numpy().reshape(-1))
            batch_loss = loss.cpu().detach().numpy()
            #print(f'Epoch: [{epoch + 1}/{num_epoch}] BATCH: [{batch_idx+1}/{len(train_loader)}]- batch Validation Loss: {batch_loss:.4f}')
            batch_loss_values_train.append(batch_loss.tolist())
            total_train_loss += batch_loss
        mean_batch_train_loss = total_train_loss/(batch_idx+1)
        loss_values_train.append(mean_batch_train_loss)
        print(f'Epoch [{epoch + 1}/{num_epoch}] - mean_batch Training Loss: {mean_batch_train_loss:.4f}')
        
        model.eval()
        model.requires_grad = False
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_idx,inputs in enumerate(validation_loader):
                mut, target = inputs[0], inputs[-1]
                mut, target = mut.to(device=device),target.to(device=device) 
                outputs = model(mut)
                batch_val_loss = criterion(outputs, target.unsqueeze(1))
                
                validation_outputs.append(outputs.cpu().detach().numpy().reshape(-1))
                batch_val_loss = batch_val_loss.cpu().detach().numpy()
                total_val_loss += batch_val_loss
                #print(f'Epoch: [{epoch + 1}/{num_epoch}] BATCH: [{batch_idx+1}/{len(validation_loader)}]- batch Validation Loss: {batch_val_loss:.4f}')
        mean_batch_val_loss = total_val_loss/(batch_idx+1) # batches' average loss as one epoch's loss
        loss_values_validation.append(mean_batch_val_loss)
        print(f'Epoch [{epoch + 1}/{num_epoch}] - mean_batch Validation Loss: {mean_batch_val_loss:.4f}')


def evaluation():
    model.eval()
    model.requires_grad = False
    test_loss = 0.0
    test_outputs = []
    with torch.no_grad():
        for batch_idx, inputs in enumerate(test_loader):
            mut, target = inputs[0], inputs[-1]
            mut, target = mut.to(device=device),target.to(device=device)
            outputs = model(mut)
            test_loss += criterion(outputs, target.unsqueeze(1))
            assert test_loss.requires_grad ==False# check grad off
            outputs=outputs.cpu().detach().numpy().reshape(-1)
            test_outputs.append(outputs)
        test_loss /= (batch_idx+1)
        print(f'Evaluation Test Loss: {test_loss:.4f}')

    