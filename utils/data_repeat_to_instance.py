# data_repeat_to_instance
# Perform the repeat operation on data
import torch

def repeat_data(*args, splitType, num_ccl, num_drug, device):
    # args:tuple
    if all(isinstance(arg, torch.Tensor) for arg in args):
        datalist = [arg for arg in args]
        print("All inputs are torch.Tensors, added to datalist.")
    else:
        datalist = [torch.tensor(arg, dtype=torch.float32).to(device=device) for arg in args]
        print("convert input args to torch.tensor.float32 and add to device for computing")
        
    if splitType == 'byCCL':
        if len(datalist) == 3: # mut/exp、drug、AUC
            # Perform the repeat operation on data_mut or data_exp
            data = datalist[0].unsqueeze(1).repeat(1, num_drug, 1).view(-1, datalist[0].shape[1])
            print(datalist[0].shape)
            print(data.shape)
            
            # Perform the repeat operation on data_drug # Repeat drugs (1~1450)(1~1450).....(1~1450)
            if len(datalist[1].shape) == 3:
                print(datalist[1].shape)
                data_drug = datalist[1].repeat(num_ccl, 1, 1)
            if len(datalist[1].shape) == 2:  
                print(datalist[1].shape)  
                data_drug = datalist[1].repeat(num_ccl, 1)
            print(data_drug.shape)
            
            # Flatten data_AUC
            data_AUC = datalist[2].reshape(-1)
            print(datalist[2].shape)  
            print(data_AUC.shape)
            return data, data_drug, data_AUC
        
        elif len(datalist) == 4:# mut、exp、drug、AUC
            # Perform the repeat operation on data_mut
            data_mut = datalist[0].unsqueeze(1).repeat(1, num_drug, 1).view(-1, datalist[0].shape[1])
            print(data_mut.shape)
            # Perform the repeat operation on data_exp
            data_exp = datalist[1].unsqueeze(1).repeat(1, num_drug, 1).view(-1, datalist[1].shape[1])
            print(data_exp.shape)
            
             # Perform the repeat operation on data_drug # Repeat drugs (1~1450)(1~1450).....(1~1450)
            if len(datalist[2].shape) == 3:# ESPF
                print(datalist[2].shape)
                data_drug = datalist[2].repeat(num_ccl, 1, 1)
            if len(datalist[2].shape) == 2:  
                print(datalist[2].shape)  
                data_drug = datalist[2].repeat(num_ccl, 1)
            print(data_drug.shape)
            
            # Flatten data_AUC
            data_AUC = datalist[3].reshape(-1)
            print(data_AUC.shape)
            return data_mut, data_exp, data_drug, data_AUC
        else:
            print("check the input args")

    elif splitType == 'byDrug':
        if len(datalist) == 3:
            # Perform the repeat operation on data_mut or data_exp
            data = datalist[0].repeat(num_drug, 1)
            print("data",data.shape)
            # print(torch.sum(data[0:1,:]),torch.sum(data[1:2,:]),torch.sum(data[2:3,:]))
            
            # Perform the repeat operation on data_drug
            if len(datalist[1].shape) == 3:
                data_drug = datalist[1].repeat_interleave(num_ccl, dim=0) # [1440, 2, 50]->[685440, 2, 50]
            if len(datalist[1].shape) == 2: 
                data_drug = datalist[1].unsqueeze(1).repeat(1, num_ccl, 1).view(-1, datalist[1].shape[1])
            print("data_drug",data_drug.shape)
            #rint(data_drug[0:2,:,:])
            #rint(data_drug[1440:1442,:,:])
            # Flatten data_AUC
            data_AUC = datalist[2].reshape(-1)
            print("data_AUC",data_AUC.shape)
            return data, data_drug, data_AUC
        elif len(datalist) == 4:
            # Perform the repeat operation on data_mut
            data_mut = datalist[0].repeat(num_drug, 1)
            print(data_mut.shape)
            # Perform the repeat operation on data_exp
            data_exp = datalist[1].repeat(num_drug, 1)
            print(data_exp.shape)
            
            # Perform the repeat operation on data_drug
            if len(datalist[2].shape) == 3:
                data_drug = datalist[2].repeat_interleave(num_ccl, dim=0) # [1440, 2, 50]->[685440, 2, 50]
            if len(datalist[2].shape) == 2: 
                data_drug = datalist[2].unsqueeze(1).repeat(1, num_ccl, 1).view(-1, datalist[2].shape[1])
            print("data_drug",data_drug.shape)
                        
            # Flatten data_AUC
            data_AUC = datalist[3].reshape(-1)
            print(data_AUC.shape)
            return data_mut, data_exp, data_drug, data_AUC
        else:
            print("check the input args")


# usage
#repeat_data(data_mut, data_drug, data_AUC_matrix, num_ccl=num_ccl, num_drug=num_drug, device=device)
#repeat_data(data_exp, data_drug, data_AUC_matrix, num_ccl=num_ccl, num_drug=num_drug, device=device)
#repeat_data(data_mut, data_exp, data_drug, data_AUC_matrix, num_ccl=num_ccl, num_drug=num_drug, device=device)

'''
# previous code
data_mut_tensor = torch.tensor(data_mut, dtype=torch.float32).to(device=device)
data_drug_tensor = torch.tensor(data_drug, dtype=torch.float32).to(device=device)
data_AUC_tensor = torch.tensor(data_AUC_matrix, dtype=torch.float32).to(device=device)

# Perform the repeat operation on data_mut
data_mut = data_mut_tensor.unsqueeze(1).repeat(1, num_drug, 1).view(-1, data_mut_tensor.shape[1])
print(data_mut.shape)

# Perform the repeat operation on data_drug
data_drug = data_drug_tensor.repeat(num_ccl, 1)
print(data_drug.shape)

# Flatten data_AUC
data_AUC = data_AUC_tensor.reshape(-1)
print(data_AUC.shape)
'''

# previous code
def repeat_dataMut_byCCL(data_mut, data_drug, data_AUC_matrix, num_ccl, num_drug, device):
    # transfer data to GPU to do the repeat operation in order to save time and space
    data_mut_tensor = torch.tensor(data_mut, dtype=torch.float32).to(device=device)
    data_drug_tensor = torch.tensor(data_drug, dtype=torch.float32).to(device=device)
    data_AUC_tensor = torch.tensor(data_AUC_matrix, dtype=torch.float32).to(device=device)

    # Perform the repeat operation on data_mut
    data_mut = data_mut_tensor.unsqueeze(1).repeat(1, num_drug, 1).view(-1, data_mut_tensor.shape[1])
    print(data_mut.shape)

    # Perform the repeat operation on data_drug
    data_drug = data_drug_tensor.repeat(num_ccl, 1)
    print(data_drug.shape)

    # Flatten data_AUC
    data_AUC = data_AUC_tensor.reshape(-1)
    print(data_AUC.shape)
    return data_mut, data_drug, data_AUC

def repeat_dataExp_byCCL(data_exp, data_drug, data_AUC_matrix, num_ccl, num_drug, device):
    # transfer data to GPU to do the repeat operation in order to save time and space
    data_exp_tensor = torch.tensor(data_exp, dtype=torch.float32).to(device=device)
    data_drug_tensor = torch.tensor(data_drug, dtype=torch.float32).to(device=device)
    data_AUC_tensor = torch.tensor(data_AUC_matrix, dtype=torch.float32).to(device=device)

    # Perform the repeat operation on data_exp
    data_exp = data_exp_tensor.unsqueeze(1).repeat(1, num_drug, 1).view(-1, data_exp_tensor.shape[1])
    print(data_exp.shape)

    # Perform the repeat operation on data_drug
    data_drug = data_drug_tensor.repeat(num_ccl, 1)
    print(data_drug.shape)

    # Flatten data_AUC
    data_AUC = data_AUC_tensor.reshape(-1)
    print(data_AUC.shape)
    return data_exp, data_drug, data_AUC

def repeat_data_byCCL(data_mut, data_exp, data_drug, data_AUC_matrix, num_ccl, num_drug, device):
    # transfer data to GPU to do the repeat operation in order to save time and space
    data_mut_tensor = torch.tensor(data_mut, dtype=torch.float32).to(device=device)
    data_exp_tensor = torch.tensor(data_exp, dtype=torch.float32).to(device=device)
    data_drug_tensor = torch.tensor(data_drug, dtype=torch.float32).to(device=device)
    data_AUC_tensor = torch.tensor(data_AUC_matrix, dtype=torch.float32).to(device=device)

    # Perform the repeat operation on data_mut
    data_mut = data_mut_tensor.unsqueeze(1).repeat(1, num_drug, 1).view(-1, data_mut_tensor.shape[1])
    print(data_mut.shape)

    # Perform the repeat operation on data_exp
    data_exp = data_exp_tensor.unsqueeze(1).repeat(1, num_drug, 1).view(-1, data_exp_tensor.shape[1])
    print(data_exp.shape)

    # Perform the repeat operation on data_drug
    data_drug = data_drug_tensor.repeat(num_ccl, 1)
    print(data_drug.shape)

    # Flatten data_AUC
    data_AUC = data_AUC_tensor.reshape(-1)
    print(data_AUC.shape)
    return data_mut, data_exp, data_drug, data_AUC
