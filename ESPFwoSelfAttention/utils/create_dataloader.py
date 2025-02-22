# create_dataloader
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from tqdm import tqdm

class OmicsDrugDataset(Dataset):
    def __init__(self, omics_data_tensor_dict, drug_features_tensor, response_matrix_tensor , splitType, include_omics):
        self.omics_data_tensor_dict = omics_data_tensor_dict
        self.drug_features_tensor = drug_features_tensor
        self.response_matrix_tensor = response_matrix_tensor
        self.splitType = splitType
        self.num_cells = list(self.omics_data_tensor_dict.values())[0].shape[0]
        self.num_drugs = drug_features_tensor.shape[0]
        self.include_omics = include_omics
        self.omics_data_tensor = {}
        self.cell_idx = None  # Initialize first
        self.drug_idx = None  # Initialize first
    def __len__(self):
        return self.num_cells * self.num_drugs
    def __getitem__(self, idx):
        if self.splitType == 'byCCL': 
            self.cell_idx = idx // self.num_drugs  
            self.drug_idx = idx % self.num_drugs   
        elif self.splitType == 'byDrug':
            self.cell_idx = idx % self.num_cells   
            self.drug_idx = idx // self.num_cells   
            
        # wrong code!!!    
        # for omic_type in self.include_omics:
        #     self.omics_data_tensor[omic_type] = self.omics_data_tensor_dict[omic_type][self.cell_idx]
        
        self.omics_data_tensor = {omic_type: self.omics_data_tensor_dict[omic_type][self.cell_idx] 
                             for omic_type in self.include_omics}
        drug_features = self.drug_features_tensor[self.drug_idx]
        response_value = self.response_matrix_tensor[self.cell_idx, self.drug_idx]
        return  self.omics_data_tensor, drug_features, response_value
        # return {'omics': self.omics_data_tensor, 'drug': drug_features, 'response': response_value}

        
def create_dataset(*args, id, batch_size):
    dataset = [arg[id] for arg in args]
    if type(dataset[0]) == torch.Tensor:
        # datas = np.array([torch.tensor(x.detach().cpu().numpy()) for x in tqdm(dataset)], dtype = object)
        # print(type(datas))
        datas = [x for x in tqdm(dataset)]
    else:
        print("type is not torch.Tensor")
    dataset = TensorDataset(*datas)
    print("create dataset finished")
    return dataset

# seed=42
# torch.manual_seed(seed)
# train_dataset = create_dataset(data_mut, data_drug, data_AUC, id = id_train, batch_size=batch_size)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataset = create_dataset(data_mut, data_drug, data_AUC, id = id_val, batch_size=batch_size)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_dataset = create_dataset(data_mut, data_drug, data_AUC, id = id_test, batch_size=batch_size)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)







def create_train_loader(data_mut, data_drug, data_AUC, id_train, batch_size):
    train_dataset = TensorDataset(torch.tensor(data_mut[id_train], dtype=torch.float32),
                                  torch.tensor(data_drug[id_train], dtype=torch.float32),
                                  torch.tensor(data_AUC[id_train], dtype=torch.float32))
    train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
    return train_loader
    
def create_val_loader(data_mut, data_drug, data_AUC, id_val, batch_size):
    val_dataset = TensorDataset(torch.tensor(data_mut[id_val], dtype=torch.float32),
                                  torch.tensor(data_drug[id_val], dtype=torch.float32),
                                  torch.tensor(data_AUC[id_val], dtype=torch.float32))
    val_loader = DataLoader(val_dataset,batch_size=batch_size, shuffle=False)
    return val_loader
    
def create_train_val_loader(data_mut, data_drug, data_AUC, id_train_val, batch_size):
    train_val_dataset = TensorDataset(torch.tensor(data_mut[id_train_val], dtype=torch.float32),
                                  torch.tensor(data_drug[id_train_val], dtype=torch.float32),
                                  torch.tensor(data_AUC[id_train_val], dtype=torch.float32))
    train_val_loader = DataLoader(train_val_dataset,batch_size=batch_size, shuffle=False)
    return train_val_loader

def create_test_loader(data_mut, data_drug, data_AUC, id_test, batch_size):
    test_dataset = TensorDataset(torch.tensor(data_mut[id_test], dtype=torch.float32),
                                  torch.tensor(data_drug[id_test], dtype=torch.float32),
                                  torch.tensor(data_AUC[id_test], dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader