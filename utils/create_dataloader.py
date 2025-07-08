# create_dataloader
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from DAPL.dataprocess import smile_to_graph, cat_tensor_with_drug
from torch_geometric.data import Data, Batch

class InstanceResponseDataset(torch.utils.data.Dataset):
    def __init__(self, response_df, expression_df, drug_smiles_df,drug_graph,include_omics, device):
        """
        response_df: 包含 DepMap_ID, DRUG_NAME, Class 的 dataframe
        expression_df: 索引為 DepMap_ID 的 expression 數值 dataframe
        drug_smiles_df: 索引為 DRUG_NAME 的 smiles 結構 dataframe (欄位 'SMILES')
        """
        self.response_df = response_df.reset_index(drop=True)
        self.drug_smiles_df = drug_smiles_df
        self.include_omics = include_omics
        self.device = device
        # 預先建立 expression 查表字典：key 為 sample_id，value 為對應 tensor
        self.expr_dict = defaultdict(dict)
        for omic_type in self.include_omics:
            for sample_id in expression_df[omic_type].index:
                # 轉為 numpy array，再轉為 tensor
                expr = expression_df[omic_type].loc[sample_id].values.astype(np.float32)
                # print("expr", expr.shape)
                self.expr_dict[omic_type][sample_id] = torch.tensor(expr, dtype=torch.float32).to(self.device)
        # 預先計算每個藥物的 graph，存入 dict
        self.drug_graph_dict = {}
        for drug_id in self.drug_smiles_df.index:
            if drug_graph is True:
                drug_smile = self.drug_smiles_df.loc[drug_id]['SMILES']
                c_size, atom_features_list, edge_index = smile_to_graph(drug_smile)
                drug_x = torch.tensor(np.array(atom_features_list), dtype=torch.float32)
                drug_edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                drug_data = Data(x=drug_x, edge_index=drug_edge_index)
                self.drug_graph_dict[drug_id] = drug_data.to(self.device)
            else:
                drug_encode = self.drug_smiles_df.loc[drug_id]["drug_encode"]
                self.drug_graph_dict[drug_id] = torch.tensor(np.array(drug_encode), dtype=torch.long).to(self.device)
    def __len__(self):
        return len(self.response_df)
    def __getitem__(self, idx):
        row = self.response_df.iloc[idx]
        sample_id = row['ModelID']
        drug_id = row['drug_name']
        target = float(row['Label'])
        # 從查表字典中直接取得 gene features
        gene_feature = {omic_type: self.expr_dict[omic_type][sample_id]
                             for omic_type in self.include_omics}
        # gene_feature = self.expr_dict[sample_id]
        # 從預先計算好的字典中取出對應的藥物 graph
        drug_data = self.drug_graph_dict[drug_id]
        target = torch.tensor(target, dtype=torch.float32).to(self.device)
        return gene_feature, drug_data, target


class OmicsDrugDataset(Dataset):
    def __init__(self, omics_data_tensor_dict, drug_features_tensor, response_matrix_tensor , splitType, include_omics):
        self.omics_data_tensor_dict = omics_data_tensor_dict
        self.drug_features_tensor = drug_features_tensor
        self.response_matrix_tensor = response_matrix_tensor
        self.splitType = splitType
        self.num_cells = list(self.omics_data_tensor_dict.values())[0].shape[0]
        self.num_drugs = drug_features_tensor.shape[0]
        print("self.num_drugs",self.num_drugs)
        self.include_omics = include_omics
        self.omics_data_tensor = {}
        self.cell_idx = None  # Initialize first
        self.drug_idx = None  # Initialize first
    def __len__(self):
        return self.num_cells * self.num_drugs
    def __getitem__(self, idx):
        if self.splitType in ['byCCL', 'ModelID']:
            self.cell_idx = idx // self.num_drugs  
            self.drug_idx = idx % self.num_drugs   
        elif self.splitType in ['byDrug', 'drug_name']:
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