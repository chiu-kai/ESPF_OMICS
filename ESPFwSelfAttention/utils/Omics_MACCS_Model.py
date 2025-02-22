# Omics_MACCS_Model.py
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# Modules------------------------------------------------------------------------------------------------------------------------------------------------------        
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon
    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
    
# End of Modules------------------------------------------------------------------------------------------------------------------------------------------------------

# Models------------------------------------------------------------------------------------------------------------------------------------------------------

class Omics_MACCS_Model(nn.Module):#Omics_DrugESPF_Model
    def __init__(self,omics_encode_dim_dict,drug_encode_dims, activation_func,activation_func_final,dense_layer_dim, device, 
                omics_numfeaetures_dict, TCGA_pretrain_weight_path_dict=None):
        super(Omics_MACCS_Model, self).__init__()

        def load_TCGA_pretrain_weight(model, pretrained_weights_path, device):
            state_dict = torch.load(pretrained_weights_path, map_location=device)  # Load the state_dict
            encoder_state_dict = {key[len("encoder."):]: value for key, value in state_dict.items() if key.startswith('encoder')}  # Extract encoder weights
            model.load_state_dict(encoder_state_dict)  # Load only the encoder part
            model_keys = set(model.state_dict().keys())  # Check if the keys match
            loaded_keys = set(encoder_state_dict.keys())
            if model_keys == loaded_keys:
                print(f"State_dict for {model} loaded successfully.")
            else:
                print(f"State_dict does not match the model's architecture for {model}.")
                print("Model keys: ", model_keys, " Loaded keys: ", loaded_keys)
        def _init_weights(model):
            for layer in model:
                if isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
                    if layer.bias is not None:
                        init.zeros_(layer.bias)

# Create subnetworks for each omic type dynamically
        self.MLP4omics_dict = nn.ModuleDict()
        for omic_type in omics_numfeaetures_dict.keys():
            self.MLP4omics_dict[omic_type] = nn.Sequential(
                nn.Linear(omics_numfeaetures_dict[omic_type], omics_encode_dim_dict[omic_type][0]),
                activation_func,
                nn.Linear(omics_encode_dim_dict[omic_type][0], omics_encode_dim_dict[omic_type][1]),
                activation_func_final,
                nn.Linear(omics_encode_dim_dict[omic_type][1], omics_encode_dim_dict[omic_type][2])
            )
            # Initialize with TCGA pretrain weight
            if TCGA_pretrain_weight_path_dict is not None:
                load_TCGA_pretrain_weight(self.MLP4omics_dict[omic_type], TCGA_pretrain_weight_path_dict[omic_type], device)
            else: # Initialize weights with Kaiming uniform initialization, bias with aero
                _init_weights(self.MLP4omics_dict[omic_type])
        
# Define subnetwork for drug 166features
        self.MLP4MACCS = nn.Sequential( # 166->[110,55,22]
            nn.Linear(166, drug_encode_dims[0]),
            activation_func,
            # nn.Dropout(hidden_dropout_prob),
            nn.Linear(drug_encode_dims[0], drug_encode_dims[1]),
            activation_func,
            # nn.Dropout(hidden_dropout_prob),
            nn.Linear(drug_encode_dims[1], drug_encode_dims[2]),
            activation_func)
        # Initialize weights with Kaiming uniform initialization, bias with aero
        _init_weights(self.MLP4MACCS)

# Define the final prediction network 
        self.model_final_add = nn.Sequential(
            nn.Linear(dense_layer_dim, dense_layer_dim),
            activation_func,
            nn.Dropout(p=0),
            nn.Linear(dense_layer_dim, dense_layer_dim),
            activation_func,
            nn.Dropout(p=0),
            nn.Linear(dense_layer_dim, 1),
            activation_func_final)
        # Initialize weights with Kaiming uniform initialization, bias with aero
        _init_weights(self.model_final_add)

    def forward(self, omics_tensor_dict,drug, device):

        omic_embeddings = []
        # Loop through each omic type and pass through its respective model
        for omic_type, omic_tensor in omics_tensor_dict.items():
            omic_embed = self.MLP4omics_dict[omic_type](omic_tensor.to(device=device))
            omic_embeddings.append(omic_embed)
        omic_embeddings = torch.cat(omic_embeddings, dim=1)  # change list to tensor, because omic_embeddings need to be tensor to torch.cat([omic_embeddings, drug_emb_masked], dim=1) 
        
        # Drug feature
        drug = drug.to(torch.float32) # long -> float, because the input of linear layer should be float,才能和float的weight相乘
        drug_final_emb = self.MLP4MACCS(drug.to(device=device))# 166->[110,55,22] # to device, because the weight is on device
        
        
        # Concatenate embeddings from all subnetworks
        combined_mut_drug_embed = torch.cat([omic_embeddings, drug_final_emb], dim=1)#dim=1: turn into 1D
        output = self.model_final_add(combined_mut_drug_embed)
        return output
    


        