import torch
import torch.nn as nn
import torch.nn.init as init

activation_func = nn.ReLU()  # ReLU activation function
activation_func_final = nn.Identity()  # no activation is applied for unbounded scores
dense_layer_dim = 250

mut_encode_dim =[1000,100,50] # [1000,200,50]
exp_encode_dim =[500,200,50] # [1000,200,50]
drug_encode_dim = [100,50,20] # [1000,100,50]

# Define the model architecture in PyTorch
class Mut_Drug_Model(nn.Module):
    def __init__(self, num_mut_features, num_drug_features, device,  TCGA_pretrain_weight_path=None):
        super(Mut_Drug_Model, self).__init__()
        
# Define subnetworks for mutations
        self.model_mut = nn.Sequential(
            nn.Linear(num_mut_features,mut_encode_dim[0]), # model_mut[0] : (Linear(in_features=2649, out_features=1000, bias=True)
            activation_func, #model_mut[1] : ReLU()
            nn.Linear(mut_encode_dim[0], mut_encode_dim[1]), # model_mut[2]
            activation_func,
            nn.Linear(mut_encode_dim[1], mut_encode_dim[2]),
            activation_func)

        if TCGA_pretrain_weight_path is not None:
            # Load the state_dict #TCGA AE Pretrained Weights
            state_dict = torch.load(TCGA_pretrain_weight_path, map_location=device)
            # match the layer name to load weight
            encoder_state_dict = {key[len("encoder."):]: value for key, value in state_dict.items() if key.startswith('encoder')}
            # Load only the encoder part
            self.model_mut.load_state_dict(encoder_state_dict)
            # Check if the keys match
            model_keys = set(self.model_mut.state_dict().keys())
            loaded_keys = set(encoder_state_dict.keys())
            if model_keys == loaded_keys:
                print("State_dict loaded successfully.")
            else:
                print("State_dict does not match the model's architecture.")
                print("Model keys: ", model_keys)
                print("Loaded keys: ", loaded_keys)
            
# Define subnetwork for drug 166features
        self.model_drug = nn.Sequential(
            nn.Linear(num_drug_features, drug_encode_dim[0]),
            activation_func,
            nn.Linear(drug_encode_dim[0], drug_encode_dim[1]),
            activation_func,
            nn.Linear(drug_encode_dim[1], drug_encode_dim[2]),
            activation_func)
        # Initialize both weights and biases with Kaiming uniform initialization
        for layer in self.model_drug:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

# Calculate final input dimension
        final_input_dim = mut_encode_dim[-1]+ drug_encode_dim[-1]            
# Define the final prediction network 
        self.model_final_add = nn.Sequential(
            nn.Linear(final_input_dim, dense_layer_dim),
            activation_func,
            nn.Linear(dense_layer_dim, dense_layer_dim),
            activation_func,
            nn.Linear(dense_layer_dim, 1),
            activation_func_final)
        # Initialize both weights and biases with Kaiming uniform initialization
        for layer in self.model_final_add:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

        
    def forward(self, mut,drug):
        mut_embed = self.model_mut(mut)
        drug_embed = self.model_drug(drug)
        # Concatenate embeddings from all subnetworks
        combined_mut_drug_embed = torch.cat([mut_embed, drug_embed], dim=1)#dim=1: turn into 1D
        output = self.model_final_add(combined_mut_drug_embed)
        return output



# Define the model architecture in PyTorch
class OnlyMutModel(nn.Module):
    def __init__(self, num_mut_features,num_drug, device,  TCGA_pretrain_weight_path=None):
        super(OnlyMutModel, self).__init__()
        
# Define subnetworks for mutations
        self.model_mut = nn.Sequential(
            nn.Linear(num_mut_features,mut_encode_dim[0]), # model_mut[0] : (Linear(in_features=2649, out_features=1000, bias=True)
            activation_func, #model_mut[1] : ReLU()
            nn.Linear(mut_encode_dim[0], mut_encode_dim[1]), # model_mut[2]
            activation_func,
            nn.Linear(mut_encode_dim[1], mut_encode_dim[2]),
            activation_func)
        # Load the state_dict #Pretrain Weights
        state_dict = torch.load(pretrain_weight_path, map_location=device)
        # Create a new state_dict with modified keys
        new_state_dict = {}
        # Map the keys from the old state_dict to the new state_dict
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                new_key = "" + key[len("encoder."):]
            elif key.startswith("decoder."):
                continue
            else:
                continue  # Keep other keys as they are
            new_state_dict[new_key] = value
        # Load the modified state_dict into the model
        self.model_mut.load_state_dict(new_state_dict)
        
        # Check if the keys match
        model_keys = set(self.model_mut.state_dict().keys())
        loaded_keys = set(new_state_dict.keys())
        if model_keys == loaded_keys:
            print("State_dict loaded successfully.")
        else:
            print("State_dict does not match the model's architecture.")
            print("Model keys: ", model_keys)
            print("Loaded keys: ", loaded_keys)
            
# Calculate final input dimension
        final_input_dim = mut_encode_dim[-1]
# Define the final prediction network for regression
        self.model_final_mut = nn.Sequential(
            nn.Linear(final_input_dim, dense_layer_dim),
            activation_func,
            nn.Linear(dense_layer_dim, dense_layer_dim),
            activation_func,
            nn.Linear(dense_layer_dim, num_drug), # output for 1450 drugs
            activation_func_final)
        # Initialize both weights and biases with Kaiming uniform initialization
        for layer in self.model_final_mut:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)
    
    def forward(self, mut):
        mut_embed = self.model_mut(mut)
        output = self.model_final_mut(mut_embed)
        return output







# Define the model architecture in PyTorch
class Mut_Exp_Drug_Model(nn.Module):
    def __init__(self, num_mut_features, num_drug_features, num_exp_features, device, TCGA_pretrain_weight_path_mut=None, TCGA_pretrain_weight_path_exp=None):
        super(Mut_Exp_Drug_Model, self).__init__()

# Define subnetworks for mutations
        self.model_mut = nn.Sequential(
            nn.Linear(num_mut_features,mut_encode_dim[0]), # model_mut[0] : (Linear(in_features=2649, out_features=1000, bias=True)
            activation_func, #model_mut[1] : ReLU()
            nn.Linear(mut_encode_dim[0], mut_encode_dim[1]), # model_mut[2]
            activation_func,
            nn.Linear(mut_encode_dim[1], mut_encode_dim[2]),
            activation_func)

        if TCGA_pretrain_weight_path_mut is not None:
            state_dict = torch.load(TCGA_pretrain_weight_path_mut, map_location=device)# Load the state_dict #TCGA AE Pretrained Weights
            encoder_state_dict = {key[len("encoder."):]: value for key, value in state_dict.items() if key.startswith('encoder')}# match the layer name to load weight
            self.model_mut.load_state_dict(encoder_state_dict) # Load only the encoder part
            model_keys = set(self.model_mut.state_dict().keys()) # Check if the keys match
            loaded_keys = set(encoder_state_dict.keys())
            if model_keys == loaded_keys:
                print("State_dict loaded successfully.")
            else:
                print("State_dict does not match the model's architecture.")
                print("Model keys: ", model_keys, "  Loaded keys: ", loaded_keys)

# Define subnetwork for drug 166features
        self.model_drug = nn.Sequential(
            nn.Linear(num_drug_features, drug_encode_dim[0]),
            activation_func,
            nn.Linear(drug_encode_dim[0], drug_encode_dim[1]),
            activation_func,
            nn.Linear(drug_encode_dim[1], drug_encode_dim[2]),
            activation_func)
        # Initialize both weights and biases with Kaiming uniform initialization
        for layer in  self.model_drug:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

# Define subnetworks for Expressions
        self.model_exp = nn.Sequential(
            nn.Linear(num_exp_features,exp_encode_dim[0]), #  (Linear(in_features=2649, out_features=1000, bias=True)
            activation_func, #ReLU()
            nn.Linear(exp_encode_dim[0], exp_encode_dim[1]), #
            activation_func,
            nn.Linear(exp_encode_dim[1], exp_encode_dim[2]),
            activation_func)

        if TCGA_pretrain_weight_path_exp is not None:
            state_dict = torch.load(TCGA_pretrain_weight_path_exp, map_location=device)# Load the state_dict #TCGA AE Pretrained Weights
            encoder_state_dict = {key[len("encoder."):]: value for key, value in state_dict.items() if key.startswith('encoder')}# match the layer name to load weight
            self.model_exp.load_state_dict(encoder_state_dict) # Load only the encoder part
            model_keys = set(self.model_exp.state_dict().keys()) # Check if the keys match
            loaded_keys = set(encoder_state_dict.keys())
            if model_keys == loaded_keys:
                print("State_dict loaded successfully.")
            else:
                print("State_dict does not match the model's architecture.")
                print("Model keys: ", model_keys, "  Loaded keys: ", loaded_keys)

# Calculate final input dimension
        final_input_dim = exp_encode_dim[-1] + mut_encode_dim[-1]+ drug_encode_dim[-1]           
# Define the final prediction network 
        self.model_final_add = nn.Sequential(
            nn.Linear(final_input_dim, dense_layer_dim),
            activation_func,
            nn.Linear(dense_layer_dim, dense_layer_dim),
            activation_func,
            nn.Linear(dense_layer_dim, 1),
            activation_func_final)
        # Initialize both weights and biases with Kaiming uniform initialization
        for layer in self.model_final_add:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

        
    def forward(self, mut,drug,exp):
        mut_embed = self.model_mut(mut)
        drug_embed = self.model_drug(drug)
        exp_embed = self.model_exp(exp)
        # Concatenate embeddings from all subnetworks
        combined_mut_drug_exp_embed = torch.cat([mut_embed, drug_embed,exp_embed], dim=1)#dim=1: turn into 1D
        output = self.model_final_add(combined_mut_drug_exp_embed)
        return output






# Define the model architecture in PyTorch
class Mut_Exp_Model(nn.Module):
    def __init__(self, num_mut_features, num_exp_features, device, TCGA_pretrain_weight_path_mut=None, TCGA_pretrain_weight_path_exp=None):
        super(Mut_Exp_Model, self).__init__()

# Define subnetworks for mutations
        self.model_mut = nn.Sequential(
            nn.Linear(num_mut_features,mut_encode_dim[0]), # model_mut[0] : (Linear(in_features=2649, out_features=1000, bias=True)
            activation_func, #model_mut[1] : ReLU()
            nn.Linear(mut_encode_dim[0], mut_encode_dim[1]), # model_mut[2]
            activation_func,
            nn.Linear(mut_encode_dim[1], mut_encode_dim[2]),
            activation_func)

        if TCGA_pretrain_weight_path_mut is not None:
            # Load the state_dict #TCGA AE Pretrained Weights
            state_dict = torch.load(TCGA_pretrain_weight_path_mut, map_location=device)
            # match the layer name to load weight
            encoder_state_dict = {key[len("encoder."):]: value for key, value in state_dict.items() if key.startswith('encoder')}
            # Load only the encoder part
            self.model_mut.load_state_dict(encoder_state_dict)
            # Check if the keys match
            model_keys = set(self.model_mut.state_dict().keys())
            loaded_keys = set(encoder_state_dict.keys())
            if model_keys == loaded_keys:
                print("State_dict loaded successfully.")
            else:
                print("State_dict does not match the model's architecture.")
                print("Model keys: ", model_keys)
                print("Loaded keys: ", loaded_keys)

# Define subnetworks for Expressions
        self.model_exp = nn.Sequential(
            nn.Linear(num_exp_features,exp_encode_dim[0]), #  (Linear(in_features=2649, out_features=1000, bias=True)
            activation_func, #ReLU()
            nn.Linear(exp_encode_dim[0], exp_encode_dim[1]), #
            activation_func,
            nn.Linear(exp_encode_dim[1], exp_encode_dim[2]),
            activation_func)

        if TCGA_pretrain_weight_path_exp is not None:
            # Load the state_dict #TCGA AE Pretrained Weights
            state_dict = torch.load(TCGA_pretrain_weight_path_exp, map_location=device)
            # match the layer name to load weight
            encoder_state_dict = {key[len("encoder."):]: value for key, value in state_dict.items() if key.startswith('encoder')}
            # Load only the encoder part
            self.model_exp.load_state_dict(encoder_state_dict)
            # Check if the keys match
            model_keys = set(self.model_exp.state_dict().keys())
            loaded_keys = set(encoder_state_dict.keys())
            if model_keys == loaded_keys:
                print("State_dict loaded successfully.")
            else:
                print("State_dict does not match the model's architecture.")
                print("Model keys: ", model_keys)
                print("Loaded keys: ", loaded_keys)

# Calculate final input dimension
        final_input_dim = exp_encode_dim[-1] + mut_encode_dim[-1]        
# Define the final prediction network 
        self.model_final_add = nn.Sequential(
            nn.Linear(final_input_dim, dense_layer_dim),
            activation_func,
            nn.Linear(dense_layer_dim, dense_layer_dim),
            activation_func,
            nn.Linear(dense_layer_dim, num_drug),
            activation_func_final)
        # Initialize both weights and biases with Kaiming uniform initialization
        for layer in self.model_final_add:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

        
    def forward(self, mut,exp):
        mut_embed = self.model_mut(mut)
        exp_embed = self.model_exp(exp)
        # Concatenate embeddings from all subnetworks
        combined_mut_exp_embed = torch.cat([mut_embed,exp_embed], dim=1)#dim=1: turn into 1D
        output = self.model_final_add(combined_mut_exp_embed)
        return output








class OnlyDrugModel(nn.Module):
    def __init__(self, num_drug_features,num_ccl , device,  TCGA_pretrain_weight_path=None):
        super(OnlyDrugModel, self).__init__()
# Define subnetwork for drug 166features
        self.model_drug = nn.Sequential(
            nn.Linear(num_drug_features, drug_encode_dim[0]),
            activation_func,
            nn.Linear(drug_encode_dim[0], drug_encode_dim[1]),
            activation_func,
            nn.Linear(drug_encode_dim[1], drug_encode_dim[2]),
            activation_func)
        # Initialize both weights and biases with Kaiming uniform initialization
        for layer in  self.model_drug:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)
                    
# Calculate final input dimension
        final_input_dim = drug_encode_dim[-1]
# Define the final prediction network for regression
        self.model_final_drug = nn.Sequential(
            nn.Linear(final_input_dim, dense_layer_dim),
            activation_func,
            nn.Linear(dense_layer_dim, dense_layer_dim),
            activation_func,
            nn.Linear(dense_layer_dim, num_ccl), # output for 446 ccl
            activation_func_final)
        # Initialize both weights and biases with Kaiming uniform initialization
        for layer in self.model_final_drug:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)
    
    def forward(self, drug):
        drug_embed = self.model_drug(drug)
        output = self.model_final_drug(drug_embed)
        return output









class OnlyExpModel(nn.Module):
    def __init__(self, num_exp_features,num_drug, device,  TCGA_pretrain_weight_path_exp=None):
        super(OnlyExpModel, self).__init__()
        
# Define subnetworks for Expression
        self.model_exp = nn.Sequential(
            nn.Linear(num_exp_features,exp_encode_dim[0]), #  (Linear(in_features=2649, out_features=1000, bias=True)
            activation_func, #ReLU()
            nn.Linear(exp_encode_dim[0], exp_encode_dim[1]), #
            activation_func,
            nn.Linear(exp_encode_dim[1], exp_encode_dim[2]),
            activation_func)
        if TCGA_pretrain_weight_path_exp is not None:
            # Load the state_dict #TCGA AE Pretrained Weights
            state_dict = torch.load(TCGA_pretrain_weight_path_exp, map_location=device)
            # match the layer name to load weight
            encoder_state_dict = {key[len("encoder."):]: value for key, value in state_dict.items() if key.startswith('encoder')}
            # Load only the encoder part
            self.model_mut.load_state_dict(encoder_state_dict)
            # Check if the keys match
            model_keys = set(self.model_mut.state_dict().keys())
            loaded_keys = set(encoder_state_dict.keys())
            if model_keys == loaded_keys:
                print("State_dict loaded successfully.")
            else:
                print("State_dict does not match the model's architecture.")
                print("Model keys: ", model_keys)
                print("Loaded keys: ", loaded_keys)
                
# Calculate final input dimension
        final_input_dim = exp_encode_dim[-1]
# Define the final prediction network for regression
        self.model_final_exp = nn.Sequential(
            nn.Linear(final_input_dim, dense_layer_dim),
            activation_func,
            nn.Linear(dense_layer_dim, dense_layer_dim),
            activation_func,
            nn.Linear(dense_layer_dim, num_drug), # output for 1450 drugs
            activation_func_final)
        # Initialize both weights and biases with Kaiming uniform initialization
        for layer in self.model_final_exp:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)
    
    def forward(self, exp):
        exp_embed = self.model_exp(exp)
        output = self.model_final_exp(exp_embed)
        return output





class Exp_Drug_Model(nn.Module):
    def __init__(self, num_exp_features, num_drug_features, device,  TCGA_pretrain_weight_path=None):
        super(Exp_Drug_Model, self).__init__()
        
# Define subnetworks for Expression
        self.model_exp = nn.Sequential(
            nn.Linear(num_exp_features,exp_encode_dim[0]), #  (Linear(in_features=2649, out_features=1000, bias=True)
            activation_func, #ReLU()
            nn.Linear(exp_encode_dim[0], exp_encode_dim[1]), #
            activation_func,
            nn.Linear(exp_encode_dim[1], exp_encode_dim[2]),
            activation_func)

        if TCGA_pretrain_weight_path_exp is not None:
            # Load the state_dict #TCGA AE Pretrained Weights
            state_dict = torch.load(TCGA_pretrain_weight_path_exp, map_location=device)
            # match the layer name to load weight
            encoder_state_dict = {key[len("encoder."):]: value for key, value in state_dict.items() if key.startswith('encoder')}
            # Load only the encoder part
            self.model_exp.load_state_dict(encoder_state_dict)
            # Check if the keys match
            model_keys = set(self.model_exp.state_dict().keys())
            loaded_keys = set(encoder_state_dict.keys())
            if model_keys == loaded_keys:
                print("State_dict loaded successfully.")
            else:
                print("State_dict does not match the model's architecture.")
                print("Model keys: ", model_keys)
                print("Loaded keys: ", loaded_keys)
            
# Define subnetwork for drug 166features
        self.model_drug = nn.Sequential(
            nn.Linear(num_drug_features, drug_encode_dim[0]),
            activation_func,
            nn.Linear(drug_encode_dim[0], drug_encode_dim[1]),
            activation_func,
            nn.Linear(drug_encode_dim[1], drug_encode_dim[2]),
            activation_func)
        # Initialize both weights and biases with Kaiming uniform initialization
        for layer in  self.model_drug:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)
                    
# Calculate final input dimension
        final_input_dim = exp_encode_dim[-1] + drug_encode_dim[-1]
# Define the final prediction network 
        self.model_final_add = nn.Sequential(
            nn.Linear(final_input_dim, dense_layer_dim),
            activation_func,
            nn.Linear(dense_layer_dim, dense_layer_dim),
            activation_func,
            nn.Linear(dense_layer_dim, 1),
            activation_func_final)
        # Initialize both weights and biases with Kaiming uniform initialization
        for layer in self.model_final_add:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, exp,drug):
        exp_embed = self.model_exp(exp)
        drug_embed = self.model_drug(drug)
        # Concatenate embeddings from all subnetworks
        combined_exp_drug_embed = torch.cat([exp_embed, drug_embed], dim=1)#dim=1: turn into 1D
        output = self.model_final_add(combined_exp_drug_embed)
        return output