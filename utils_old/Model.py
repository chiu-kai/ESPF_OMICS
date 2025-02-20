# Mut_Drug_Model、Omics_DrugESPF_Model

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import copy
import numpy as np
# from utils.config import *


torch.manual_seed(42)
np.random.seed(42)

# ----------------PDAC--------------------
class Mut_Drug_Model(nn.Module):
    def __init__(self,mut_encode_dim,drug_encode_dim, activation_func,activation_func_final,dense_layer_dim, device, 
                 num_mut_features, num_drug_features,  TCGA_pretrain_weight_path=None):
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
            state_dict = torch.load(TCGA_pretrain_weight_path, map_location=device)# Load the state_dict #TCGA AE Pretrained Weights
            encoder_state_dict = {key[len("encoder."):]: value for key, value in state_dict.items() if key.startswith('encoder')}# match the layer name to load weight
            self.model_mut.load_state_dict(encoder_state_dict)# Load only the encoder part
            model_keys = set(self.model_mut.state_dict().keys())# Check if the keys match
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

class Embeddings(nn.Module): # word embedding + positional encoding
    """Construct the embeddings from protein/target, position embeddings."""
    def __init__(self, hidden_size,max_drug_len,hidden_dropout_prob,substructure_size):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(substructure_size, hidden_size)#(2586,128)
        self.position_embeddings = nn.Embedding(max_drug_len, hidden_size)#(50, 128)
        self.LayerNorm = LayerNorm(hidden_size)#128
        self.dropout = nn.Dropout(hidden_dropout_prob)#0.1
    def forward(self, input_ids):
        seq_length = input_ids.size(1) #input_ids:(batchsize=64,50)# seq_length:50
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device) #position_ids:torch.Size([50])
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)#position_ids:torch.Size([64, 50])
        words_embeddings = self.word_embeddings(input_ids) #input_ids:(batchsize=64,50)
        position_embeddings = self.position_embeddings(position_ids)
        # words_embeddings: torch.Size([64, 50, 128])50個sub,其對應的representation 
        # position_embeddings: torch.Size([64, 50, 128])

        embeddings = words_embeddings + position_embeddings # embeddings:torch.Size([64, 50, 128])
        embeddings = self.LayerNorm(embeddings)#LayerNorm embeddings torch.Size([64, 50, 128])
        embeddings = self.dropout(embeddings)#dropout embeddings torch.Size([64, 50, 128])
        return embeddings # emb.shape:torch.Size([64, 50, 128])
    


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()# (128,8,0.1)
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads #8 頭數
        self.attention_head_size = int(hidden_size / num_attention_heads)#128/8=16 頭的維度
        self.all_head_size = self.num_attention_heads * self.attention_head_size#8*16=128 頭的維度總和等於feature數

        self.query = nn.Linear(hidden_size, self.all_head_size)#(128,128)
        self.key = nn.Linear(hidden_size, self.all_head_size)#(128,128)
        self.value = nn.Linear(hidden_size, self.all_head_size)#(128,128)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)#0.1

    def transpose_for_scores(self, x): # x: torch.Size([64, 50, 128])
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # (8,16)
        # x.size()[:-1] torch.Size([64, 50]) # new_x_shape: torch.Size([64, 50, 8, 16])
        x = x.view(*new_x_shape) # changes the shape of x to the new_x_shape # x torch.Size([64, 50, 8, 16])
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask): 
        # hidden_states:emb.shape:torch.Size([64, 50, 128]); attention_mask: ex_e_mask:torch.Size([64, 1, 1, 50])
        mixed_query_layer = self.query(hidden_states) #hidden_states: torch.Size([64, 50, 128])
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states) # mixed_value_layer: torch.Size([64, 50, 128])

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer) #value_layer:torch.Size([64, 8, 50, 16])
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))# key_layer.transpose(-1, -2):torch.Size([64, 8, 16, 50])
        # attention_scores:torch.Size([64, 8, 50, 50])
        # Scaled Dot-Product: Prevent the dot products from growing too large, causing gradient Vanishing.
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # /16
        # attention_scores:torch.Size([64, 8, 50, 50])
        attention_scores = attention_scores + attention_mask #torch.Size([64, 1, 1, 50])[-0,-0,-0,-0,....,-10000,-10000,....]
        # attention_scores+ attention_mask:torch.Size([64, 8, 50, 50])
        
        # Normalize the attention scores to probabilities.
        attention_probs_0 = nn.Softmax(dim=-1)(attention_scores) # attSention_probs:torch.Size([64, 8, 50, 50])
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs_0)

        context_layer = torch.matmul(attention_probs, value_layer) #context_layer:torch.Size([64, 8, 50, 16])
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() #context_layer:torch.Size([64, 50, 8, 16])
        # context_layer.size()[:-2] torch.Size([64, 50])
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) #new_context_layer_shape:torch.Size([64, 50, 128]) #(128,)
        context_layer = context_layer.view(*new_context_layer_shape) #context_layer:torch.Size([64, 50, 128])
        return context_layer, attention_probs_0
    
class SelfOutput(nn.Module): # apply linear and skip conneaction and LayerNorm and dropout after self-attention
    def __init__(self, hidden_size, dropout_prob):
        super(SelfOutput, self).__init__()# (128,0.1)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states    

class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.selfAttention = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)# apply linear and skip conneaction and LayerNorm and dropout after self-attention

    def forward(self, input_tensor, attention_mask):
        # input_tensor:emb.shape:torch.Size([64, 50, 128]); attention_mask: ex_e_mask:torch.Size([64, 1, 1, 50])
        self_output, attention_probs_0 = self.selfAttention(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs_0    


class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()# (128,512)
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states

class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()# (512,128,0.1)
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states # transformer 最後的輸出

class Encoder(nn.Module):  # Transformer Encoder for drug feature
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):#(128,512,8,0.1,0.1)
        super(Encoder, self).__init__() # (128,512,8,0.1,0.1)
        self.attention = Attention(hidden_size, num_attention_heads,
                                   attention_probs_dropout_prob, hidden_dropout_prob)# (128,8,0.1,0.1)
        self.intermediate = Intermediate(hidden_size, intermediate_size)# (128,512)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)# (512,128,0.1)

    def forward(self, hidden_states, attention_mask):
        # hidden_states:emb.shape:torch.Size([64, 50, 128]); attention_mask: ex_e_mask:torch.Size([64, 1, 1, 50])
        attention_output,attention_probs_0 = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output , attention_probs_0    # transformer 最後的輸出

class Encoder_MultipleLayers(nn.Module): # 用Encoder更新representation n_layer次 # DeepTTA paper寫6次
    def __init__(self, n_layer, hidden_size, intermediate_size,
                 num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob): #(8,128,512,8,0.1,0.1)
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads,
                        attention_probs_dropout_prob, hidden_dropout_prob) # (128,512,8,0.1,0.1)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask) 
            #if output_all_encoded_layers:
            #    all_encoder_layers.append(hidden_states)
        #if not output_all_encoded_layers:
        #    all_encoder_layers.append(hidden_states)
        return hidden_states  # transformer 最後的輸出


# End of Modules------------------------------------------------------------------------------------------------------------------------------------------------------




# Models------------------------------------------------------------------------------------------------------------------------------------------------------

class Omics_DrugESPF_Model(nn.Module):
    def __init__(self,omics_encode_dim_dict,drug_encode_dims, activation_func,activation_func_final,dense_layer_dim, device, 
                 hidden_size, intermediate_size, num_attention_heads , attention_probs_dropout_prob, hidden_dropout_prob, omics_numfeaetures_dict, max_drug_len, TCGA_pretrain_weight_path_dict=None):
        super(Omics_DrugESPF_Model, self).__init__()

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
                    init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                    if layer.bias is not None:
                        init.zeros_(layer.bias)

# Create subnetworks for each omic type dynamically
        self.model_omics_dict = nn.ModuleDict()
        for omic_type in omics_numfeaetures_dict.keys():
            self.model_omics_dict[omic_type] = nn.Sequential(
                nn.Linear(omics_numfeaetures_dict[omic_type], omics_encode_dim_dict[omic_type][0]),
                activation_func,
                nn.Linear(omics_encode_dim_dict[omic_type][0], omics_encode_dim_dict[omic_type][1]),
                activation_func_final,
                nn.Linear(omics_encode_dim_dict[omic_type][1], omics_encode_dim_dict[omic_type][2])
            )
            # Initialize with TCGA pretrain weight
            if TCGA_pretrain_weight_path_dict is not None:
                load_TCGA_pretrain_weight(self.model_omics_dict[omic_type], TCGA_pretrain_weight_path_dict[omic_type], device)
            else: # Initialize weights with Kaiming uniform initialization, bias with aero
                _init_weights(self.model_omics_dict[omic_type])

# Define subnetwork for drug ESPF features
        self.emb_f = Embeddings(hidden_size,max_drug_len,hidden_dropout_prob,substructure_size = 2586)#(128,50,0.1,2586)
        # if attention is not True  
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob) # (128,0.1) # apply linear and skip conneaction and LayerNorm and dropout after attention
        # if attention is True  
        self.TransformerEncoder = Encoder(hidden_size, intermediate_size, num_attention_heads,attention_probs_dropout_prob, hidden_dropout_prob)#(128,512,8,0.1,0.1)
        
        
        self.model_drug = nn.Sequential(
            nn.Linear(max_drug_len * hidden_size, drug_encode_dims[0]),
            activation_func,
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(drug_encode_dims[0], drug_encode_dims[1]),
            activation_func,
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(drug_encode_dims[1], drug_encode_dims[2]),
            activation_func)
        # Initialize weights with Kaiming uniform initialization, bias with aero
        _init_weights(self.model_drug)

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

        self.print_flag = True
        self.attention_probs = None
    def forward(self, omics_tensor_dict,drug, device,Transformer=None):
        omic_embeddings = []
        # Loop through each omic type and pass through its respective model
        for omic_type, omic_tensor in omics_tensor_dict.items():
            omic_embed = self.model_omics_dict[omic_type](omic_tensor.to(device=device))
            omic_embeddings.append(omic_embed)
        omic_embeddings = torch.cat(omic_embeddings, dim=1)  # change list to tensor, because omic_embeddings need to be tensor to torch.cat([omic_embeddings, drug_emb_masked], dim=1) 

        mask = drug[:, 1, :].to(device=device) # torch.Size([bsz, 50]),dytpe(long)
        drug_embed = drug[:, 0, :].to(device=device) # drug_embed :torch.Size([bsz, 50]),dytpe(long)
        drug_embed = self.emb_f(drug_embed) # (bsz, 50, 128)

        if Transformer is False:
            if self.print_flag is True:
                print("\n Transformer is not applied \n")
                self.print_flag  = False
            # to apply mask to emb, treat mask like attention score matrix (weight), then do softmax and dropout, then multiply with emb
            mask_weight = torch.tensor(mask, dtype=torch.float32).unsqueeze(1).repeat(1, 50, 1)# (bsz, 50)->(bsz,50,50)
            mask_weight = (1.0 - mask_weight) * -10000.0
            mask_weight = nn.Softmax(dim=-1)(mask_weight)
            mask_weight = self.dropout(mask_weight)
            drug_emb_masked = torch.matmul(mask_weight, drug_embed) # emb_masked: torch.Size([bsz, 50, 128])

        elif Transformer is True:
            if self.print_flag is True:
                print("\n Transformer is applied \n")
                self.print_flag  = False
            mask = mask.unsqueeze(1).unsqueeze(2) # mask.shape: torch.Size([bsz, 1, 1, 50])
            mask = (1.0 - mask) * -10000.0
            drug_emb_masked, attention_probs  = self.TransformerEncoder(drug_embed, mask)# hidden_states:drug_embed.shape:torch.Size([64, 50, 128]); mask: ex_e_mask:torch.Size([64, 1, 1, 50])
            # drug_emb_masked: torch.Size([bsz, 50, 128]) 
            # attention_probs_0 = nn.Softmax(dim=-1)(attention_scores) # attention_probs_0:torch.Size([64, 8, 50, 50])
            self.attention_probs = attention_probs
        elif Transformer is None:
                print("\n Transformer is assign to None , please assign to False or True \n")

        drug_emb_masked = drug_emb_masked.reshape(-1,drug_emb_masked.shape[1]*drug_emb_masked.shape[2]) # flatten to (bsz, 50*128)
        drug_emb_masked = self.model_drug(drug_emb_masked) # 6400->1600->400->100

        
        # Concatenate embeddings from all subnetworks
        combined_mut_drug_embed = torch.cat([omic_embeddings, drug_emb_masked], dim=1)#dim=1: turn into 1D
        output = self.model_final_add(combined_mut_drug_embed)
        return output, self.attention_probs
    






    