# Mut_Drug_Model、Omics_DrugESPF_Model

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import copy
import numpy as np


torch.manual_seed(42)
np.random.seed(42)

# ----------------PDAC--------------------
class Mut_Drug_Model(nn.Module):
    def __init__(self,mut_encode_dim,drug_encode_dim, activation_func,activation_func_final,dense_layer_dim, device, 
                 num_mut_features, num_drug_features,  TCGA_pretrain_weight_path=None):
        super(Mut_Drug_Model, self).__init__()
        
# Define subnetworks for mutations
        self.MLP4mut = nn.Sequential(
            nn.Linear(num_mut_features,mut_encode_dim[0]), # MLP4mut[0] : (Linear(in_features=2649, out_features=1000, bias=True)
            activation_func, #MLP4mut[1] : ReLU()
            nn.Linear(mut_encode_dim[0], mut_encode_dim[1]), # MLP4mut[2]
            activation_func,
            nn.Linear(mut_encode_dim[1], mut_encode_dim[2]),
            activation_func)

        if TCGA_pretrain_weight_path is not None:
            state_dict = torch.load(TCGA_pretrain_weight_path, map_location=device)# Load the state_dict #TCGA AE Pretrained Weights
            encoder_state_dict = {key[len("encoder."):]: value for key, value in state_dict.items() if key.startswith('encoder')}# match the layer name to load weight
            self.MLP4mut.load_state_dict(encoder_state_dict)# Load only the encoder part
            model_keys = set(self.MLP4mut.state_dict().keys())# Check if the keys match
            loaded_keys = set(encoder_state_dict.keys())
            if model_keys == loaded_keys:
                print("State_dict loaded successfully.")
            else:
                print("State_dict does not match the model's architecture.")
                print("Model keys: ", model_keys)
                print("Loaded keys: ", loaded_keys)
            
# Define subnetwork for drug 166features
        self.MLP4MACCS = nn.Sequential(
            nn.Linear(num_drug_features, drug_encode_dim[0]),
            activation_func,
            nn.Linear(drug_encode_dim[0], drug_encode_dim[1]),
            activation_func,
            nn.Linear(drug_encode_dim[1], drug_encode_dim[2]),
            activation_func)
        # Initialize both weights and biases with Kaiming uniform initialization
        for layer in self.MLP4MACCS:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
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
                init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
                '''
                mode: either `'fan_in'` (default) or `'fan_out'`. 
                Choosing `'fan_in'`preserves the magnitude of the variance of the weights in the forward pass. 
                Choosing `'fan_out'` preserves the magnitudes in the backwards pass.  
                '''
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, mut,drug):
        mut_embed = self.MLP4mut(mut)
        drug_embed = self.MLP4MACCS(drug)
        # Concatenate embeddings from all subnetworks
        combined_mut_drug_embed = torch.cat([mut_embed, drug_embed], dim=1)#dim=1: turn into 1D
        output = self.model_final_add(combined_mut_drug_embed)
        return output
    





# Modules------------------------------------------------------------------------------------------------------------------------------------------------------        
'''
class PositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout=.1, max_len=1024):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)
    positional_encoding = torch.zeros(max_len, d_model) # [max_len, d_model]
    position = torch.arange(0, max_len).float().unsqueeze(1) # [max_len, 1]
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         (-torch.log(torch.Tensor([10000])) / d_model)) # [max_len / 2]
    positional_encoding[:, 0::2] = torch.sin(position * div_term) # even
    positional_encoding[:, 1::2] = torch.cos(position * div_term) # odd
    # [max_len, d_model] -> [1, max_len, d_model] -> [max_len, 1, d_model]
    positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)
    # register pe to buffer and require no grads
    self.register_buffer('pe', positional_encoding)
  def forward(self, x):
    # x: [seq_len, batch, d_model]
    # we can add positional encoding to x directly, and ignore other dimension
    x = x + self.pe[:x.size(0), ...]
    return self.dropout(x)
'''
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
        self.word_embeddings = nn.Embedding(substructure_size, hidden_size)#(2586,128)# 50個onehot categorical id(0~2585)用128維來表示類別資訊
        self.position_embeddings = nn.Embedding(max_drug_len, hidden_size)#(50, 128)# 50個pos id(0~50)用128維vector來表示位置資訊
        self.LayerNorm = LayerNorm(hidden_size)#128
        self.dropout = nn.Dropout(hidden_dropout_prob)#0.1
    def forward(self, input_ids):
        seq_length = input_ids.size(1) #input_ids:(batchsize=64,50)# seq_length:50 # 50個onehot categorical id(0~2585)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device) #position_ids:torch.Size([50]) (0~50)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)#position_ids:torch.Size([64, 50])
        words_embeddings = self.word_embeddings(input_ids) #input_ids:(batchsize=64,50)# generate(50,128)類別特徵
        position_embeddings = self.position_embeddings(position_ids)# generate(50,128)位置特徵
        # words_embeddings: torch.Size([bsz, 50, 128])50個sub,其對應的representation 
        # position_embeddings: torch.Size([bsz, 50, 128])

        embeddings = words_embeddings + position_embeddings # embeddings:torch.Size([bsz, 50, 128])
        embeddings = self.LayerNorm(embeddings)#LayerNorm embeddings torch.Size([bsz, 50, 128])
        embeddings = self.dropout(embeddings)#dropout embeddings torch.Size([bsz, 50, 128])
        return embeddings # emb.shape:torch.Size([bsz, 50, 128])
    


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

    def transpose_for_scores(self, x): # x: torch.Size([bsz, 50, 128]) # diveide the whole 128 features into 8 heads, result 16 features per head
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # (8,16)
        # x.size()[:-1] torch.Size([bsz, 50]) # new_x_shape: torch.Size([bsz, 50, 8, 16])
        x = x.view(*new_x_shape) # changes the shape of x to the new_x_shape # x torch.Size([bsz, 50, 8, 16])
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask): 
        # hidden_states:emb.shape:torch.Size([bsz, 50, 128]); attention_mask: ex_e_mask:torch.Size([bsz, 1, 1, 50])
        mixed_query_layer = self.query(hidden_states) #hidden_states: torch.Size([bsz, 50, 128])
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states) # mixed_value_layer: torch.Size([bsz, 50, 128])

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer) #value_layer:torch.Size([bsz, 8, 50, 16])
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))# key_layer.transpose(-1, -2):torch.Size([bsz, 8, 16, 50])
        # attention_scores:torch.Size([bsz, 8, 50, 50])
        # Scaled Dot-Product: Prevent the dot products from growing too large, causing gradient Vanishing.
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # /16
        # attention_scores:torch.Size([bsz, 8, 50, 50])
        attention_scores = attention_scores + attention_mask #torch.Size([bsz, 1, 1, 50])[-0,-0,-0,-0,....,-10000,-10000,....]
        # attention_scores+ attention_mask:torch.Size([bsz, 8, 50, 50])
        
        # Normalize the attention scores to probabilities.
        attention_probs_0 = nn.Softmax(dim=-1)(attention_scores) # attSention_probs:torch.Size([bsz, 8, 50, 50])
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_drop = self.dropout(attention_probs_0)

        context_layer = torch.matmul(attention_probs_drop, value_layer) #context_layer:torch.Size([bsz, 8, 50, 16])
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() #context_layer:torch.Size([bsz, 50, 8, 16])
        # context_layer.size()[:-2] torch.Size([bsz, 50])
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) #new_context_layer_shape:torch.Size([bsz, 50, 128]) #(128,)
        context_layer = context_layer.view(*new_context_layer_shape) #context_layer:torch.Size([bsz, 50, 128])
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

class Output(nn.Module):# do linear, skip connection, LayerNorm, dropout after intermediate(Feed Forward block)
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


class Encoder(nn.Module):  # Transformer Encoder for drug feature # Drug_SelfAttention
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


class Drug_Cell_SelfAttention(nn.Module): # substructures 和omics的 self-attention
    def __init__(self, drug_dim, cell_dim, attn_dim):
        super(Drug_Cell_SelfAttention, self).__init__()



        
class CrossAttention(nn.Module): # substructures 和 pathways 的 cross-attention
    def __init__(self, drug_dim, cell_dim, attn_dim):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(drug_dim, attn_dim)  # Project drug features to query space
        self.key_proj = nn.Linear(cell_dim, attn_dim)    # Project cell features to key space
        self.value_proj = nn.Linear(cell_dim, attn_dim)  # Project cell features to value space
        self.scale = attn_dim ** -0.5  # Scaling factor for dot product attention

    def forward(self, drug_subunits, cell_subunits):
        # drug_subunits: (batch_size, num_drug_subunits, drug_dim)
        # cell_subunits: (batch_size, num_cell_subunits, cell_dim)

        # Project drug features to query, and cell features to key and value
        queries = self.query_proj(drug_subunits)        # Shape: (batch_size, num_drug_subunits, attn_dim)
        keys = self.key_proj(cell_subunits)             # Shape: (batch_size, num_cell_subunits, attn_dim)
        values = self.value_proj(cell_subunits)         # Shape: (batch_size, num_cell_subunits, attn_dim)

        # Compute attention scores (scaled dot product)
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale  # Shape: (batch_size, num_drug_subunits, num_cell_subunits)
        attn_weights = F.softmax(attn_scores, dim=-1)     # Normalize scores to get attention weights

        # Compute the final attended output
        attended_values = torch.matmul(attn_weights, values)  # Shape: (batch_size, num_drug_subunits, attn_dim)
        
        return attn_weights, attended_values
'''
# Example usage
batch_size = 2
num_drug_subunits, drug_dim = 5, 128
num_cell_subunits, cell_dim = 7, 256
attn_dim = 16

drug_subunits = torch.randn(batch_size, num_drug_subunits, drug_dim)
cell_subunits = torch.randn(batch_size, num_cell_subunits, cell_dim)

cross_attn = CrossAttention(drug_dim=drug_dim, cell_dim=cell_dim, attn_dim=attn_dim)
attn_weights, attended_values = cross_attn(drug_subunits, cell_subunits)

print("Attention Weights Shape:", attn_weights.shape)  # (batch_size, num_drug_subunits, num_cell_subunits)
print("Attended Values Shape:", attended_values.shape)  # (batch_size, num_drug_subunits, attn_dim)
'''
class Type_Encoding(nn.Module):
    def __init__(self, drug_cell_dim , num_types = 2, trans=False):
        self.embedding_dim = drug_cell_dim
        self.type_embedding = nn.Embedding(num_types, self.embedding_dim) # embedding_dim = drug_cell_dim
 
        



# End of Modules------------------------------------------------------------------------------------------------------------------------------------------------------




# Models------------------------------------------------------------------------------------------------------------------------------------------------------
class Omics_DrugESPF_Model(nn.Module):
    def __init__(self,omics_encode_dim_dict,drug_encode_dims, activation_func,activation_func_final,dense_layer_dim, device, ESPF, Drug_SelfAttention,
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
                self._init_weights(self.MLP4omics_dict[omic_type])

# Define subnetwork for drug ESPF features
        if ESPF is True:
            self.emb_f = Embeddings(hidden_size,max_drug_len,hidden_dropout_prob,substructure_size = 2586)#(128,50,0.1,2586)
            # if attention is not True 
            if Drug_SelfAttention is False: 
                self.dropout = nn.Dropout(attention_probs_dropout_prob)
            # self.output = SelfOutput(hidden_size, hidden_dropout_prob) # (128,0.1) # apply linear and skip conneaction and LayerNorm and dropout after attention
            # if attention is True  
            elif Drug_SelfAttention is False: 
                self.TransformerEncoder = Encoder(hidden_size, intermediate_size, num_attention_heads,attention_probs_dropout_prob, hidden_dropout_prob)#(128,512,8,0.1,0.1)
        
            self.MLP4ESPF = nn.Sequential(
                nn.Linear(max_drug_len * hidden_size, drug_encode_dims[0]),
                activation_func,
                nn.Dropout(hidden_dropout_prob),
                nn.Linear(drug_encode_dims[0], drug_encode_dims[1]),
                activation_func,
                nn.Dropout(hidden_dropout_prob),
                nn.Linear(drug_encode_dims[1], drug_encode_dims[2]),
                activation_func)
            # Initialize weights with Kaiming uniform initialization, bias with aero
            self._init_weights(self.MLP4ESPF)
        else: # MACCS166
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
            self._init_weights(self.MLP4MACCS)
  
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
        self._init_weights(self.model_final_add)

        self.print_flag = True
        self.attention_probs = None # store Attention score matrix
    
    def _init_weights(self, model):
        for layer in model:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)
    def forward(self, omics_tensor_dict,drug, device,ESPF,Drug_SelfAttention):

        omic_embeddings = []
        # Loop through each omic type and pass through its respective model
        for omic_type, omic_tensor in omics_tensor_dict.items():
            omic_embed = self.MLP4omics_dict[omic_type](omic_tensor.to(device=device))
            omic_embeddings.append(omic_embed)
        omic_embeddings = torch.cat(omic_embeddings, dim=1)  # change list to tensor, because omic_embeddings need to be tensor to torch.cat([omic_embeddings, drug_emb_masked], dim=1) 

        if ESPF is True:
            mask = drug[:, 1, :].to(device=device) # torch.Size([bsz, 50]),dytpe(long)
            drug_embed = drug[:, 0, :].to(device=device) # drug_embed :torch.Size([bsz, 50]),dytpe(long)
            drug_embed = self.emb_f(drug_embed) # (bsz, 50, 128) # Embeddings take int inputs, so no need to convert to float like nn.Linear layer

            if Drug_SelfAttention is False:
                if self.print_flag is True:
                    print("\n Drug_SelfAttention is not applied \n")
                    self.print_flag  = False
                # to apply mask to emb, treat mask like attention score matrix (weight), then do softmax and dropout, then multiply with emb
                mask_weight =mask.clone().float().unsqueeze(1).repeat(1, 50, 1)# (bsz, 50)->(bsz,50,50)
                mask_weight = (1.0 - mask_weight) * -10000.0
                mask_weight = nn.Softmax(dim=-1)(mask_weight)
                mask_weight = self.dropout(mask_weight)
                drug_emb_masked = torch.matmul(mask_weight, drug_embed) # emb_masked: torch.Size([bsz, 50, 128])
                # 沒做: class SelfOutput(nn.Module): # apply linear and skip conneaction and LayerNorm and dropout after 

            elif Drug_SelfAttention is True:
                if self.print_flag is True:
                    print("\n Drug_SelfAttention is applied \n")
                    self.print_flag  = False
                mask = mask.unsqueeze(1).unsqueeze(2) # mask.shape: torch.Size([bsz, 1, 1, 50])
                mask = (1.0 - mask) * -10000.0
                drug_emb_masked, attention_probs_0  = self.TransformerEncoder(drug_embed, mask)# hidden_states:drug_embed.shape:torch.Size([bsz, 50, 128]); mask: ex_e_mask:torch.Size([bsz, 1, 1, 50])
                # drug_emb_masked: torch.Size([bsz, 50, 128]) 
                # attention_probs_0 = nn.Softmax(dim=-1)(attention_scores) # attention_probs_0:torch.Size([bsz, 8, 50, 50])(without dropout)
                self.attention_probs = attention_probs_0
            elif Drug_SelfAttention is None:
                    print("\n Drug_SelfAttention is assign to None , please assign to False or True \n")

            drug_emb_masked = drug_emb_masked.reshape(-1,drug_emb_masked.shape[1]*drug_emb_masked.shape[2]) # flatten to (bsz, 50*128)
            drug_final_emb = self.MLP4ESPF(drug_emb_masked) # 6400->1600->400->100

        else: # MACCS166 
            if self.print_flag is True:
                print("\n MACCS166 is applied \n")
                self.print_flag  = False
            drug = drug.to(torch.float32) # long -> float, because the input of linear layer should be float,才能和float的weight相乘
            drug_final_emb = self.MLP4MACCS(drug.to(device=device))# 166->[110,55,22] # to device, because the weight is on device
            
        
        # Concatenate embeddings from all subnetworks
        combined_mut_drug_embed = torch.cat([omic_embeddings, drug_final_emb], dim=1)#dim=1: turn into 1D
        output = self.model_final_add(combined_mut_drug_embed)
        return output, self.attention_probs