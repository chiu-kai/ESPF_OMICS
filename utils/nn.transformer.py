# model torch.nn.TransformerEncoder
# how to write low-level optimizations instead of explicit Python-level control by myself.
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F



class Omics_DCSA_Model(nn.Module):
    def __init__(self,omics_encode_dim_dict,drug_encode_dims, activation_func,activation_func_final,dense_layer_dim, device, ESPF, Drug_SelfAttention, pos_emb_type,
                 hidden_size, intermediate_size, num_attention_heads , attention_probs_dropout_prob, hidden_dropout_prob, omics_numfeaetures_dict, max_drug_len, TCGA_pretrain_weight_path_dict=None):
        super(Omics_DCSA_Model, self).__init__()
        self.num_attention_heads = num_attention_heads

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

            #apply a linear tranformation to omics embedding to match the hidden size of the drug
            self.match_drug_dim = nn.Linear(omics_encode_dim_dict[omic_type][2], hidden_size)
            self._init_weights(self.match_drug_dim)
        
#ESPF            
        self.emb_f = Embeddings(hidden_size,max_drug_len,hidden_dropout_prob, pos_emb_type,substructure_size = 2586)#(128,50,0.1,2586)
        self._init_weights(self.emb_f)

        if Drug_SelfAttention is False: 
            self.dropout = nn.Dropout(attention_probs_dropout_prob)
        # self.output = SelfOutput(hidden_size, hidden_dropout_prob) # (128,0.1) # apply linear and skip conneaction and LayerNorm and dropout after attention

# if attention is True  
        elif Drug_SelfAttention is True: 
            self.TransformerEncoder = Encoder(hidden_size, intermediate_size, num_attention_heads,attention_probs_dropout_prob, hidden_dropout_prob)#(128,512,8,0.1,0.1)
            self._init_weights(self.TransformerEncoder)

# Drug_Cell_SelfAttention
        self.Drug_Cell_SelfAttention = Encoder(hidden_size+num_attention_heads, intermediate_size, num_attention_heads,attention_probs_dropout_prob, hidden_dropout_prob)#(128+8,512,8,0.1,0.1)
        self._init_weights(self.Drug_Cell_SelfAttention)

# Define the final prediction network 
        dense_layer_dim=7064
        self.model_final_add = nn.Sequential(
            nn.Linear(7064, 700),
            activation_func,
            nn.Dropout(p=0),
            nn.Linear(700, 70),
            activation_func,
            nn.Dropout(p=0),
            nn.Linear(70, 1),
            activation_func_final)
        # Initialize weights with Kaiming uniform initialization, bias with aero
        self._init_weights(self.model_final_add)

        self.print_flag = True
        self.attention_probs = None # store Attention score matrix
    
    def _init_weights(self, model):
        if isinstance(model, nn.Linear):  # 直接初始化 nn.Linear 層
            init.kaiming_uniform_(model.weight, a=0, mode='fan_in', nonlinearity='relu')
            if model.bias is not None:
                init.zeros_(model.bias)
        elif isinstance(model, nn.LayerNorm):
            init.ones_(model.weight)
            init.zeros_(model.bias)
        elif isinstance(model, nn.ModuleList) or isinstance(model, nn.Sequential):  # 遍歷子層
            for layer in model:
                self._init_weights(layer)

    def forward(self, omics_tensor_dict,drug, device, **kwargs):
        Drug_SelfAttention = kwargs['Drug_SelfAttention']
        omic_embeddings_ls = []
        # Loop through each omic type and pass through its respective model
        for omic_type, omic_tensor in omics_tensor_dict.items():
            omic_embed = self.MLP4omics_dict[omic_type](omic_tensor.to(device=device)) #(bsz, 50)
            #apply a linear tranformation to omics embedding to match the hidden size of the drug
            omic_embed = self.match_drug_dim(omic_embed) #(bsz, 128)
            omic_embeddings_ls.append(omic_embed)
        # omic_embeddings = torch.cat(omic_embeddings_ls, dim=1)  # change list to tensor, because omic_embeddings need to be tensor to torch.cat([omic_embeddings, drug_emb_masked], dim=1) 

    #ESPF encoding        
        mask = drug[:, 1, :].to(device=device) # torch.Size([bsz, 50]),dytpe(long)
        drug_embed = drug[:, 0, :].to(device=device) # drug_embed :torch.Size([bsz, 50]),dytpe(long)
        drug_embed = self.emb_f(drug_embed) # (bsz, 50, 128) #word embedding、position encoding、LayerNorm、dropout
        # Embeddings take int inputs, so no need to convert to float like nn.Linear layer
        
# mask for Drug Cell SelfAttention
        omics_items = torch.ones(mask.size(0), len(omic_embeddings_ls), dtype=mask.dtype, device=mask.device)  # Shape: [bsz, len(omic_embeddings_ls)]
        DrugCell_mask = torch.cat([mask, omics_items], dim=1)  # Shape: [bsz, 50 + len(omic_embeddings_ls)]

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
            # 沒做: positional encoding
            AttenScorMat_DrugSelf = None
            
        elif Drug_SelfAttention is True:
            if self.print_flag is True:
                print("\n Drug_SelfAttention is applied \n")
                self.print_flag  = False
            mask = mask.unsqueeze(1).unsqueeze(2) # mask.shape: torch.Size([bsz, 1, 1, 50])
            mask = (1.0 - mask) * -10000.0
            drug_emb_masked, AttenScorMat_DrugSelf  = self.TransformerEncoder(drug_embed, mask)# hidden_states:drug_embed.shape:torch.Size([bsz, 50, 128]); mask: ex_e_mask:torch.Size([bsz, 1, 1, 50])
            # drug_emb_masked: torch.Size([bsz, 50, 128]) 
            # attention_probs_0 = nn.Softmax(dim=-1)(attention_scores) # attention_probs_0:torch.Size([bsz, 8, 50, 50])(without dropout)
        elif Drug_SelfAttention is None:
                print("\n Drug_SelfAttention is assign to None , please assign to False or True \n")


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, attention_probs_dropout_prob, hidden_dropout_prob):
        super(CustomTransformerEncoderLayer, self).__init__()
        # MultiheadAttention for self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,              # e.g., 128
            num_heads=num_attention_heads,      # e.g., 8
            dropout=attention_probs_dropout_prob,  # e.g., 0.1
            batch_first=True                    # Matches your [bsz, seq_len, hidden_size] input
        )
        # Feed-forward network (FFN)
        self.intermediate = nn.Linear(hidden_size, intermediate_size)  # e.g., 128 -> 512
        self.output = nn.Linear(intermediate_size, hidden_size)       # e.g., 512 -> 128
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)                # e.g., 0.1

        # Optional: Store attention weights for later access
        self.attn_weights = None
        
    def forward(self, src, src_key_padding_mask=None):
        # src: [bsz, seq_len, hidden_size], e.g., [bsz, 50, 128]
        # src_key_padding_mask: [bsz, seq_len], e.g., [bsz, 50], True for padded positions

        # Self-attention with attention weights
        attn_output, attn_weights = self.self_attention(
            query=src,
            key=src,
            value=src,
            key_padding_mask=src_key_padding_mask,
            need_weights=True)  # Ensure attention weights are returned
        
        # attn_output: [bsz, seq_len, hidden_size], e.g., [bsz, 50, 128]
        # attn_weights: [bsz, num_heads, seq_len, seq_len], e.g., [bsz, 8, 50, 50]
        # Store attention weights (optional)
        self.attn_weights = attn_weights.detach()
        # Residual connection and normalization
        src = self.norm1(src + self.dropout(attn_output))
        # Feed-forward network
        ffn_output = self.output(F.relu(self.intermediate(src)))
        # Residual connection and normalization
        src = self.norm2(src + self.dropout(ffn_output))

        return src







class Omics_DCSA_Model(nn.Module):
    def __init__(self,omics_encode_dim_dict,drug_encode_dims, activation_func,activation_func_final,dense_layer_dim, device, ESPF, Drug_SelfAttention, pos_emb_type,
                 hidden_size, intermediate_size, num_attention_heads , attention_probs_dropout_prob, hidden_dropout_prob, omics_numfeaetures_dict, max_drug_len, TCGA_pretrain_weight_path_dict=None):
        super(Omics_DCSA_Model, self).__init__()
        self.num_attention_heads = num_attention_heads
        
        # Instantiate the custom layer
        self.encoder_layer = CustomTransformerEncoderLayer( hidden_size, num_attention_heads, intermediate_size, attention_probs_dropout_prob, hidden_dropout_prob
        )
        # Stack the layer into a TransformerEncoder (e.g., 1 layer, but you can increase num_layers)
        self.TransformerEncoder = nn.TransformerEncoder( encoder_layer=self.encoder_layer, num_layers=3  # Number of stacked layers;
        )
        self.Drug_Cell_SelfAttention = nn.TransformerEncoder( 
            encoder_layer=CustomTransformerEncoderLayer( hidden_size+num_attention_heads, num_attention_heads, intermediate_size, attention_probs_dropout_prob, hidden_dropout_prob
        ),
            num_layers=3  # Number of stacked layers;
        )

    def forward(self, omics_tensor_dict,drug, device, **kwargs):
        Drug_SelfAttention = kwargs['Drug_SelfAttention']
        omic_embeddings_ls = []
        # Loop through each omic type and pass through its respective model
        for omic_type, omic_tensor in omics_tensor_dict.items():
            omic_embed = self.MLP4omics_dict[omic_type](omic_tensor.to(device=device)) #(bsz, 50)
            #apply a linear tranformation to omics embedding to match the hidden size of the drug
            omic_embed = self.match_drug_dim(omic_embed) #(bsz, 128)
            omic_embeddings_ls.append(omic_embed)
        # omic_embeddings = torch.cat(omic_embeddings_ls, dim=1)  # change list to tensor, because omic_embeddings need to be tensor to torch.cat([omic_embeddings, drug_emb_masked], dim=1) 

    #ESPF encoding        
        mask = drug[:, 1, :].to(device=device) # torch.Size([bsz, 50]),dytpe(long)
        drug_embed = drug[:, 0, :].to(device=device) # drug_embed :torch.Size([bsz, 50]),dytpe(long)
        drug_embed = self.emb_f(drug_embed) # (bsz, 50, 128) #word embedding、position encoding、LayerNorm、dropout
        # Embeddings take int inputs, so no need to convert to float like nn.Linear layer
        
# mask for Drug Cell SelfAttention
        omics_items = torch.ones(mask.size(0), len(omic_embeddings_ls), dtype=mask.dtype, device=mask.device)  # Shape: [bsz, len(omic_embeddings_ls)]
        DrugCell_mask = torch.cat([mask, omics_items], dim=1)  # Shape: [bsz, 50 + len(omic_embeddings_ls)]
        # DrugCell_mask : 1是有0是沒有

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
            # 沒做: positional encoding
            AttenScorMat_DrugSelf = None
            
        elif Drug_SelfAttention is True:
            if self.print_flag is True:
                print("\n Drug_SelfAttention is applied \n")
                self.print_flag  = False
            mask = (1 - mask).bool()   # 1是pad,0不是pad # nn.TransformerEncoder的mask是相反的
            drug_emb_masked, AttenScorMat_DrugSelf = self.TransformerEncoder(drug_embed, src_key_padding_mask = mask)
            print(drug_emb_masked.shape)
            # drug_emb_masked: torch.Size([bsz, 50, 128]) 
            # attention_probs_0 = nn.Softmax(dim=-1)(attention_scores) # attention_probs_0:torch.Size([bsz, 8, 50, 50])(without dropout)
        elif Drug_SelfAttention is None:
                print("\n Drug_SelfAttention is assign to None , please assign to False or True \n")

# Drug_Cell_SelfAttention
        # omic_embeddings_ls:[(bsz,128),(bsz,128)] 
        # # drug_emb_masked:[bsz,50,128] #已經做完word embedding和position encoding
        omic_embeddings = torch.stack(omic_embeddings_ls, dim=1) #shape:[bsz,c,128] #Stack omic_embeddings_ls along the second dimension, c: number of omic types

        append_embeddings = torch.cat([drug_emb_masked, omic_embeddings], dim=1) #shape:[bsz,50+c,128] #Concatenate along the second dimension
# Type encoding (to distinguish between drug and omics)
        drug_type_encoding = torch.ones_like(drug_emb_masked[..., :1])  # Shape: [bsz, 50, 1]
        omics_type_encoding = torch.zeros_like(omic_embeddings[..., :1])  # Shape: [bsz, i, 1]
        # Concatenate type encoding with the respective data
        drug_emb_masked = torch.cat([drug_emb_masked, drug_type_encoding], dim=-1)  # Shape: [bsz, 50, 129]
        omic_embeddings = torch.cat([omic_embeddings, omics_type_encoding], dim=-1)  # Shape: [bsz, c, 129]

        # Final concatenated tensor (drug sequence and omics data with type encoding)
        append_embeddings = torch.cat([drug_emb_masked, omic_embeddings], dim=1)  # Shape: [bsz, 50+c, 129]

        padding_dim = self.num_attention_heads - 1  # Extra dimensions to add # padding_dim=7
        pad = torch.zeros(append_embeddings.size(0), append_embeddings.size(1), padding_dim, device=append_embeddings.device)
        append_embeddings = torch.cat([append_embeddings, pad], dim=-1)  # New shape: [bsz, 50+i, new_hidden_size=136]
        
        DrugCell_mask = (1 - DrugCell_mask).bool()   # 1是pad,0不是pad
        append_embeddings, AttenScorMat_DrugCellSelf  = self.Drug_Cell_SelfAttention(append_embeddings, DrugCell_mask)
        # append_embeddings: torch.Size([bsz, 50+c, 136]) # AttenScorMat_DrugCellSelf:torch.Size([bsz, 8, 50+c, 50+c])(without dropout)

        #skip connect the omics embeddings # not as necessary as skip connect the drug embeddings 
        append_embeddings = torch.cat([ torch.cat(omic_embeddings_ls, dim=1), append_embeddings.reshape(append_embeddings.size(0), -1)], dim=1) # dim=1: turn into 1D 
        #omic_embeddings_ls(bsz, 128) , append_embeddings(bsz, 50+c, 136)
        # drug 有50*128，omices有i*128，可能會差太多，看drug要不要先降維根omics一樣i*128 # 先不要
    
    # Final MLP
        output = self.model_final_add(append_embeddings)

        return output, AttenScorMat_DrugSelf, AttenScorMat_DrugCellSelf
