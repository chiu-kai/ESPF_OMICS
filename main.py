# main
import argparse
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.init as init
from sklearn.model_selection import KFold
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import random

from utils.ESPF_drug2emb import drug2emb_encoder
from utils.Model import Omics_DrugESPF_Model
from utils_main.split_data_id import split_id
from utils_main.create_dataloader import OmicsDrugDataset
from utils_main.train import train, evaluation
from utils.correlation import correlation_func
from utils.plot import loss_curve, correlation_density,Density_Plot_of_AUC_Values
from utils.tools import get_data_value_range,set_seed,get_vram_usage



import os

import argparse
import importlib.util

# 設定命令列引數
parser = argparse.ArgumentParser(description="import config to main")# python3 ./main.py --config utils/config.py
parser.add_argument("--config", required=True, help="Path to the config.py file")
args = parser.parse_args()

# 動態載入 config.py

spec = importlib.util.spec_from_file_location("config", args.config)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
# 將 config 模組中的變數導入當前命名空間
for key, value in vars(config).items():
    if not key.startswith("_"):  # 過濾內部變數，例如 __builtins__
        globals()[key] = value



print(os.getcwd())

test_dataset = True # True:small dataset, False: full dataset
test = True

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Training on device {device}.")

#--------------------------------------------------------------------------------------------------------------------------
if test is True:
    batch_size = 3
    num_epoch = 2
    print("batch_size",batch_size,"num_epoch:",num_epoch)
# print("include_omics",include_omics)
# print("dense_layer_dim",dense_layer_dim)


#--------------------------------------------------------------------------------------------------------------------------

set_seed(seed)
for omic_type in include_omics:
    # Read the file
    omics_data_dict[omic_type] = pd.read_csv(omics_files[omic_type], sep='\t', index_col=0)

    if test_dataset is True:
        # Specify the index as needed
        omics_data_dict[omic_type] = omics_data_dict[omic_type][:76]  # Adjust the row selection as needed
    omics_data_tensor_dict[omic_type]  = torch.tensor(omics_data_dict[omic_type].values, dtype=torch.float32)
    omics_numfeatures_dict[omic_type] = omics_data_tensor_dict[omic_type].shape[1]
    print(f"{omic_type} tensor shape:", omics_data_tensor_dict[omic_type].shape)
    print(f"{omic_type} num_features",omics_numfeatures_dict[omic_type])

#--------------------------------------------------------------------------------------------------------------------------
#load data
# data_mut, gene_names_mut,ccl_names_mut  = load_ccl("/root/data/CCLE/CCLE_match_TCGAgene_PRISMandEXPsample_binary_mutation_476_6009.txt")
drug_df= pd.read_csv("../data/no_Imputation_PRISM_Repurposing_Secondary_Screen_data/MACCS(Secondary_Screen_treatment_info)_union_NOrepeat.csv", sep=',', index_col=0)
AUC_df = pd.read_csv("../data/no_Imputation_PRISM_Repurposing_Secondary_Screen_data/Drug_sensitivity_AUC_(PRISM_Repurposing_Secondary_Screen)_subsetted_NOrepeat.csv", sep=',', index_col=0)
# data_AUC_matrix, drug_names_AUC, ccl_names_AUC = load_AUC_matrix(splitType,"/root/Winnie/no_Imputation_PRISM_Repurposing_Secondary_Screen_data/Drug_sensitivity_AUC_(PRISM_Repurposing_Secondary_Screen)_subsetted.csv") # splitType = "byCCL" or "byDrug" 決定AUCmatrix要不要轉置
print("drug_df",(np.shape(drug_df)))
print("AUC_df",np.shape(AUC_df))

# matched AUCfile and omics_data samples
matched_samples = sorted(set(AUC_df.T.columns) & set(list(omics_data_dict.values())[0].T.columns))
print(len(matched_samples))
AUC_df= (AUC_df.T[matched_samples]).T
print("AUC_df",AUC_df.shape)

if test_dataset is True:
    drug_df=drug_df[:42]
    AUC_df=AUC_df.iloc[:76,:42]
print("drug_df",drug_df.shape)
print("AUC_df",AUC_df.shape)

#--------------------------------------------------------------------------------------------------------------------------
# 檢查有無重複的SMILES
if ESPF is True:
    drug_smiles =drug_df["smiles"] # 
    print("drug_smiles.shape",drug_smiles.shape)
    drug_names =drug_df.index
    print("(drug_smiles.unique()).shape",(drug_smiles.unique()).shape)
    # 挑出重複的SMILES
    duplicate =  drug_smiles[drug_smiles.duplicated(keep=False)]
    print("duplicate drug",duplicate)

    #--------------------------------------------------------------------------------------------------------------------------
    #ESPF
    vocab_path = "./dataset/ESPF/drug_codes_chembl_freq_1500.txt" # token
    sub_csv = pd.read_csv("./dataset/ESPF/subword_units_map_chembl_freq_1500.csv")# token with frequency

    # drug_encode = pd.Series(drug_smiles.unique()).apply(drug2emb_encoder, args=(vocab_path, sub_csv, max_drug_len))# 將drug_smiles 使用_drug2emb_encoder function編碼成subword vector
    drug_encode = pd.Series(drug_smiles).apply(drug2emb_encoder, args=(vocab_path, sub_csv, max_drug_len))
    # uniq_smile_dict = dict(zip(drug_smiles.unique(),drug_encode))# zip drug_smiles和其subword vector編碼 成字典

    # print(type(smile_encode))
    # print(smile_encode.shape)
    # print(type(smile_encode.index))
    print((drug_encode.index.values).shape)

else:
    drug_encode = drug_df["MACCS166bits"]
#--------------------------------------------------------------------------------------------------------------------------
num_ccl = list(omics_data_dict.values())[0].shape[0]
num_drug = drug_encode.shape[0]
print("num_ccl,num_drug: ",num_ccl,num_drug)

#--------------------------------------------------------------------------------------------------------------------------
# Convert your data to tensors if they're in numpy
drug_features_tensor = torch.tensor(np.array(drug_encode.values.tolist()), dtype=torch.long)
response_matrix_tensor = torch.tensor(AUC_df.values, dtype=torch.float32)

#--------------------------------------------------------------------------------------------------------------------------
# randomly split
# 90% for training(10% for validation) and 10% for testing
id_unrepeat_train, id_unrepeat_val, id_unrepeat_test, _ , id_train, id_val, id_test= split_id(num_ccl=num_ccl,num_drug=num_drug,splitType=splitType,repeat=True,kFold=False)

#create dataset
set_seed(seed)
dataset = OmicsDrugDataset(omics_data_tensor_dict, drug_features_tensor, response_matrix_tensor, splitType, include_omics)

train_dataset = Subset(dataset, id_train.tolist())
val_dataset = Subset(dataset, id_val.tolist())
test_dataset = Subset(dataset, id_test.tolist())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#--------------------------------------------------------------------------------------------------------------------------

# train
# Init the neural network 
set_seed(seed)
model = Omics_DrugESPF_Model(omics_encode_dim_dict, drug_encode_dims, activation_func, activation_func_final, dense_layer_dim, device,
                        drug_embedding_feature_size, intermediate_size, num_attention_heads , attention_probs_dropout_prob, hidden_dropout_prob, omics_numfeatures_dict, max_drug_len,
                        TCGA_pretrain_weight_path_dict= TCGA_pretrain_weight_path_dict).to(device=device)
num_param = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fK" % (num_param/1e3))

optimizer = optim.Adam(model.parameters(), lr=learning_rate)# Initialize optimizer

best_epoch, best_weight, best_val_loss, train_epoch_loss_list, val_epoch_loss_list,_,attention_score_matrix , gradient_fig,gradient_norms_list = train( model, activation_func_final,
    optimizer,      batch_size,      num_epoch,      patience,      warmup_iters,      Decrease_percent,    continuous,
    learning_rate,      criterion,      valueMultiply,      train_loader,      val_loader,
    device,ESPF,Transformer,      seed=42, kfoldCV = None)

print("best Epoch : ",best_epoch,"best_val_loss : ",best_val_loss," batch_size : ",batch_size,
        "learning_rate : ",learning_rate," warmup_iters :" ,warmup_iters  ," with Decrease_percent : ",Decrease_percent )

# Saving the model weughts
hyperparameter_folder_path = f'./results/BestValLoss{best_val_loss:.7f}_BestEpo{best_epoch}_{hyperparameter_folder_part}' # /root/Winnie/PDAC
os.makedirs(hyperparameter_folder_path, exist_ok=True)
save_path = os.path.join(hyperparameter_folder_path, f'Best_Weight.pt')
torch.save(best_weight, save_path)
#--------------------------------------------------------------------------------------------------------------------------
# Save the config file to the result directory
# destination_config_path = os.path.join(hyperparameter_folder_path, os.path.basename(args.config))
shutil.copy(args.config, os.path.join(hyperparameter_folder_path, os.path.basename(args.config)))
#--------------------------------------------------------------------------------------------------------------------------
# store train and val loss per epoch
import json
epoch_loss_dict = {"train_epoch_loss_list": train_epoch_loss_list, "val_epoch_loss_list": val_epoch_loss_list}
json_data = json.dumps(epoch_loss_dict, indent=0)
with open(f"{hyperparameter_folder_path}/epoch_loss.json", "w") as json_file:
    json_file.write(json_data)
#--------------------------------------------------------------------------------------------------------------------------

loss_curve(model_name, train_epoch_loss_list, val_epoch_loss_list, best_epoch, best_val_loss,hyperparameter_folder_path, ylim_top=0.025)

#--------------------------------------------------------------------------------------------------------------------------
if gradient_norms_list:
    with open(f"{hyperparameter_folder_path}/gradient_norms_list.txt", "w") as txt_file:
        txt_file.write("\n".join(map(str, gradient_norms_list)))

if gradient_fig:
    #
    gradient_fig.savefig(f'{hyperparameter_folder_path}/Gradient Norms Over Epochs')
    
    


# Evaluation
#Evaluation on best fold best split id (train, val) with best_fold_best_weight 
set_seed(seed)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Evaluation on the train set
model.load_state_dict(best_weight)  
model = model.to(device=device)
epoch = None
train_loss, train_targets, train_outputs = evaluation(model, activation_func_final, epoch, num_epoch, val_epoch_loss_list, criterion, valueMultiply, train_loader, device,ESPF, Transformer, correlation='train', kfoldCV = None)
# Evaluation on the validation set
model.load_state_dict(best_weight)  
model = model.to(device=device)
val_loss, val_targets, val_outputs = evaluation(model, activation_func_final, epoch, num_epoch, val_epoch_loss_list, criterion, valueMultiply, val_loader, device,ESPF, Transformer, correlation='val', kfoldCV = None)
# Evaluation on the test set
model.load_state_dict(best_weight)
model = model.to(device=device)
test_loss, test_targets, test_outputs = evaluation(model, activation_func_final, epoch, num_epoch, val_epoch_loss_list, criterion, valueMultiply, test_loader, device,ESPF, Transformer, correlation='test', kfoldCV = None)

#--------------------------------------------------------------------------------------------------------------------------
# Correlation
train_pearson, train_spearman = correlation_func(splitType, AUC_df.values,AUC_df.index,AUC_df.columns,id_unrepeat_train,train_targets,train_outputs)
# print("\n")
# print("val set"+"="*20)
# print("val set"+"="*20)
val_pearson, val_spearman = correlation_func(splitType, AUC_df.values,AUC_df.index,AUC_df.columns,id_unrepeat_val,val_targets,val_outputs)
# print("\n")
# print("test set"+"="*20)
# print("test set"+"="*20)
test_pearson, test_spearman = correlation_func(splitType, AUC_df.values,AUC_df.index,AUC_df.columns,id_unrepeat_test,test_targets,test_outputs)
#--------------------------------------------------------------------------------------------------------------------------
#plot correlation_density
correlation_density(model_name,train_pearson,val_pearson,test_pearson,train_spearman,val_spearman,test_spearman, hyperparameter_folder_path)

# plot GroundTruth AUC and predicted AUC distribution
predicted_AUC = np.concatenate(train_outputs + val_outputs + test_outputs).tolist()
# print(predicted_AUC[:10])
print("predicted_AUC",np.array(predicted_AUC).shape)
GroundTruth_AUC = np.concatenate(train_targets + val_targets + test_targets).tolist()
print("GroundTruth_AUC",np.array(GroundTruth_AUC).shape)
# print(GroundTruth_AUC[:10])
#--------------------------------------------------------------------------------------------------------------------------
datas = [(train_targets, train_outputs, 'Train', 'red'),
                (val_targets, val_outputs, 'Validation', 'green'),
                (test_targets, test_outputs, 'Test', 'purple')]
# plot Density_Plot_of_AUC_Values of train val test datasets
Density_Plot_of_AUC_Values(datas,hyperparameter_folder_path)

#----------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------
output_file = f"{hyperparameter_folder_path}/result_performance.txt"
with open(output_file, "w") as file:
    # data range
    get_data_value_range(GroundTruth_AUC,"GroundTruth_AUC", file=file)
    get_data_value_range(predicted_AUC,"predicted_AUC", file=file)
    print('best epoch: ',best_epoch, file=file)
    print(f'Evaluation Training Loss: {train_loss:.6f}', file=file)
    print(f'Evaluation validation Loss: {val_loss:.6f}', file=file)
    print(f'Evaluation Test Loss: {test_loss:.6f}', file=file)
    print("\n")
# Pearson and Spearman statistics
    for name, pearson in [("Train", train_pearson),
                                    ("Validation", val_pearson),
                                    ("Test", test_pearson)]:
        print(f"Mean {name} Pearson {model_name}: {np.mean(pearson):.6f}", file=file)
        print(f"Median {name} Pearson {model_name}: {np.median(pearson):.6f}", file=file)
        print(f"Mode {name} Pearson {model_name}: {stats.mode(np.round(pearson,2))[0]}, count={stats.mode(np.round(pearson,2))[1]}", file=file)
    for name, spearman in [("Train", train_spearman),
                                    ("Validation", val_spearman),
                                    ("Test", test_spearman)]:
        print(f"Mean {name} Spearman {model_name}: {np.mean(spearman):.6f}", file=file)
        print(f"Median {name} Spearman {model_name}: {np.median(spearman):.6f}", file=file)
        print(f"Mode {name} Spearman {model_name}: {stats.mode(np.round(spearman,2))[0]}, count={stats.mode(np.round(spearman,2))[1]}", file=file)
    for name, pearson in [("Train", train_pearson),
                                    ("Validation", val_pearson),
                                    ("Test", test_pearson)]:
        print(f"Mean Median Mode {name} Pearson {model_name}:\t{np.mean(pearson):.6f}\t{np.median(pearson):.6f}\t{stats.mode(np.round(pearson,2))}", file=file)
    for name, spearman in [("Train", train_spearman),
                                    ("Validation", val_spearman),
                                    ("Test", test_spearman)]:
        print(f"Mean Median Mode {name} Pearson {model_name}:\t{np.mean(spearman):.6f}\t{np.median(spearman):.6f}\t{stats.mode(np.round(spearman,2))}", file=file)
    print("Output saved to:", output_file)