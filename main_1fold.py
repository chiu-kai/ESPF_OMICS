#main_kfold
import argparse
# import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader, Subset
import torch.nn.init as init
from sklearn.model_selection import KFold
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import random
import gc
import os
import importlib.util

from utils.ESPF_drug2emb import drug2emb_encoder
from utils.Model import Omics_DrugESPF_Model
from utils.split_data_id import split_id,repeat_func
from utils.create_dataloader import OmicsDrugDataset
from utils.train import train, evaluation
from utils.correlation import correlation_func
from utils.plot import loss_curve, correlation_density,Density_Plot_of_AUC_Values
from utils.tools import get_data_value_range,set_seed,get_vram_usage
from utils.Metrics import MetricsCalculator


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

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Training on device {device}.")

#--------------------------------------------------------------------------------------------------------------------------
set_seed(seed)
for omic_type in include_omics:
    # Read the file
    omics_data_dict[omic_type] = pd.read_csv(omics_files[omic_type], sep='\t', index_col=0)

    if test is True:
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

# matched AUCfile and omics_data samples
matched_samples = sorted(set(AUC_df.T.columns) & set(list(omics_data_dict.values())[0].T.columns))

AUC_df= (AUC_df.T[matched_samples]).T
if AUCtransform == "-log2":
    AUC_df = -np.log2(AUC_df)
if AUCtransform == "-log10":
    AUC_df = -np.log10(AUC_df)

if test is True:
    batch_size = 3
    num_epoch = 2
    print("batch_size",batch_size,"num_epoch:",num_epoch)
    drug_df=drug_df[:42]
    AUC_df=AUC_df.iloc[:76,:42]
    print("drug_df",drug_df.shape)
    print("AUC_df",AUC_df.shape)

#--------------------------------------------------------------------------------------------------------------------------
# 檢查有無重複的SMILES
if ESPF is True:
    drug_smiles =drug_df["smiles"] # 
    drug_names =drug_df.index
    # 挑出重複的SMILES
    duplicate =  drug_smiles[drug_smiles.duplicated(keep=False)]

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
    # print((drug_encode.index.values).shape)#(42,)
    # print((drug_encode).shape)#(42,)
    # print(type(drug_encode))#<class 'pandas.core.series.Series'>
    #print((drug_encode.values).shape)#(42,)
    # print(drug_encode.values.tolist())
    # Convert your data to tensors if they're in numpy
    drug_features_tensor = torch.tensor(np.array(drug_encode.values.tolist()), dtype=torch.long)
else:
    drug_encode = drug_df["MACCS166bits"]
    drug_encode_list = [list(map(int, item.split(','))) for item in drug_encode.values]
    print("MACCS166bits_drug_encode_list type: ",type(drug_encode_list))
    # Convert your data to tensors if they're in numpy
    drug_features_tensor = torch.tensor(np.array(drug_encode_list), dtype=torch.long)
#--------------------------------------------------------------------------------------------------------------------------
num_ccl = list(omics_data_dict.values())[0].shape[0]
num_drug = drug_encode.shape[0]
print("num_ccl,num_drug: ",num_ccl,num_drug)

#--------------------------------------------------------------------------------------------------------------------------
# Convert your data to tensors if they're in numpy
response_matrix_tensor = torch.tensor(AUC_df.values, dtype=torch.float32)

#--------------------------------------------------------------------------------------------------------------------------
# randomly split
# 90% for training(10% for validation) and 10% for testing

id_unrepeat_train, id_unrepeat_val, id_unrepeat_test, _ , id_train, id_val, id_test= split_id(num_ccl,num_drug,splitType,kfoldCV,repeat=True)

#create dataset
set_seed(seed)
dataset = OmicsDrugDataset(omics_data_tensor_dict, drug_features_tensor, response_matrix_tensor, splitType, include_omics)

test_dataset = Subset(dataset, id_test.tolist())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

train_dataset = Subset(dataset, id_train.tolist())# create dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dataset = Subset(dataset, id_val.tolist())
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# train
# Init the neural network 
set_seed(seed)
model = Omics_DrugESPF_Model(omics_encode_dim_dict, drug_encode_dims, activation_func, activation_func_final, dense_layer_dim, device,
                        drug_embedding_feature_size, intermediate_size, num_attention_heads , attention_probs_dropout_prob, hidden_dropout_prob, omics_numfeatures_dict, max_drug_len,
                        TCGA_pretrain_weight_path_dict= TCGA_pretrain_weight_path_dict).to(device=device)
num_param = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fK" % (num_param/1e3))

optimizer = optim.Adam(model.parameters(), lr=learning_rate)# Initialize optimizer

best_epoch, best_weight, best_val_loss, train_epoch_loss_list, val_epoch_loss_list,best_val_epoch_train_loss,attention_score_matrix , gradient_fig,gradient_norms_list = train( model,
    optimizer,      batch_size,      num_epoch,      patience,      warmup_iters,      Decrease_percent,    continuous,
    learning_rate,      criterion,     train_loader,      val_loader,
    device,ESPF,Transformer, seed, kfoldCV)

print("best Epoch : ",best_epoch,"best_val_loss : ",best_val_loss,"best_val_epoch_train_loss : ",best_val_epoch_train_loss," batch_size : ",batch_size,
        "learning_rate : ",learning_rate," warmup_iters :" ,warmup_iters  ," with Decrease_percent : ",Decrease_percent )

# Saving the model weughts
hyperparameter_folder_path = f'./results/BestValLoss{best_val_loss:.7f}_BestValEpo{best_epoch}_{hyperparameter_folder_part}' 
os.makedirs(hyperparameter_folder_path, exist_ok=True)
save_path = os.path.join(hyperparameter_folder_path, f'BestValWeight.pt')
torch.save(best_weight, save_path)

# store train and val loss per epoch
# Convert all items to Python-native float(JSON-serializable types)
# train_epoch_loss_list = [float(value) if isinstance(value, (np.float32, np.float64)) else value for value in train_epoch_loss_list]
# val_epoch_loss_list = [float(value) if isinstance(value, (np.float32, np.float64)) else value for value in val_epoch_loss_list]
# import json
# epoch_loss_dict = {"train_epoch_loss_list": train_epoch_loss_list, "val_epoch_loss_list": val_epoch_loss_list}
# json_data = json.dumps(epoch_loss_dict, indent=0)
# with open(f"{hyperparameter_folder_path}/epoch-loss.json", "w") as json_file:
#     json_file.write(json_data)
#--------------------------------------------------------------------------------------------------------------------------
# if gradient_norms_list:
#     with open(f"{hyperparameter_folder_path}/gradient-norms-list.txt", "w") as txt_file:
#         txt_file.write("\n".join(map(str, gradient_norms_list)))
# if gradient_fig:
#     gradient_fig.savefig(f'{hyperparameter_folder_path}/Gradient-Norms-Over-Epochs')

loss_curve(model_name, train_epoch_loss_list, val_epoch_loss_list, best_epoch, best_val_loss,hyperparameter_folder_path, ylim_top=None)



# Evaluation
#Evaluation on best fold best split id (train, val) with best_fold_best_weight 
set_seed(seed)
train_dataset = Subset(dataset, id_train.tolist())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# set_seed(seed)
# model = Omics_DrugESPF_Model(omics_encode_dim_dict, drug_encode_dims, activation_func, activation_func_final, dense_layer_dim, device,
#                         drug_embedding_feature_size, intermediate_size, num_attention_heads , attention_probs_dropout_prob, hidden_dropout_prob, omics_numfeatures_dict, max_drug_len,
#                         TCGA_pretrain_weight_path_dict= TCGA_pretrain_weight_path_dict).to(device=device)
# num_param = sum([param.nelement() for param in model.parameters()])
# print("Number of parameter: %.2fK" % (num_param/1e3))

# Compute and print all metrics
metrics_calculator = MetricsCalculator() 
# Evaluation on the train set
model.load_state_dict(best_weight)  
model.to(device=device)
train_loss, train_targets, train_outputs = evaluation(model, val_epoch_loss_list, criterion, train_loader, device,ESPF, Transformer, threshold, few_weight, more_weight, correlation='train')
train_metrics = metrics_calculator.compute_all_metrics(np.concatenate(train_targets), np.concatenate(train_outputs),set_name='train_set')
# metrics_calculator.print_results(set_name='train_set')
# Evaluation on the validation set
val_loss, val_targets, val_outputs = evaluation(model, val_epoch_loss_list, criterion, val_loader, device,ESPF, Transformer, threshold, few_weight, more_weight, correlation='val')
val_metrics = metrics_calculator.compute_all_metrics(np.concatenate(val_targets), np.concatenate(val_outputs),set_name='val_set')
# metrics_calculator.print_results(set_name='val_set')
# Evaluation on the test set
test_loss, test_targets, test_outputs = evaluation(model, val_epoch_loss_list, criterion, test_loader, device,ESPF, Transformer, threshold, few_weight, more_weight, correlation='test')
test_metrics = metrics_calculator.compute_all_metrics(np.concatenate(test_targets), np.concatenate(test_outputs),set_name='test_set')
# metrics_calculator.print_results(set_name='test_set') # test_targets.dtype float32; test_loss.dtype float32


#--------------------------------------------------------------------------------------------------------------------------
# Correlation
train_pearson, train_spearman,train_AllSameValuesList_count = correlation_func(splitType, AUC_df.values,AUC_df.index,AUC_df.columns,best_fold_id_unrepeat_train,train_targets,train_outputs)
# print("\n")
# print("val set"+"="*20)
# print("val set"+"="*20)
val_pearson, val_spearman,val_AllSameValuesList_count = correlation_func(splitType, AUC_df.values,AUC_df.index,AUC_df.columns,best_fold_id_unrepeat_val,val_targets,val_outputs)
# print("\n")
# print("test set"+"="*20)
# print("test set"+"="*20)
test_pearson, test_spearman,test_AllSameValuesList_count = correlation_func(splitType, AUC_df.values,AUC_df.index,AUC_df.columns,id_unrepeat_test,test_targets,test_outputs)
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
    
    file.write(f'\nhyperparameter_print\n{hyperparameter_print}')
    #----------------After Training-------------- #驗證與Evaluation是否一致
    
    file.write(f"best_val_loss in training step epochs: {best_val_loss:.7f}\n")
    file.write(f"best_val_epoch_train_loss in training step epochs: {best_val_epoch_train_loss:.7f}\n")
    
    file.write(f'criterion: {criterion.loss_type}\n')
    file.write(f'best epoch: {best_epoch}\n')
    #'----------------After Evaluation-------------- #紀錄
    file.write(f'Evaluation Train Loss: {train_loss:.7f}\n')
    file.write(f'Evaluation validation Loss: {val_loss:.7f}\n')
    file.write(f'Evaluation Test Loss: {test_loss:.7f}\n')
    
    # Metrics
    for name, metrics in [("Train_set", train_metrics),("val_set", val_metrics),("test_set", test_metrics)]:
        for key, value in metrics.items():
            if key != 'Evaluation':
                file.write(f"Metrics {name} {key} : {value:.6f}\n")

# Pearson and Spearman statistics
    for name, pearson in [("Train", train_pearson),
                                    ("Validation", val_pearson),
                                    ("Test", test_pearson)]:
        file.write(f"Mean {name} Pearson {model_name}: {np.mean(pearson):.6f} ± {np.std(pearson):.4f}\n")
        file.write(f"Skewness {name} Pearson {model_name}: {stats.skew(pearson, bias=False, nan_policy='raise'):.6f}\n")
        file.write(f"Median {name} Pearson {model_name}: {np.median(pearson):.6f}\n")
        file.write(f"Mode {name} Pearson {model_name}: {stats.mode(np.round(pearson,2))[0]}, count={stats.mode(np.round(pearson,2))[1]}\n")
    for name, spearman in [("Train", train_spearman),
                                    ("Validation", val_spearman),
                                    ("Test", test_spearman)]:
        file.write(f"Mean {name} Spearman {model_name}: {np.mean(spearman):.6f} ± {np.std(spearman):.4f}\n")
        file.write(f"Skewness {name} Spearman {model_name}: {stats.skew(spearman, bias=False, nan_policy='raise'):.6f}\n")
        file.write(f"Median {name} Spearman {model_name}: {np.median(spearman):.6f}\n")
        file.write(f"Mode {name} Spearman {model_name}: {stats.mode(np.round(spearman,2))[0]}, count={stats.mode(np.round(spearman,2))[1]}\n")
        

    # check All Same Predicted Values Item_Count in {name}set # EX: 一個藥對應每個ccl時，輸出值都一樣
    for name, AllSameValuesList_count in [("Train", train_AllSameValuesList_count),
                                    ("Validation", val_AllSameValuesList_count),
                                    ("Test", test_AllSameValuesList_count)]:
        file.write(f"All Same Predicted Values Item_Count in {name}set: {AllSameValuesList_count}\n")

    for name, pearson in [("Train", train_pearson),
                                        ("Validation", val_pearson),
                                        ("Test", test_pearson)]:
        file.write(f"Mean Median Mode {name} Pearson {model_name}:\t{np.mean(pearson):.6f} ± {np.std(pearson):.4f}\t{stats.skew(pearson, bias=False, nan_policy='raise'):.6f}\t {np.median(pearson):.6f}\t{stats.mode(np.round(pearson,2))}\n")
    for name, spearman in [("Train", train_spearman),
                                        ("Validation", val_spearman),
                                        ("Test", test_spearman)]:
        file.write(f"Mean Median Mode {name} Spearman {model_name}:\t{np.mean(spearman):.6f} ± {np.std(spearman):.4f}\t{stats.skew(spearman, bias=False, nan_policy='raise'):.6f}\t {np.median(spearman):.6f}\t{stats.mode(np.round(spearman,2))}\n")
    print("Output saved to:", output_file)

