#main_kfold_GDSC
# pip install subword-nmt seaborn lifelines openpyxl matplotlib scikit-learn openTSNE
# pip install torchmetrics==1.2.0 pandas==2.1.4 numpy==1.26.4
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader, Subset
import torch.nn.init as init
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import random
import gc
import os
import importlib.util
import pickle
import torchmetrics
from scipy.stats import ttest_ind
import time

from utils.ESPF_drug2emb import drug2emb_encoder
from utils.Model import Omics_DrugESPF_Model, Omics_DCSA_Model
from utils.split_data_id import split_id,repeat_func
from utils.create_dataloader import OmicsDrugDataset
from utils.train import train, evaluation
from utils.correlation import correlation_func
from utils.plot import loss_curve, correlation_density, Density_Plot_of_AUC_Values, Confusion_Matrix_plot, TCGA_predAUDRC_box_plot_twoClass
from utils.tools import get_data_value_range,set_seed,get_vram_usage
print("*"*100)

# 設定命令列引數
parser = argparse.ArgumentParser(description="import config to main")
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

# information
struct_time   = time.localtime()
timestamp    = time.strftime("%Y-%m%d-%H%M", struct_time)

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Training on device {device}.")

# 檢查exp和AUC的samples是否一致
if deconfound_EXPembedding is True:
    with open(omics_files['Exp'], 'rb') as f:
        latent_dict = pickle.load(f)
        exp_df = pd.DataFrame(latent_dict).T
        
else:
    exp_df = pd.read_csv(omics_files["Exp"], sep=',', index_col=0)
exp_df = exp_df.sort_index(axis=0).sort_index(axis=1)
AUC_df_numerical = pd.read_csv(AUC_df_path_numerical, sep=',', index_col=0)
AUC_df_numerical = AUC_df_numerical.sort_index(axis=0).sort_index(axis=1)
print(f"exp_df samples: {len(exp_df.index)} , AUC_df_numerical samples: {len(AUC_df_numerical.index)}")
matched_samples = sorted(set(AUC_df_numerical.index) & set(exp_df.index))
print("len(matched_samples)",len(matched_samples))
# 讀取omics資料
set_seed(seed)

scaler_dict = {}  # To store scalers for each omic_type
for omic_type in include_omics:
    if deconfound_EXPembedding is True:
        omics_data_dict[omic_type] = exp_df.loc[matched_samples]
    else:
        omics_data_dict[omic_type] = pd.read_csv(omics_files[omic_type], sep=',', index_col=0).loc[matched_samples]
        omics_data_dict[omic_type] = omics_data_dict[omic_type].sort_index(axis=0).sort_index(axis=1)
        if omic_type == "Exp":# apply Column-wise Standardization 
            scaler = StandardScaler() 
            omics_data_dict[omic_type] = pd.DataFrame(scaler.fit_transform(omics_data_dict[omic_type]),index=omics_data_dict[omic_type].index,columns=omics_data_dict[omic_type].columns)
            scaler_dict[omic_type] = scaler  # save the fitted scaler for latter inference
    if test is True:
        # Specify the index as needed
        omics_data_dict[omic_type] = omics_data_dict[omic_type][:76]  # Adjust the row selection as needed
        
    omics_data_tensor_dict[omic_type]  = torch.tensor(omics_data_dict[omic_type].values, dtype=torch.float32).to(device)
    omics_numfeatures_dict[omic_type] = omics_data_tensor_dict[omic_type].shape[1]

    print(f"{omic_type} tensor shape:", omics_data_tensor_dict[omic_type].shape)
    print(f"{omic_type} num_features",omics_numfeatures_dict[omic_type])


drug_df = pd.read_csv( drug_df_path, sep=',', index_col=0)
print("drug_df",drug_df.shape)
drug_df = drug_df.sort_index(axis=0).sort_index(axis=1)
if "BRD_ID" in drug_df.columns:
    drug_df["BRD_ID"] = drug_df["BRD_ID"].replace({"BRD-K61250484-001-02-3": "BRD-6125",
                                                    "BRD-K91701654-001-03-1 (CID5354033)": "BRD-K91701654-001-03-1",
                                                    "BRD-K18787491-001-08-6 (CID3006531)": "BRD-K18787491-001-08-6"})
print("drug_df",drug_df.shape)
print("AUC_df_numerical",AUC_df_numerical.shape)
# matched AUCfile and omics_data samples
AUC_df_numerical= (AUC_df_numerical.loc[matched_samples])
print("AUC_df_numerical match samples",AUC_df_numerical.shape)
# median_value = np.nanmedian(AUC_df_numerical.values)  # Directly calculate median, ignoring NaNs
# print("median_value",median_value)    
if 'BCE' in criterion.loss_type :
    AUC_df = pd.read_csv(AUC_df_path, sep=',', index_col=0).loc[matched_samples] # binary data
    AUC_df = AUC_df.sort_index(axis=0).sort_index(axis=1)
    print("AUC_df",AUC_df.shape)
    if "BRD_ID" in drug_df.columns:
        drug_df = drug_df[drug_df["BRD_ID"].isin(AUC_df.columns.str.extract(r"(BRD-[^\)]+)", expand=False))]
    print("drug_df",drug_df.shape)
else:
    AUC_df = AUC_df_numerical.copy()
del AUC_df_numerical

if AUCtransform == "-log2":
    AUC_df = -np.log2(AUC_df)
if AUCtransform == "-log10":
    AUC_df = -np.log10(AUC_df)

if test is True:
    drug_df=drug_df[:42]
    AUC_df=AUC_df.iloc[:76,:42]
    print("drug_df",drug_df.shape)
    print("AUC_df",AUC_df.shape)

if 'weighted' in criterion.loss_type :    
    if 'BCE' in criterion.loss_type :
        weighted_threshold = None
        total_samples = (~np.isnan(AUC_df.values)).sum().item()
        fewWt_samples = (AUC_df.values == 0).sum().item()
        moreWt_samples = (AUC_df.values == 1).sum().item()
        few_weight = total_samples / (2 * fewWt_samples)  
        more_weight = total_samples / (2 * moreWt_samples) 
    else:
        # Set threshold based on the 90th percentile # 將高於threshold的AUC權重增加
        weighted_threshold = np.nanpercentile(AUC_df.values, 90)    
        total_samples = (~np.isnan(AUC_df.values)).sum().item()
        fewWt_samples = (AUC_df.values > weighted_threshold).sum().item()
        moreWt_samples = total_samples - fewWt_samples
        few_weight = total_samples / (2 * fewWt_samples)  
        more_weight = total_samples / (2 * moreWt_samples)   
        # print("weighted_threshold",weighted_threshold)
        # print("total_samples",total_samples)
        # print("few_samples",few_samples)
        # print("more_samples",more_samples)
        # print("few_weight",few_weight)
        # print("more_weight",more_weight)
else:
    weighted_threshold = None
    few_weight = None
    more_weight = None


# convert SMILES to subword token by ESPF
if ESPF is True:
    drug_smiles =drug_df["SMILES"] # 
    drug_names =drug_df.index
    # 挑出重複的SMILES
    duplicate =  drug_smiles[drug_smiles.duplicated(keep=False)]
    vocab_path = "./ESPF/drug_codes_chembl_freq_1500.txt" # token
    sub_csv = pd.read_csv("./ESPF/subword_units_map_chembl_freq_1500.csv")# token with frequency
    drug_encode = pd.Series(drug_smiles).apply(drug2emb_encoder, args=(vocab_path, sub_csv, max_drug_len))
    drug_features_tensor = torch.tensor(np.array([i[:2] for i in drug_encode.values]), dtype=torch.long).to(device)#drug_features_tensor = torch.tensor(np.array(drug_encode.values.tolist()), dtype=torch.long).to(device)
else:
    drug_encode = drug_df["MACCS166bits"]
    drug_encode_list = [list(map(int, item.split(','))) for item in drug_encode.values]
    print("MACCS166bits_drug_encode_list type: ",type(drug_encode_list))
    drug_features_tensor = torch.tensor(np.array(drug_encode_list), dtype=torch.long).to(device)
#--------------------------------------------------------------------------------------------------------------------------
num_ccl = list(omics_data_dict.values())[0].shape[0]
num_drug = drug_encode.shape[0]
print("num_ccl,num_drug: ",num_ccl,num_drug)

# Convert your data to tensors if they're in numpy
# AUC_df = AUC_df.apply(pd.to_numeric, errors='coerce')# Ensure all values are numeric, coercing non-numeric ones to NaN
response_matrix_tensor = torch.tensor(AUC_df.values, dtype=torch.float32).to(device)
print(response_matrix_tensor.shape)
print(drug_encode.values[0][2])

# generate data split id
id_unrepeat_test, id_unrepeat_train_val = split_id(num_ccl,num_drug,splitType,kfoldCV,repeat=True)
# repeat the test id
if splitType in ['byCCL', 'ModelID']:
    repeatNum = num_drug
elif splitType in ['byDrug', 'drug_name']:
    repeatNum = num_ccl
id_test = repeat_func(id_unrepeat_test, repeatNum, setname='test')

#create dataset
set_seed(seed)
dataset = OmicsDrugDataset(omics_data_tensor_dict, drug_features_tensor, response_matrix_tensor, splitType, include_omics)

test_dataset = Subset(dataset, id_test.tolist())
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) #, num_workers=4, pin_memory=True


# k-fold run
kfold_losses= {}
kfold_metrics={}
BF_BE_trainLoss_WO_penalty_ls = []#  for train every epoch loss plot (BF)
BF_BE_valLoss_WO_penalty_ls = []#  for validation every epoch loss plot (BF)
BF_test_loss = float('inf')
BF_best_weight=None
set_seed(seed)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=kfoldCV, shuffle=True, random_state=seed) #shuffle the order of split subset
for fold, (id_unrepeat_train, id_unrepeat_val) in enumerate(kfold.split(id_unrepeat_train_val)):
    print(f'FOLD {fold}')
    print('--------------------------------------------------------------')
    print(id_unrepeat_train.shape,id_unrepeat_val.shape)
    # correct the id 
    id_unrepeat_train = np.array(id_unrepeat_train_val)[id_unrepeat_train.tolist()]
    id_unrepeat_val = np.array(id_unrepeat_train_val)[id_unrepeat_val.tolist()]
    # repeat the id 
    id_train = repeat_func(id_unrepeat_train, repeatNum, setname='train')
    id_val = repeat_func(id_unrepeat_val, repeatNum, setname='val')

    set_seed(seed)
    train_dataset = Subset(dataset, id_train.tolist())# create dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #, num_workers=4, pin_memory=True
    val_dataset = Subset(dataset, id_val.tolist())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) #, num_workers=4, pin_memory=True

    # train
    # Init the neural network 
    set_seed(seed)
    if model_name == "Omics_DrugESPF_Model":
        model = Omics_DrugESPF_Model(omics_encode_dim_dict, drug_encode_dims, activation_func, activation_func_final, dense_layer_dim, device, ESPF, Drug_SelfAttention, pos_emb_type,
                            drug_embedding_feature_size, intermediate_size, num_attention_heads , attention_probs_dropout_prob, hidden_dropout_prob, omics_numfeatures_dict, max_drug_len,
                            n_layer,deconfound_EXPembedding,TCGA_pretrain_weight_path_dict= TCGA_pretrain_weight_path_dict)
    elif model_name == "Omics_DCSA_Model":
        model = Omics_DCSA_Model(omics_encode_dim_dict, drug_encode_dims, activation_func, activation_func_final, dense_layer_dim, device, ESPF, Drug_SelfAttention, pos_emb_type,
                            drug_embedding_feature_size, intermediate_size, num_attention_heads , attention_probs_dropout_prob, hidden_dropout_prob, omics_numfeatures_dict, max_drug_len,
                            n_layer,deconfound_EXPembedding,TCGA_pretrain_weight_path_dict= TCGA_pretrain_weight_path_dict)

    model.to(device=device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)# Initialize optimizer

    (best_epoch, best_weight, BE_val_loss, BE_val_train_loss_WO_penalty, 
     BEpo_trainLoss_W_penalty_ls, BEpo_trainLoss_WO_penalty_ls, 
     BEpo_valLoss_W_penalty_ls, BEpo_valLoss_WO_penalty_ls, 
     BE_val_targets, BE_val_outputs, BE_train_targets , BE_train_outputs,
     gradient_fig, gradient_norms_list) = train( model, optimizer, 
                                               criterion, train_loader, val_loader, device,
                                               ESPF, Drug_SelfAttention, seed ,
                                               weighted_threshold, few_weight, more_weight, TrackGradient)
    # BE_val_loss = mean_batch_eval_loss_WO_penalty
    print("best Epoch : ",best_epoch,"BE_val_loss : ",BE_val_loss,
          "BE_val_train_loss_WO_penalty : ",BE_val_train_loss_WO_penalty," batch_size : ",batch_size,
          "learning_rate : ",learning_rate," decrese_epoch :" ,decrese_epoch  ,
          " with Decrease_percent : ",Decrease_percent )

    kfold_losses[fold] = { 'train': BE_val_train_loss_WO_penalty,  # Train loss in best Validation epoch
                           'val': BE_val_loss,  # best epoch
                           'test': None,  # Placeholder for test loss
                          }   
    val_metrics, best_prob_threshold = metrics_calculator(torch.cat(BE_val_targets), torch.cat(BE_val_outputs), best_prob_threshold, metric ,dataset="val")
    train_metrics, _ = metrics_calculator(torch.cat(BE_train_targets), torch.cat(BE_train_outputs), best_prob_threshold, metric, dataset="train")
    
    kfold_metrics[fold] = {'train': train_metrics, 'val': val_metrics, 'test': None 
                           }   
    
    # Evaluation on the test set for each fold's best model to pick the best fold for later inference
    model.load_state_dict(best_weight)  
    model.to(device=device)
    
    (BE_test_targets,BE_test_outputs,
    test_loss_WO_penalty,test_outputs_before_final_activation_list) = evaluation (model, None,None, 
                                        criterion, test_loader, device, ESPF, Drug_SelfAttention,
                                        weighted_threshold, few_weight, more_weight, 
                                        outputcontrol='correlation')
    
    test_metrics, _  = metrics_calculator(torch.cat(BE_test_targets), torch.cat(BE_test_outputs),best_prob_threshold, metric, dataset="test")

    kfold_losses[fold]['test'] = test_loss_WO_penalty
    kfold_metrics[fold]['test'] = test_metrics

    # save best fold testing loss model weight
    if test_loss_WO_penalty < BF_test_loss:
        BF_test_loss = test_loss_WO_penalty
        BF_BE_trainLoss_WO_penalty_ls = BEpo_trainLoss_WO_penalty_ls #  for train epoch loss plot
        BF_BE_valLoss_WO_penalty_ls = BEpo_valLoss_WO_penalty_ls #  for validation epoch loss plot
        if criterion.regular_type is not None:
            BF_BEpo_trainLoss_W_penalty_ls = BEpo_trainLoss_W_penalty_ls #  for train epoch loss plot
            BF_BEpo_valLoss_W_penalty_ls = BEpo_valLoss_W_penalty_ls #  for validation epoch loss plot
        BF_best_epoch = best_epoch #  for validation epoch loss plot
        BF_BE_val_loss = BE_val_loss #  for validation epoch loss plot # BE_val_loss = BE_val_loss_WO_penalty
        BF_best_weight = copy.deepcopy(best_weight) # best fold best epoch 
        BF = fold
        BF_id_unrepeat_train= id_unrepeat_train# for correlation
        BF_id_unrepeat_val= id_unrepeat_val  
        BF_val_targets   , BF_val_outputs    = BE_val_targets   , BE_val_outputs
        BF_train_targets , BF_train_outputs  = BE_train_targets , BE_train_outputs
        BF_test_targets  , BF_test_outputs   = BE_test_targets  , BE_test_outputs
        BF_test_outputs_before_final_activation_list=test_outputs_before_final_activation_list
        BF_best_prob_threshold = best_prob_threshold # for BCE metric
    del model 
    # Set the current device
    torch.cuda.set_device("cuda:0")
    # Optionally, force garbage collection to release memory 
    gc.collect()
    # Empty PyTorch cache
    torch.cuda.empty_cache() # model 會從GPU消失，所以要evaluation時要重新load model
# Saving the model weughts
hyperparameter_folder_path = f"./results/{timestamp}_BF{BF}_{criterion.loss_type}_test_loss{BF_test_loss:.7f}_BestValEpo{BF_best_epoch}_{hyperparameter_folder_part}_{''.join([f'{k}{v}' for k, v in omics_numfeatures_dict.items()])}_nlayer{n_layer}_DA-{DA_Folder}" # /root/Winnie/PDAC
os.makedirs(hyperparameter_folder_path, exist_ok=True)
save_path = os.path.join(hyperparameter_folder_path, f'BestValWeight.pt')
torch.save(BF_best_weight, save_path)


#plot loss curve
loss_curve(model_name, BF_BE_trainLoss_WO_penalty_ls, BF_BE_valLoss_WO_penalty_ls, 
           BF_best_epoch, BF_BE_val_loss,hyperparameter_folder_path,
             loss_type="loss_WO_penalty")
if criterion.regular_type is not None:
    loss_curve(model_name, BF_BEpo_trainLoss_W_penalty_ls, BF_BEpo_valLoss_W_penalty_ls, 
               BF_best_epoch, BF_BE_val_loss,hyperparameter_folder_path, 
               loss_type="loss_W_penalty")




if "BCE" not in criterion.loss_type:
    # calculate correlation
    (train_pearson, train_spearman,
    train_AllSameValuesList_count) = correlation_func(splitType, AUC_df.values,AUC_df.index,AUC_df.columns,
                                                    BF_id_unrepeat_train, 
                                                    torch.cat(BF_train_targets), torch.cat(BF_train_outputs))

    (val_pearson, val_spearman,
    val_AllSameValuesList_count) = correlation_func(splitType, AUC_df.values,AUC_df.index,AUC_df.columns,
                                                    BF_id_unrepeat_val, 
                                                    torch.cat(BF_val_targets), torch.cat(BF_val_outputs))


    (test_pearson, test_spearman,
    test_AllSameValuesList_count) = correlation_func(splitType, AUC_df.values,AUC_df.index,AUC_df.columns,
                                                    id_unrepeat_test, 
                                                    torch.cat(BF_test_targets), torch.cat(BF_test_outputs))
    #--------------------------------------------------------------------------------------------------------------------------
    #plot correlation_density
    correlation_density(model_name,train_pearson,val_pearson,test_pearson,
                        train_spearman,val_spearman,test_spearman, 
                        hyperparameter_folder_path)

#--------------------------------------------------------------------------------------------------------------------------
    datas = [(BF_train_targets, BF_train_outputs, 'Train', 'red'),
             (BF_val_targets, BF_val_outputs, 'Validation', 'green'),
             (BF_test_targets, BF_test_outputs, 'Test', 'purple')]
    # plot Density_Plot_of_AUC_Values of train val test datasets
    Density_Plot_of_AUC_Values(datas,hyperparameter_folder_path)

#----------------------------------------------------------------------------------


if 'BCE' in criterion.loss_type :
    (train_cm , train_GT_0_count, train_GT_1_count, 
    train_pred_binary_0_count, train_pred_binary_1_count) =metrics_calculator.confusion_matrix(torch.cat(BF_train_targets), torch.cat(BF_train_outputs), BF_best_prob_threshold)
    (val_cm ,  val_GT_0_count, val_GT_1_count, 
    val_pred_binary_0_count, val_pred_binary_1_count ) =metrics_calculator.confusion_matrix(torch.cat(BF_val_targets), torch.cat(BF_val_outputs), BF_best_prob_threshold)
    (test_cm ,  test_GT_0_count, test_GT_1_count, 
    test_pred_binary_0_count, test_pred_binary_1_count ) =metrics_calculator.confusion_matrix(torch.cat(BF_test_targets), torch.cat(BF_test_outputs), BF_best_prob_threshold)

    # plot confusion matrix
    cm_datas = [(train_cm, 'Train', 'Reds'),  (val_cm, 'Validation', 'Greens'),   (test_cm, 'Test', 'Blues')]
    Confusion_Matrix_plot(cm_datas,hyperparameter_folder_path=hyperparameter_folder_path)


output_file = f"{hyperparameter_folder_path}/BF{BF}_result_performance.txt"
with open(output_file, "w") as file:
    if "BCE" not in criterion.loss_type:
        # data range
        get_data_value_range(torch.cat(BF_train_targets + BF_val_targets + BF_test_targets).tolist(),"GroundTruth_AUC", file=file)
        get_data_value_range(torch.cat(BF_train_outputs + BF_val_outputs + BF_test_outputs).tolist(),"predicted_AUC", file=file)

    file.write(f'\nhyperparameter_print\n{hyperparameter_print}')  
    
    file.write(f'kfold_losses:\n {kfold_losses}\n')# all fold loss on each set

    file.write(f'criterion: {criterion.loss_type}, weight_regularization: {criterion.regular_type}, regular_lambda: {criterion.regular_lambda}, penalty_value:{criterion.penalty_value}\n\n')
    file.write(f'weighted_threshold:{weighted_threshold}, fewSample_weight:{few_weight}, moreSample_weight:{more_weight}\n\n')
    # Calculate mean and standard deviation of the all folds loss
    for set in ['train', 'val', 'test']:
        Folds_losses = [loss[set] for loss in kfold_losses.values()]
        file.write(f"Average KFolds Model {set.capitalize()} {criterion.loss_type}: {np.mean(Folds_losses):.6f} ± {np.std(Folds_losses):.6f}\n")
    
    for type in metrics_type_set:
        for set in ['train', 'val', 'test']:
            Folds_values = [value[set][type] for value in kfold_metrics.values()]
            file.write(f"Average KFolds Model {set.capitalize()} {type}: {torch.mean(torch.stack(Folds_values)):.6f} ± {torch.std(torch.stack(Folds_values)):.6f}\n")

    file.write(f'BF: {BF}\n')
    file.write(f'BF_best_epoch: {BF_best_epoch}\n')

    file.write(f"Best fold {BF} {criterion.loss_type} train Loss: {(kfold_losses[BF]['train']):.7f}\n")
    file.write(f"Best fold {BF} {criterion.loss_type} val Loss: {BF_BE_val_loss:.7f}\n") # = (kfold_losses[BF]['val'])
    file.write(f"Best fold {BF} {criterion.loss_type} test Loss: {BF_test_loss:.7f}\n") # = (kfold_losses[BF]['test'])


    for type in metrics_type_set:
        for set in ['train', 'val', 'test']:
            BFolds_value = [value[set][type] for value in kfold_metrics.values()][BF]
            file.write(f"Best Fold {BF} {set.capitalize()} {type}: {BFolds_value:.7f}\n")
    
    if 'BCE' in criterion.loss_type :
        file.write(f"Best Fold {BF} Train TP TN FP FN: {train_cm[1,1]}_{train_cm[0,0]}_{train_cm[0,1]}_{train_cm[1,0]}\n"
                   f"Best Fold {BF} Val TP TN FP FN: {val_cm[1,1]}_{val_cm[0,0]}_{val_cm[0,1]}_{val_cm[1,0]}\n"
                   f"Best Fold {BF} Test TP TN FP FN: {test_cm[1,1]}_{test_cm[0,0]}_{test_cm[0,1]}_{test_cm[1,0]}\n"
                   f"Best Fold {BF} Train GT_count_0_1: {train_GT_0_count}_{train_GT_1_count}\n"
                   f"Best Fold {BF} Train pred_binary_count_0_1: {train_pred_binary_0_count}_{train_pred_binary_1_count}\n"
                   f"Best Fold {BF} Val GT_count_0_1: {val_GT_0_count}_{val_GT_1_count}\n"
                   f"Best Fold {BF} Val pred_binary_count_0_1: {val_pred_binary_0_count}_{val_pred_binary_1_count}\n"
                   f"Best Fold {BF} Test GT_count_0_1: {test_GT_0_count}_{test_GT_1_count}\n"
                   f"Best Fold {BF} Test pred_binary_count_0_1: {test_pred_binary_0_count}_{test_pred_binary_1_count}\n")   
        file.write(f"\nBF_best_val_prob_threshold: {BF_best_prob_threshold}\n") 
    else:
    # Pearson and Spearman statistics
        # <=0的都=0
        train_pearson = np.maximum( 0, np.array(train_pearson) )
        val_pearson = np.maximum( 0, np.array(val_pearson) )
        test_pearson = np.maximum( 0, np.array(test_pearson) )
        train_spearman = np.maximum( 0, np.array(train_spearman) )
        val_spearman = np.maximum( 0, np.array(val_spearman) )
        test_spearman = np.maximum( 0, np.array(test_spearman) )

        results = {"Mean": [], "Median": [], "Mode": [], "Skewness": []}
        for name, pearson in [("Train", train_pearson), ("Validation", val_pearson), ("Test", test_pearson)]:
            results["Mean"].append(f"Mean {name} Pearson: {np.mean(pearson):.6f} ± {np.std(pearson):.4f}")
            results["Median"].append(f"Median {name} Pearson: {np.median(pearson):.6f}")

            mode_value, mode_count = stats.mode(np.round(pearson, 2), keepdims=True)
            results["Mode"].append(f"Mode {name} Pearson: {mode_value[0]} count={mode_count[0]}")

            results["Skewness"].append(f"Skewness {name} Pearson: {stats.skew(pearson, bias=False, nan_policy='raise'):.6f}")
        file.write("\n".join("\n".join(v) for v in results.values()) + "\n")

        results = {"Mean": [], "Median": [], "Mode": [], "Skewness": []}
        for name, spearman in [("Train", train_spearman), ("Validation", val_spearman), ("Test", test_spearman)]:
            results["Mean"].append(f"Mean {name} spearman: {np.mean(spearman):.6f} ± {np.std(spearman):.4f}")
            results["Median"].append(f"Median {name} spearman: {np.median(spearman):.6f}")

            mode_value, mode_count = stats.mode(np.round(spearman, 2), keepdims=True)
            results["Mode"].append(f"Mode {name} spearman: {mode_value[0]} count={mode_count[0]}")

            results["Skewness"].append(f"Skewness {name} spearman: {stats.skew(spearman, bias=False, nan_policy='raise'):.6f}")
        file.write("\n".join("\n".join(v) for v in results.values()) + "\n")

        # check All Same Predicted Values Item_Count in {name}set # EX: 一個藥對應每個ccl時，輸出值都一樣
        for name, AllSameValuesList_count in [("Train", train_AllSameValuesList_count),
                                        ("Validation", val_AllSameValuesList_count),
                                        ("Test", test_AllSameValuesList_count)]:
            file.write(f"All Same Predicted Values Item_Count in {name}set: {AllSameValuesList_count}\n")

    file.write(f"BF_test_targets\n{BF_test_targets[0][:10]}\n")
    file.write(f"BF_test_outputs_before_final_activation_list\n{BF_test_outputs_before_final_activation_list[0][:10]}\n")
    file.write(f"BF_test_outputs\n{BF_test_outputs[0][:10]}\n")
    print("Output saved to:", output_file)
    os.chmod(output_file, 0o444)

    
if model_inference is True:
    set_seed(seed)
    if model_name == "Omics_DrugESPF_Model":
        model = Omics_DrugESPF_Model(omics_encode_dim_dict, drug_encode_dims, activation_func, activation_func_final, dense_layer_dim, device, ESPF, Drug_SelfAttention, pos_emb_type,
                            drug_embedding_feature_size, intermediate_size, num_attention_heads , attention_probs_dropout_prob, hidden_dropout_prob, omics_numfeatures_dict, max_drug_len,
                            n_layer, deconfound_EXPembedding, TCGA_pretrain_weight_path_dict= None)
    elif model_name == "Omics_DCSA_Model":
        model = Omics_DCSA_Model(omics_encode_dim_dict, drug_encode_dims, activation_func, activation_func_final, dense_layer_dim, device, ESPF, Drug_SelfAttention, pos_emb_type,
                            drug_embedding_feature_size, intermediate_size, num_attention_heads , attention_probs_dropout_prob, hidden_dropout_prob, omics_numfeatures_dict, max_drug_len,
                            n_layer, deconfound_EXPembedding, TCGA_pretrain_weight_path_dict= None)
    model.to(device=device)
    model.load_state_dict(BF_best_weight) 

    drug_list=["cisplatin", "5-fluorouracil", "gemcitabine", "sorafenib", "temozolomide"]
    drugs_metrics={}
    for drug_name in drug_list:
        if deconfound_EXPembedding is True:
            with open(f"../data/DAPL/share/pretrain/{DA_Folder}/{cohort}/{drug_name}_latent_results.pkl", 'rb') as f:
                latent_dict = pickle.load(f)
                CohortExp_df = pd.DataFrame(latent_dict).T # 32
        else:
            CohortExp_df = pd.read_csv(f"../data/DAPL/share/{cohort}_fromDAPL/{drug_name}/{cohort.lower()}data{geneNUM}.csv", sep=',', index_col=0) #1426
        label_df = pd.read_csv(f"../data/DAPL/share/{cohort}_fromDAPL/{drug_name}/{cohort.lower()}label{geneNUM}.csv", sep=',', index_col=0)
        # label_df = 1 - label_df # make label 0 to 1, 1 to 0 to match regressionpredicted output. after that 0: sensitive, 1: resistant
        CohortExp_df = CohortExp_df.sort_index(axis=0).sort_index(axis=1)
        print(f"{cohort}exp {drug_name}data",CohortExp_df.shape)
        label_df = label_df.sort_index(axis=0).sort_index(axis=1)
        print(f"label_df {drug_name}data",label_df.shape)
        for omic_type in include_omics:
            if deconfound_EXPembedding is True:
                omics_data_dict["Exp"] = CohortExp_df
            else:
                if omic_type == "Exp":
                    scaler = scaler_dict[omic_type]
                    omics_data_dict[omic_type] = pd.DataFrame(scaler.transform(CohortExp_df),index=CohortExp_df.index,columns=CohortExp_df.columns) # use fitted CCLE scaler to transform TCGA data
            omics_data_tensor_dict[omic_type]  = torch.tensor(omics_data_dict[omic_type].values, dtype=torch.float32).to(device)
            omics_numfeatures_dict[omic_type] = omics_data_tensor_dict[omic_type].shape[1]

            print(f"{omic_type} tensor shape:", omics_data_tensor_dict[omic_type].shape)
            print(f"{omic_type} num_features",omics_numfeatures_dict[omic_type])

#         drug_df_path= "../data/DAPL/share/GDSC_drug_merge_pubchem_dropNA_MACCS.csv"
        drug_df = pd.read_csv( drug_df_path, sep=',')
        # get specific drug and ccl
        drug_df = drug_df[drug_df['name'].str.lower() == drug_name.lower()]
        print(drug_df)
        if ESPF is True:
            drug_smiles =drug_df["SMILES"] # 
            print("drug_smiles",drug_smiles)
            drug_names =drug_df.index
            # 挑出重複的SMILES
            duplicate =  drug_smiles[drug_smiles.duplicated(keep=False)]
            #ESPF
            vocab_path = "./ESPF/drug_codes_chembl_freq_1500.txt" # token
            sub_csv = pd.read_csv("./ESPF/subword_units_map_chembl_freq_1500.csv")# token with frequency
            # 將drug_smiles 使用_drug2emb_encoder function編碼成subword vector
            drug_encode = pd.Series(drug_smiles).apply(drug2emb_encoder, args=(vocab_path, sub_csv, max_drug_len))
            drug_features_tensor = torch.tensor(np.array([i[:2] for i in drug_encode.values]), dtype=torch.long).to(device)
        else:
            drug_encode = drug_df["MACCS166bits"]
            drug_encode_list = [list(map(int, item.split(','))) for item in drug_encode.values]
            print("MACCS166bits_drug_encode_list type: ")# ,type(drug_encode_list))
            # Convert your data to tensors if they're in numpy
            drug_features_tensor = torch.tensor(np.array(drug_encode_list), dtype=torch.long).to(device)
        #--------------------------------------------------------------------------------------------------------------------------
        num_ccl = list(omics_data_tensor_dict.values())[0].shape[0]
        num_drug = drug_encode.shape[0]
        print("num_ccl,num_drug: ",num_ccl,num_drug)

        response_matrix_tensor = torch.tensor(label_df.values, dtype=torch.float32).to(device).unsqueeze(1)
        # print(omics_data_tensor_dict)
        print(drug_features_tensor.shape)# Fc1c[nH]c(=O)[nH]c1=O 
        print(response_matrix_tensor.shape)

        if 'weighted' in criterion.loss_type :    
            if 'BCE' in criterion.loss_type :
                weighted_threshold = None
                total_samples = (~np.isnan(AUC_df.values)).sum().item()
                fewWt_samples = (AUC_df.values == 0).sum().item()
                moreWt_samples = (AUC_df.values == 1).sum().item()
                few_weight = total_samples / (2 * fewWt_samples)  
                more_weight = total_samples / (2 * moreWt_samples) 
            else:
                # Set threshold based on the 90th percentile # 將高於threshold的AUC權重增加
                weighted_threshold = np.nanpercentile(AUC_df.values, 90)    
                total_samples = (~np.isnan(AUC_df.values)).sum().item()
                fewWt_samples = (AUC_df.values > weighted_threshold).sum().item()
                moreWt_samples = total_samples - fewWt_samples
                few_weight = total_samples / (2 * fewWt_samples)  
                more_weight = total_samples / (2 * moreWt_samples)   
        else:
            weighted_threshold = None
            few_weight = None
            more_weight = None

        set_seed(seed)
        dataset = OmicsDrugDataset(omics_data_tensor_dict, drug_features_tensor, response_matrix_tensor, splitType, include_omics)
        onedrug_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # eval_targets, eval_outputs,predAUCwithUnknownGT, AttenScorMat_DrugSelf, AttenScorMat_DrugCellSelf,eval_outputs_before_final_activation_list, mean_batch_eval_lossWOpenalty
        (eval_targets, eval_outputs,predAUCwithUnknownGT,
        AttenScorMat_DrugSelf ,AttenScorMat_DrugCellSelf,
        eval_outputs_before_final_activation_list,  
        mean_batch_eval_loss_WO_penalty)  = evaluation(model, None,None,
                                                    criterion, onedrug_loader, device,ESPF,Drug_SelfAttention, 
                                                    weighted_threshold, few_weight, more_weight, 
                                                    outputcontrol='inference')
        # Calculate classification metrics                                            
        drugs_metrics[drug_name], _  = metrics_calculator(torch.cat(eval_targets), torch.cat(eval_outputs), BF_best_prob_threshold, metric, dataset="test")
        drugs_metrics[drug_name]["eval_targets"]=eval_targets
        drugs_metrics[drug_name]["eval_outputs"]=eval_outputs
        drugs_metrics[drug_name]["eval_outputs_before_final_activation_list"]=eval_outputs_before_final_activation_list
        drugs_metrics[drug_name][criterion.loss_type] = mean_batch_eval_loss_WO_penalty

        if 'BCE' in criterion.loss_type :
            (test_cm ,  test_GT_0_count, test_GT_1_count, 
            test_pred_binary_0_count, test_pred_binary_1_count ) =metrics_calculator.confusion_matrix(torch.cat(eval_targets), torch.cat(eval_outputs), BF_best_prob_threshold)

            drugs_metrics[drug_name]["CM"] = test_cm
            # # plot confusion matrix
            cm_datas = [(test_cm, cohort, 'Blues')]
            Confusion_Matrix_plot(cm_datas,hyperparameter_folder_path=hyperparameter_folder_path,drug=drug_name)

        else:#regression use prob_threshold to get binary outcome
            df = pd.DataFrame({'predicted AUDRC': torch.cat(eval_outputs).cpu().numpy(),
                    'Label': torch.cat(eval_targets).cpu().numpy()})
            # Perform t-test between the two groups
            sensitive = df[df['Label'] == 1]['predicted AUDRC']
            resistant = df[df['Label'] == 0]['predicted AUDRC']
            t_stat, p_val = ttest_ind(sensitive, resistant)
            drugs_metrics[drug_name]["pvalue"]= p_val
            if p_val<=0.05:
                TCGA_predAUDRC_box_plot_twoClass(drug_name,cohort,df,sensitive,resistant,p_val,hyperparameter_folder_path)
        # not a reasonable way to calculate AUROC and AUPRC to explain the model performance
           
            
    
    output_file = f"{hyperparameter_folder_path}/BF{BF}_{cohort}_inference_result.txt"
    with open(output_file, "w") as file:
        if 'BCE' in criterion.loss_type :
            for drug_name, metrics in drugs_metrics.items():
                file.write(f"\n{drug_name}\n")
                file.write(f"BF_best_prob_threshold: {BF_best_prob_threshold} according to {metric}\n")
                file.write(f"  test {criterion.loss_type}loss: {metrics[criterion.loss_type].item():.6f}\n")
                for key in metrics_type_set:
                    file.write(f"  '{key}': {metrics[key].item():.4f}\n")
                for key in ["eval_targets","eval_outputs_before_final_activation_list","eval_outputs"]:
                    file.write(f"\n{key}\n{metrics[key][0][:20]}\n\n")
        else:
            for drug_name, metrics in drugs_metrics.items():
                file.write(f"{drug_name}\n")
                file.write(f"  test {criterion.loss_type}loss: {metrics[criterion.loss_type].item():.6f}\n")
                if metrics['pvalue'].item() <= 0.05:
                    file.write(f"\n pvalue <= 0.05 ")
                else:
                    file.write(f"\n pvalue > 0.05 ")
                file.write(f"{drug_name} pvalue: {metrics['pvalue'].item():.4f}\n\n")
                for key in ["eval_targets","eval_outputs_before_final_activation_list","eval_outputs"]:
                    file.write(f"\n{key}\n{metrics[key][0][:20]}\n")
                        
                    
    os.chmod(output_file, 0o444)
    del model
    torch.cuda.set_device("cuda:0")# Set the current device
    gc.collect()# Optionally, force garbage collection to release memory 
    torch.cuda.empty_cache() # Empty PyTorch cache