#main_kfold_instance_inference.py
# pip install subword-nmt seaborn lifelines openpyxl matplotlib scikit-learn openTSNE
# pip install torchmetrics==1.2.0 pandas==2.1.4 numpy==1.26.4
# python3 ./main_kfold_instance_inference.py --config utils/config_GDSC.py
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import  DataLoader
import torch.nn.init as init
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import copy
import gc
import os
import importlib.util
import pickle
from scipy.stats import ttest_ind
import time

from utils.ESPF_drug2emb import drug2emb_encoder
from utils.Model import Omics_ESPF_Model, OmicsESPF_DCSA_Model, GIN_DCSA_model
from utils.create_dataloader import OmicsDrugDataset,InstanceResponseDataset
from utils.train import train, evaluation
from utils.correlation import correlation_func
from utils.plot import barplot_perdrug_performance, Inference_Probability_Distribution, Confusion_Matrix_plot, TCGA_predAUDRC_box_plot_twoClass
from utils.tools import set_seed
print("*"*100)

# 設定命令列引數
parser = argparse.ArgumentParser(description="import config to main")
parser.add_argument("--config", required=True, help="Path to the config.py file")
parser.add_argument("--path", type=str, required=False, help="best_weight_path")
parser.add_argument("--threshold", type=str, required=False, help="best_prob_threshold")
parser.add_argument("--BF", type=str, required=False, help="best_fold")

args = parser.parse_args()
# 動態載入 config.py
spec = importlib.util.spec_from_file_location("config", args.config)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
# 將 config 模組中的變數導入當前命名空間
for key, value in vars(config).items():
    if not key.startswith("_"):  # 過濾內部變數，例如 __builtins__
        globals()[key] = value
        
if args.path is not None:
    best_weight_path = f'./results/{args.path}/'
    best_prob_threshold = float(args.threshold)
    BF = int(args.BF)
else:
    print("Skipping argument, using config.")

# information
struct_time   = time.localtime()
timestamp    = time.strftime("%Y-%m%d-%H%M", struct_time)

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Training on device {device}.")

# 檢查exp和AUC的samples是否一致
if DA_Folder != 'None':
    with open(omics_files['Exp'], 'rb') as f:
        latent_dict = pickle.load(f)
        exp_df = pd.DataFrame(latent_dict).T
else:
    exp_df = pd.read_csv(omics_files["Exp"], sep=',', index_col=0)
exp_df = exp_df.sort_index(axis=0).sort_index(axis=1)
AUC_df_numerical = pd.read_csv(AUC_df_path_numerical, sep=',', index_col=0)
AUC_df_numerical = AUC_df_numerical.sort_values(by='drug_name').sort_values(by='ModelID')
print(f"exp_df samples: {len(exp_df.index)} , AUC_df_numerical samples: {len(AUC_df_numerical.index)}")
matched_samples = sorted(set(AUC_df_numerical['ModelID']) & set(exp_df.index))
print("len(matched_samples)",len(matched_samples))
# 讀取omics資料
set_seed(seed)

scaler_dict = {}  # To store scalers for each omic_type
for omic_type in include_omics:
    if DA_Folder != 'None':
        omics_data_dict[omic_type] = exp_df.loc[matched_samples]
    else:
        omics_data_dict[omic_type] = pd.read_csv(omics_files[omic_type], sep=',', index_col=0).loc[matched_samples]
        omics_data_dict[omic_type] = omics_data_dict[omic_type].sort_index(axis=0).sort_index(axis=1)
        if omic_type == "Exp":# apply Column-wise Standardization 
            scaler = StandardScaler() 
            omics_data_dict[omic_type] = pd.DataFrame(scaler.fit_transform(omics_data_dict[omic_type]),index=omics_data_dict[omic_type].index,columns=omics_data_dict[omic_type].columns)
            scaler_dict[omic_type] = scaler  # save the fitted scaler for latter inference
        
    # omics_data_tensor_dict[omic_type]  = torch.tensor(omics_data_dict[omic_type].values, dtype=torch.float32).to(device)
    omics_numfeatures_dict[omic_type] = omics_data_dict[omic_type].shape[1]
    # print(f"{omic_type} tensor shape:", omics_data_tensor_dict[omic_type].shape)
    print(f"{omic_type} num_features",omics_numfeatures_dict[omic_type])

if drug_pretrain_freeze_emb_pth is not None:
    with open(drug_pretrain_freeze_emb_pth, 'rb') as f:
        drug_df = pd.DataFrame(pickle.load(f)).sort_index(axis=0).sort_index(axis=1)
    drug_df = drug_df.set_index("drug_name", drop=True)
else:
    drug_df = pd.read_csv( drug_df_path, sep=',', index_col=0)

if one_drug is not None:
    drug_df = drug_df[drug_df['name'].str.lower() == one_drug.lower()]
drug_df = drug_df.sort_index(axis=0).sort_index(axis=1)
if "BRD_ID" in drug_df.columns:
    drug_df["BRD_ID"] = drug_df["BRD_ID"].replace({"BRD-K61250484-001-02-3": "BRD-6125",
                                                    "BRD-K91701654-001-03-1 (CID5354033)": "BRD-K91701654-001-03-1",
                                                    "BRD-K18787491-001-08-6 (CID3006531)": "BRD-K18787491-001-08-6"})
print("drug_df",drug_df.shape)
if one_drug is not None:
    AUC_df_numerical = AUC_df_numerical[AUC_df_numerical['drug_name'].str.lower() == one_drug.lower()]# matched AUCfile and drug samples

# matched AUCfile and omics_data samples
AUC_df_numerical = AUC_df_numerical[AUC_df_numerical['ModelID'].isin(matched_samples)]
print("AUC_df_numerical match samples",AUC_df_numerical.shape)
# median_value = np.nanmedian(AUC_df_numerical.values)  # Directly calculate median, ignoring NaNs
# print("median_value",median_value)    
if 'BCE' in criterion.loss_type :
    AUC_df = AUC_df_numerical.copy()
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
    drug_df=drug_df[:100]
    AUC_df = AUC_df[AUC_df['drug_name'].isin(drug_df.index)]
    print("drug_df",drug_df.shape)
    print("AUC_df",AUC_df.shape)

if 'weighted' in criterion.loss_type :    
    if 'BCE' in criterion.loss_type :
        weighted_threshold = None
        total_samples = (~np.isnan(AUC_df["Label"])).sum().item()
        fewWt_samples = (AUC_df["Label"] == 0).sum().item()
        moreWt_samples = (AUC_df["Label"] == 1).sum().item()
        few_weight = total_samples / (2 * fewWt_samples)  
        more_weight = total_samples / (2 * moreWt_samples)
    else:
        # Set threshold based on the 90th percentile # 將高於threshold的AUC權重增加
        weighted_threshold = np.nanpercentile(AUC_df[response], 90)    
        total_samples = (~np.isnan(AUC_df[response])).sum().item()
        fewWt_samples = (AUC_df[response] > weighted_threshold).sum().item()
        moreWt_samples = total_samples - fewWt_samples
        few_weight = total_samples / (2 * fewWt_samples)  
        more_weight = total_samples / (2 * moreWt_samples)  
else:
    weighted_threshold = None
    few_weight = None
    more_weight = None


# convert SMILES to subword token by ESPF
if ESPF is True:
    # 挑出重複的SMILES
    duplicate =  drug_df["SMILES"][drug_df["SMILES"].duplicated(keep=False)]
    vocab_path = "./ESPF/drug_codes_chembl_freq_1500.txt" # token
    sub_csv = pd.read_csv(ESPF_file)# token with frequency
    drug_df["drug_encode"] = pd.Series(drug_df["SMILES"]).apply(drug2emb_encoder, args=(vocab_path, sub_csv, max_drug_len))
    print("drug_encode",type(drug_df["drug_encode"]))
    drug_df["drug_encode"] = [i[:2] for i in drug_df["drug_encode"].values]
    # drug_features_tensor = torch.tensor(np.array([i[:2] for i in drug_encode.values]), dtype=torch.long).to(device)#drug_features_tensor = torch.tensor(np.array(drug_encode.values.tolist()), dtype=torch.long).to(device)
elif ESPF is False and model_name == "Omics_ESPF_Model"or model_name == "GIN_DCSA_model": # 直接用MACCS166bits當drug feature  
    drug_df["drug_encode"]=[list(map(int, item.split(','))) for item in drug_df["MACCS166bits"].values]
    # drug_features_tensor = torch.tensor(np.array(drug_encode_list), dtype=torch.long).to(device)
elif drug_pretrain_freeze_emb is not None:
    drug_df["drug_encode"] = drug_df[drug_pretrain_freeze_emb] 
else:
    pass
#--------------------------------------------------------------------------------------------------------------------------
num_ccl = list(omics_data_dict.values())[0].shape[0]
num_drug = drug_df["drug_encode"].shape[0]
print("num_ccl,num_drug: ",num_ccl,num_drug)

# Convert your data to tensors if they're in numpy
# AUC_df = AUC_df.apply(pd.to_numeric, errors='coerce')# Ensure all values are numeric, coercing non-numeric ones to NaN
# response_matrix_tensor = torch.tensor(AUC_df.values, dtype=torch.float32).to(device)
# print(response_matrix_tensor.shape)

if splitType == 'whole':
    train_val_df, test_df = train_test_split(AUC_df, test_size=0.1, random_state=42,stratify=AUC_df['Label'])

else:
    # splitType = ModelID or drug_name
    all_samples = AUC_df[splitType].unique()
    train_val_samples, test_samples = train_test_split(all_samples, test_size=0.1, random_state=42)
    train_val_df = AUC_df[AUC_df[splitType].isin(train_val_samples)]
    test_df = AUC_df[AUC_df[splitType].isin(test_samples)].sort_values(by='lnIC50')

#create dataset
set_seed(seed)
def collate_fn(batch):
        gene_feature, drug_list, target = zip(*batch)
        return list(gene_feature), list(drug_list), list(target)
test_dataset = InstanceResponseDataset(test_df, omics_data_dict, drug_df, drug_graph, drug_pretrain_freeze_emb, include_omics, device)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) #, num_workers=4, pin_memory=True


if model_name == "Omics_ESPF_Model":
    model = Omics_ESPF_Model(omics_encode_dim_dict, drug_encode_dims, activation_func, activation_func_final, dense_layer_dim, device, ESPF, Drug_SelfAttention, pos_emb_type,
                        drug_emb_dim, intermediate_size, num_attention_heads , attention_probs_dropout_prob, hidden_dropout_prob, omics_numfeatures_dict, max_drug_len,
                        n_layer,DA_Folder,TCGA_pretrain_weight_path_dict= TCGA_pretrain_weight_path_dict)
elif model_name == "OmicsESPF_DCSA_Model":
    model = OmicsESPF_DCSA_Model(omics_encode_dim_dict, drug_encode_dims, activation_func, activation_func_final, dense_layer_dim, device, ESPF, Drug_SelfAttention, pos_emb_type,
                        drug_emb_dim, intermediate_size, num_attention_heads , attention_probs_dropout_prob, hidden_dropout_prob, omics_numfeatures_dict, max_drug_len,
                        n_layer,DA_Folder,TCGA_pretrain_weight_path_dict= TCGA_pretrain_weight_path_dict)
elif model_name == "GIN_DCSA_model":
    model = GIN_DCSA_model(omics_encode_dim_dict, activation_func,activation_func_final,dense_layer_dim, device,
                        drug_emb_dim, intermediate_size, num_attention_heads , attention_probs_dropout_prob, hidden_dropout_prob, omics_numfeatures_dict, 
                        n_layer, DA_Folder,TCGA_pretrain_weight_path_dict=TCGA_pretrain_weight_path_dict)

model.to(device=device)

best_weight = best_weight_path + "BestValWeight.pt"

# load Drug_Cell_SelfAttention to Drug_Cell_SelfAttention.layers.0
def rename_keys_for_layer0(state_dict):
    new_state_dict = {}
    for key in state_dict:
        if key.startswith("Drug_SelfAttention."):
            # Insert 'layers.0.' after 'Drug_SelfAttention.'
            new_key = key.replace("Drug_SelfAttention.", "Drug_SelfAttention.layers.0.")
            new_state_dict[new_key] = state_dict[key]
        elif key.startswith("Drug_Cell_SelfAttention."):
            # Insert 'layers.0.' after 'Drug_Cell_SelfAttention.'
            new_key = key.replace("Drug_Cell_SelfAttention.", "Drug_Cell_SelfAttention.layers.0.")
            new_state_dict[new_key] = state_dict[key]
        elif key.startswith("TransformerEncoder."):
            # Insert 'layers.0.' after 'TransformerEncoder.'
            new_key = key.replace("TransformerEncoder.", "TransformerEncoder.layers.0.")
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict
best_weight = torch.load(best_weight)
model.load_state_dict(best_weight)
(eval_targets, eval_outputs,predAUCwithUnknownGT, 
AttenScorMat_DrugSelf,AttenScorMat_DrugCellSelf,
eval_outputs_before_final_activation_list,
 mean_batch_eval_loss_WO_penalty)= evaluation(model, None,None,
                                             criterion, test_loader, device,ESPF,Drug_SelfAttention, 
                                             weighted_threshold, few_weight, more_weight, 
                                             outputcontrol='inference')

(test_cm ,  test_GT_0_count, test_GT_1_count, 
test_pred_binary_0_count, test_pred_binary_1_count ) =metrics_calculator.confusion_matrix(torch.cat(eval_targets), torch.cat(eval_outputs),best_prob_threshold )

# plot confusion matrix
cm_datas = [ (test_cm, 'Blues')]
Confusion_Matrix_plot(cm_datas,hyperparameter_folder_path=best_weight_path,datasetName='GDSC Testset')    

    
    
if model_inference is True:
    set_seed(seed)
    if model_name == "Omics_ESPF_Model":
        model = Omics_ESPF_Model(omics_encode_dim_dict, drug_encode_dims, activation_func, activation_func_final, dense_layer_dim, device, ESPF, Drug_SelfAttention, pos_emb_type,
                            drug_emb_dim, intermediate_size, num_attention_heads , attention_probs_dropout_prob, hidden_dropout_prob, omics_numfeatures_dict, max_drug_len,
                            n_layer, DA_Folder, TCGA_pretrain_weight_path_dict= None)
    elif model_name == "OmicsESPF_DCSA_Model":
        model = OmicsESPF_DCSA_Model(omics_encode_dim_dict, drug_encode_dims, activation_func, activation_func_final, dense_layer_dim, device, ESPF, Drug_SelfAttention, pos_emb_type,
                            drug_emb_dim, intermediate_size, num_attention_heads , attention_probs_dropout_prob, hidden_dropout_prob, omics_numfeatures_dict, max_drug_len,
                            n_layer, DA_Folder, TCGA_pretrain_weight_path_dict= None)
    elif model_name == "GIN_DCSA_model":
        model = GIN_DCSA_model(omics_encode_dim_dict, activation_func,activation_func_final,dense_layer_dim, device,
                            drug_emb_dim, intermediate_size, num_attention_heads , attention_probs_dropout_prob, hidden_dropout_prob, omics_numfeatures_dict, 
                            n_layer, DA_Folder,TCGA_pretrain_weight_path_dict=None)

    model.to(device=device)
    model.load_state_dict(best_weight) 

def get_unique_filename(path):
    base, ext = os.path.splitext(path)
    counter = 1
    new_path = path
    while os.path.exists(new_path):
        new_path = f"{base}({counter}){ext}"
        counter += 1
    return new_path


for datasetName in datasetName_lst:

    label_df_pth = infer_paths[datasetName]['label']
    EXP_pth = infer_paths[datasetName]['exp'] 
    DA_EXP_pth = infer_paths[datasetName].get('DA', 'None') # 如果沒有DA路徑，則設為'None'
    
    if DA_Folder != 'None' and DA_EXP_pth == 'None':
        print(f"Skipping {datasetName}: no DA path available.")# 若 DA_Folder 不為 'None' 但該 dataset 沒有 DA 路徑，則跳過
        continue
    if DA_Folder != 'None':
        CohortExp_df = pd.read_csv(DA_EXP_pth, sep=',', index_col=0)
    else:
        CohortExp_df = pd.read_csv(EXP_pth, sep=',', index_col=0) #1426
        
    label_df = pd.read_csv(label_df_pth, sep=',')
    label_df.rename(columns={label_df.columns[0]: "ModelID"}, inplace=True)
    label_df['drug_name'] = label_df['drug_name'].str.lower() # match the drug name in drug_df
    CohortExp_df = CohortExp_df.sort_index(axis=0).sort_index(axis=1)
    print(f"{datasetName} exp data",CohortExp_df.shape)
    label_df = label_df.sort_index(axis=0).sort_index(axis=1)
    print(f"{datasetName} label_df data",label_df.shape)

    for omic_type in include_omics:
        if DA_Folder != 'None':
            omics_data_dict["Exp"] = CohortExp_df
        else:
            if omic_type == "Exp":
                scaler = scaler_dict[omic_type]
                omics_data_dict[omic_type] = pd.DataFrame(scaler.transform(CohortExp_df),index=CohortExp_df.index,columns=CohortExp_df.columns) # use fitted CCLE scaler to transform TCGA data
        # omics_data_tensor_dict[omic_type]  = torch.tensor(omics_data_dict[omic_type].values, dtype=torch.float32).to(device)
        omics_numfeatures_dict[omic_type] = omics_data_dict[omic_type].shape[1]
        # print(f"{omic_type} tensor shape:", omics_data_tensor_dict[omic_type].shape)
        print(f"{omic_type} num_features",omics_numfeatures_dict[omic_type])

    if drug_pretrain_freeze_emb_pth is not None:
        with open(drug_pretrain_freeze_emb_pth, 'rb') as f:
            drug_df = pd.DataFrame(pickle.load(f)).sort_index(axis=0).sort_index(axis=1)
        drug_df = drug_df.set_index("drug_name", drop=True)
    else:
        drug_df = pd.read_csv( drug_df_path, sep=',') # "../data/GDSC/GDSC_drug_merge_pubchem_dropNA_MACCS.csv"
        drug_df['name'] = drug_df['name'].str.lower()
        drug_df = drug_df.set_index('name', drop=False)
    print(drug_df.shape)            
        
    if ESPF is True:
        # 挑出重複的SMILES
        duplicate =  drug_df["SMILES"][drug_df["SMILES"].duplicated(keep=False)]
        vocab_path = "./ESPF/drug_codes_chembl_freq_1500.txt" # token
        sub_csv = pd.read_csv(ESPF_file)# token with frequency
        drug_df["drug_encode"] = pd.Series(drug_df["SMILES"]).apply(drug2emb_encoder, args=(vocab_path, sub_csv, max_drug_len))
        drug_df["drug_encode"] = [i[:2] for i in drug_df["drug_encode"].values]
    elif ESPF is False and model_name == "Omics_ESPF_Model"or model_name == "GIN_DCSA_model": # 直接用MACCS166bits當drug feature  
        drug_df["drug_encode"]=[list(map(int, item.split(','))) for item in drug_df["MACCS166bits"].values]
    elif drug_pretrain_freeze_emb is not None:
        drug_df["drug_encode"] = drug_df[drug_pretrain_freeze_emb] 
    else:
        pass
    #--------------------------------------------------------------------------------------------------------------------------
    num_ccl = list(omics_data_dict.values())[0].shape[0]
    num_drug = drug_df["drug_encode"].shape[0]
    print("num_ccl,num_drug: ",num_ccl,num_drug)
# Fc1c[nH]c(=O)[nH]c1=O 
    set_seed(seed)
    dataset = InstanceResponseDataset(label_df, omics_data_dict, drug_df, drug_graph, drug_pretrain_freeze_emb, include_omics, device)
    whole_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # eval_targets, eval_outputs,predAUCwithUnknownGT, AttenScorMat_DrugSelf, AttenScorMat_DrugCellSelf,eval_outputs_before_final_activation_list, mean_batch_eval_lossWOpenalty
    (eval_targets, eval_outputs,predAUCwithUnknownGT,
    AttenScorMat_DrugSelf ,AttenScorMat_DrugCellSelf,
    eval_outputs_before_final_activation_list,
    mean_batch_eval_loss_WO_penalty)  = evaluation(model, None,None,
                                                criterion, whole_loader, device,ESPF,Drug_SelfAttention, 
                                                weighted_threshold, few_weight, more_weight, 
                                                outputcontrol='inference')
    # Calculate classification metrics  
    dataset_metrics={}
    
    dataset_metrics[datasetName], _  = metrics_calculator(torch.cat(eval_targets), torch.cat(eval_outputs), best_prob_threshold, metric, dataset=datasetName)
    dataset_metrics[datasetName]["eval_targets"]=eval_targets
    dataset_metrics[datasetName]["eval_outputs"]=eval_outputs
    dataset_metrics[datasetName]["eval_outputs_before_final_activation_list"]=eval_outputs_before_final_activation_list
    dataset_metrics[datasetName][criterion.loss_type] = mean_batch_eval_loss_WO_penalty

    if 'BCE' in criterion.loss_type :
        (test_cm ,  test_GT_0_count, test_GT_1_count, 
        test_pred_binary_0_count, test_pred_binary_1_count ) =metrics_calculator.confusion_matrix(torch.cat(eval_targets), torch.cat(eval_outputs), best_prob_threshold)
        dataset_metrics[datasetName]["CM"] = test_cm
        # # plot confusion matrix
        cm_datas = [(test_cm, 'Blues')]
        Confusion_Matrix_plot(cm_datas,hyperparameter_folder_path=best_weight_path,datasetName=datasetName)
#         tSNE_embed_plot(tSNE_embed_list, eval_targets, hyperparameter_folder_path=best_weight_path, datasetName=datasetName)

    else:#regression use prob_threshold to get binary outcome
        df = pd.DataFrame({'predicted AUDRC': torch.cat(eval_outputs).cpu().numpy(),
                'Label': torch.cat(eval_targets).cpu().numpy()})
        # Perform t-test between the two groups
        sensitive = df[df['Label'] == 1]['predicted AUDRC']
        resistant = df[df['Label'] == 0]['predicted AUDRC']
        t_stat, p_val = ttest_ind(sensitive, resistant)
        dataset_metrics[datasetName]["pvalue"]= p_val
        if p_val<=0.05:
            TCGA_predAUDRC_box_plot_twoClass(datasetName, df, sensitive, resistant, p_val, hyperparameter_folder_path=best_weight_path)    
    output_file = f"{best_weight_path}/{datasetName}_inference_result.txt"
    output_file = get_unique_filename(output_file)
    with open(output_file, "w") as file:
        if 'BCE' in criterion.loss_type :
            for datasetName, metrics in dataset_metrics.items():
                file.write(f"\n{datasetName}\n")
                file.write(f"BF_best_prob_threshold: {best_prob_threshold} according to {metric}\n")
                file.write(f"  test {criterion.loss_type}loss: {metrics[criterion.loss_type].item():.6f}\n")
                for key in metrics_type_set:
                    file.write(f"  '{key}': {metrics[key].item():.4f}\n")
                for key in ["eval_targets","eval_outputs_before_final_activation_list","eval_outputs"]:
                    file.write(f"\n{key}\n{metrics[key][0][:20]}\n\n")
        else:
            for datasetName, metrics in dataset_metrics.items():
                file.write(f"{datasetName}\n")
                file.write(f"  test {criterion.loss_type}loss: {metrics[criterion.loss_type].item():.6f}\n")
                if metrics['pvalue'].item() <= 0.05:
                    file.write(f"\n pvalue <= 0.05 ")
                else:
                    file.write(f"\n pvalue > 0.05 ")
                file.write(f"{datasetName} pvalue: {metrics['pvalue'].item():.4f}\n\n")
                for key in ["eval_targets","eval_outputs_before_final_activation_list","eval_outputs"]:
                    file.write(f"\n{key}\n{metrics[key][0][:20]}\n")       
        os.chmod(output_file, 0o444)# Read-only
        
    # TCGA per drug performance     
    label_df['predict_value'] = np.concatenate(predAUCwithUnknownGT)
    label_df["predict_label"] = (label_df["predict_value"] > float(best_prob_threshold)).astype(int)

    # if datasetName == 'TCGA_DeepCDR':
    #     cancerType_ls=['CESC']
    #     label_df["primary_disease"]='CESC'
    # elif datasetName == 'TCGA_DAPL':
    #     cancerType_ls=label_df["primary_disease"].unique().tolist()
    # elif datasetName == 'TCGA_DiSyn':
    #     cancerType_ls=label_df["cancers"].unique().tolist()
    #     label_df.rename(columns={'cancers': 'primary_disease'}, inplace=True)
    TP_df = label_df[(label_df["Label"] == 1) & (label_df["predict_label"] == 1)]
    TN_df = label_df[(label_df["Label"] == 0) & (label_df["predict_label"] == 0)]
    FP_df = label_df[(label_df["Label"] == 0) & (label_df["predict_label"] == 1)]
    FN_df = label_df[(label_df["Label"] == 1) & (label_df["predict_label"] == 0)]
    def count_by_drug(df, name):
        return (df.groupby("drug_name").size().rename(name))
    def count_by_cancerType(df, name):
        return (df.groupby("primary_disease").size().rename(name))
    drug_confusion = ( count_by_drug(TP_df, "TP").to_frame()
                .join(count_by_drug(TN_df, "TN"), how="outer")
                .join(count_by_drug(FP_df, "FP"), how="outer")
                .join(count_by_drug(FN_df, "FN"), how="outer")
                .fillna(0) .astype(int))
    # cancerType_confusion = ( count_by_cancerType(TP_df, "TP").to_frame()
    #                 .join(count_by_cancerType(TN_df, "TN"), how="outer")
    #                 .join(count_by_cancerType(FP_df, "FP"), how="outer")
    #                 .join(count_by_cancerType(FN_df, "FN"), how="outer")
    #                 .fillna(0) .astype(int))

    # 計算dataset各個藥的metrics
    drugs_metrics={}
    dataset_drugs = drug_confusion.index.tolist()#取得 dataset drug list
    for drug_name in dataset_drugs:
        drugs_metrics[drug_name], _  = metrics_calculator(torch.tensor(label_df[label_df["drug_name"] == drug_name]['Label'].values), 
                                                        torch.tensor(label_df[label_df["drug_name"] == drug_name]['predict_value'].values), 
                                                        best_prob_threshold, metric, dataset=datasetName)
    dataset_perform_df = drug_confusion.join(pd.DataFrame(drugs_metrics).T.map(lambda x: x.item() if hasattr(x, 'item') else x))
    dataset_perform_df.to_csv(f"{best_weight_path}/{datasetName}_perDrug_performance.csv", index=True, encoding='utf-8-sig')
    barplot_perdrug_performance(dataset_drugs, drugs_metrics, datasetName,hyperparameter_folder_path=best_weight_path)

    #plot Inference_Probability_Distribution
    Inference_Probability_Distribution( eval_outputs, eval_targets, float(best_prob_threshold), hyperparameter_folder_path=best_weight_path, datasetName=datasetName)

del model
torch.cuda.set_device("cuda:0")# Set the current device
gc.collect()# Optionally, force garbage collection to release memory 
torch.cuda.empty_cache() # Empty PyTorch cache