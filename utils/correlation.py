# corration.py
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

#精簡版
def correlation_func(splitType, data_AUC_matrix,ccl_names_AUC,drug_names_AUC,id_unrepeat,targets,outputs):
    '''
    targets and outputs need to be 1D torch tensor
    '''    
    ans_dict = {} # GroundTruth AUC #key:ccl, value:[GT_auc]
    pred_dict = {} # Predicted AUC #key:ccl, value:[Pred_auc]
    pearson=[]
    spearman=[]
    start=0
    if splitType in ['byCCL', 'ModelID']:
        if len(data_AUC_matrix) != len(ccl_names_AUC):
            data_AUC_matrix = data_AUC_matrix.T
            print(f'transpose data_AUC_matrix to match split {splitType} ({data_AUC_matrix.shape})')
    elif splitType in ['byDrug', 'drug_name']:
        if len(data_AUC_matrix) != len(drug_names_AUC):
            data_AUC_matrix = data_AUC_matrix.T
            print(f"transpose data_AUC_matrix to match split {splitType} ({data_AUC_matrix.shape})")
    for ID in id_unrepeat : #ccl_unrepeat_id/ drug_unrepeat_id
        mask = ~np.isnan(data_AUC_matrix[ID]) # id  的有值mask
        ValueCount = (mask.reshape(-1)).tolist().count(True) # Value's Count in mask
        # ID_paired_drugName[ID]=(np.array(drug_names_AUC)[mask.tolist()].tolist())
        # ID_paired_drugCount[ID]=drugCount #ValueCount
        ans_dict[ID] = (targets)[start:start+ValueCount] # store GT values of each CCL/Drug(id)
        pred_dict[ID] = (outputs)[start:start+ValueCount]# store predicted values of each CCL/Drug(id)
        start+=ValueCount  
        
    AllSameValuesList_count = 0    
    for key,GTvalues in ans_dict.items():
        predvalues = pred_dict[key]
        # Check if predvalues is empty
        if predvalues.size == 0:
            print(f"\n    No predictions for key: {key}")
            continue  # Skip to the next iteration

        # check All Same Predicted Values Item_Count in {name}set # EX: 一個藥對應每個ccl時，輸出值都一樣
        if torch.all(predvalues == predvalues[0]):
            AllSameValuesList_count+=1 
        # 只有有一item不是都預測同一個值，那就會存到pearson list中去計算mean,modian,mode，所以mean,modian,mode可能部會是所有的item都在裡面
        else:
            if len(GTvalues)>=2: # pearsonr和spearmanr需要2個值以上，但是test=True時，drug-cell pair可能只有一個或零個
                correlation_coef, p_value = pearsonr(predvalues.cpu().numpy(), GTvalues.cpu().numpy())
                # correlation_coef=abs(correlation_coef)
                spearman_correlation_coef, p_value = spearmanr(predvalues.cpu().numpy(), GTvalues.cpu().numpy())
                # spearman_correlation_coef = abs(spearman_correlation_coef)
                if splitType in ['byCCL', 'ModelID']:
                    pearson.append(correlation_coef)
                    spearman.append(spearman_correlation_coef)     
#                     print(ccl_names_AUC[key],":",correlation_coef)
                elif splitType in ['byDrug', 'drug_name']:
                    pearson.append(correlation_coef)
                    spearman.append(spearman_correlation_coef) 
#                     print(drug_names_AUC[key],":",correlation_coef)
            else:
                if splitType in ['byCCL', 'ModelID']:
                    print(f"No Correlation: len(GTvalues)<2: {ccl_names_AUC[key]} ")
                elif splitType in ['byDrug', 'drug_name']:
                    print(f"No Correlation: len(GTvalues)<2: {drug_names_AUC[key]} ")
    return pearson, spearman, AllSameValuesList_count

# Usage:
# train_pearson, train_spearman = correlation_func(splitType, data_AUC_matrix,ccl_names_AUC,drug_names_AUC,id_unrepeat_train,train_targets,train_outputs)
# val_pearson, val_spearman = correlation_func(splitType, data_AUC_matrix,ccl_names_AUC,drug_names_AUC,id_unrepeat_val,val_targets,val_outputs)
# test_pearson, test_spearman = correlation_func(splitType, data_AUC_matrix,ccl_names_AUC,drug_names_AUC,id_unrepeat_test,test_targets,test_outputs)




#old
def corration(id_cell_train,id_cell_val,id_cell_test,data_AUC_matrix,ccl_names_AUC,train_targets,train_outputs,val_targets,val_outputs,test_targets,test_outputs):
    ans_train_dict = {} # GroundTruth AUC #key:ccl, value:[GT_auc]
    pred_train_dict = {} # Predicted AUC #key:ccl, value:[Pred_auc]
    ans_val_dict = {} # GroundTruth AUC #key:ccl, value:[GT_auc]
    pred_val_dict = {} # Predicted AUC #key:ccl, value:[Pred_auc]
    ans_test_dict = {} # GroundTruth AUC #key:ccl, value:[GT_auc]
    pred_test_dict = {} # Predicted AUC #key:ccl, value:[Pred_auc]
    train_pearson=[]
    train_spearman=[]
    val_pearson=[]
    val_spearman=[]
    test_pearson=[]
    test_spearman=[]

    train_start=0
    val_start=0
    test_start=0
    # cclID_paired_drugName={}# 儲存id ccl有值的drug name
    # cclID_paired_drugCount={}# 儲存id ccl有值的drug 數目

    for cclID in id_cell_train : #ccl_id
        mask = ~np.isnan(data_AUC_matrix[cclID]) # id ccl 的有值mask
        drugCount = (mask.reshape(-1)).tolist().count(True)
        # cclID_paired_drugName[cclID]=(np.array(drug_names_AUC)[mask.tolist()].tolist())
        # cclID_paired_drugCount[cclID]=drugCount
        
        ans_train_dict[cclID] = (np.concatenate(train_targets))[train_start:train_start+drugCount]
        pred_train_dict[cclID] = (np.concatenate(train_outputs))[train_start:train_start+drugCount]
        train_start+=drugCount  

    for key,value in ans_train_dict.items():
        cor_pred = pred_train_dict[key]
        correlation_coef, p_value = pearsonr(cor_pred, value)
        # correlation_coef=abs(correlation_coef)
        spearman_correlation_coef, p_value = spearmanr(cor_pred, value)
        # spearman_correlation_coef = abs(spearman_correlation_coef)
        print("item number in a id", len(value))
        print(ccl_names_AUC[key],":",correlation_coef)
        train_pearson.append(correlation_coef)
        train_spearman.append(spearman_correlation_coef)    
    print("\n")
    print("val set"+"="*20)
    print("val set"+"="*20)

    for cclID in id_cell_val: #ccl_id
        mask = ~np.isnan(data_AUC_matrix[cclID]) # id ccl 的有值mask
        drugCount = (mask.reshape(-1)).tolist().count(True)
        # cclID_paired_drugName[cclID]=(np.array(drug_names_AUC)[mask.tolist()].tolist())
        # cclID_paired_drugCount[cclID]=drugCount
        
        ans_val_dict[cclID] = (np.concatenate(val_targets))[val_start:val_start+drugCount]
        pred_val_dict[cclID] = (np.concatenate(val_outputs))[val_start:val_start+drugCount]
        val_start+=drugCount
    for key,value in ans_val_dict.items():
        cor_pred = pred_val_dict[key]
        correlation_coef, p_value = pearsonr(cor_pred, value)
        # correlation_coef=abs(correlation_coef)
        spearman_correlation_coef, p_value = spearmanr(cor_pred, value)
        # spearman_correlation_coef = abs(spearman_correlation_coef)
        print("item number in a id", len(value))
        print(ccl_names_AUC[key],":",correlation_coef)
        val_pearson.append(correlation_coef)
        val_spearman.append(spearman_correlation_coef)  
    print("\n")
    print("test set"+"="*20)
    print("test set"+"="*20)    
    for cclID in id_cell_test: #ccl_id
        mask = ~np.isnan(data_AUC_matrix[cclID]) # id ccl 的有值mask
        drugCount = (mask.reshape(-1)).tolist().count(True)
        # cclID_paired_drugName[cclID]=(np.array(drug_names_AUC)[mask.tolist()].tolist())
        # cclID_paired_drugCount[cclID]=drugCount
        
        ans_test_dict[cclID] = (np.concatenate(test_targets))[test_start:test_start+drugCount]
        pred_test_dict[cclID] = (np.concatenate(test_outputs))[test_start:test_start+drugCount]
        test_start+=drugCount
    for key,value in ans_test_dict.items():
        cor_pred = pred_test_dict[key]
        correlation_coef, p_value = pearsonr(cor_pred, value)
        # correlation_coef=abs(correlation_coef)
        spearman_correlation_coef, p_value = spearmanr(cor_pred, value)
        # spearman_correlation_coef = abs(spearman_correlation_coef)
        print("item number in a id", len(value))
        print(ccl_names_AUC[key],":",correlation_coef)
        test_pearson.append(correlation_coef)
        test_spearman.append(spearman_correlation_coef)  