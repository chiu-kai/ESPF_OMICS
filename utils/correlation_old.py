import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
import numpy as np

def correlation(train_val_outputs,test_outputs,ccl_name,data_AUC,drug_name,id_train_val,id_test):
    train_val_output=np.concatenate(train_val_outputs).tolist()
    print(np.shape(train_val_output))
    test_output=np.concatenate(test_outputs).tolist()
    print(np.shape(test_output))
    #print(data_AUC.shape)
    
    ccl_name = np.array(ccl_name).tolist()
    data_AUC = np.array(data_AUC).tolist()
    drug_name = np.array(drug_name).tolist()
    
    id_train_val = id_train_val
    id_test = id_test
    
    # Create a dictionary to store collected data for each unique ccl_name
    ccl_train_val_dict = {}
    drug_train_val_dict = {}
    ccl_train_val_pearson=[]
    ccl_train_val_spearman=[]
    drug_train_val_pearson=[]
    drug_train_val_spearman=[]
    for idx,id in enumerate(id_train_val):
        ccl = ccl_name[id]
        if ccl not in ccl_train_val_dict: # ccl如果不再dict裡，就加上去
            ccl_train_val_dict[ccl] = [[],[]] #[[true],[pred]]
        ccl_train_val_dict[ccl][0].append(data_AUC[id])# ccl，就append AUC
        ccl_train_val_dict[ccl][1].append(train_val_output[idx])# ccl，就append predict value
        
        drug = drug_name[id]
        if drug not in drug_train_val_dict: # drug如果不再dict裡，就加上去
            drug_train_val_dict[drug] = [[],[]] #[[true],[pred]]
        drug_train_val_dict[drug][0].append(data_AUC[id])# drug，就append AUC
        drug_train_val_dict[drug][1].append(train_val_output[idx])# drug，就append predict value
    
    for key,value in ccl_train_val_dict.items():
        correlation_coef, p_value = pearsonr(value[0], value[1])
        correlation_coef=abs(correlation_coef)
        spearman_correlation_coef, p_value = spearmanr(value[0], value[1])
        spearman_correlation_coef = abs(spearman_correlation_coef)
        print(len(value[0]))
        print(key,":",correlation_coef)
        ccl_train_val_pearson.append(correlation_coef)
        ccl_train_val_spearman.append(spearman_correlation_coef)
    for key,value in drug_train_val_dict.items():
        correlation_coef, p_value = pearsonr(value[0], value[1])
        correlation_coef=abs(correlation_coef)
        spearman_correlation_coef, p_value = spearmanr(value[0], value[1])
        spearman_correlation_coef = abs(spearman_correlation_coef)
        #print(len(value[0]))
        #print(key,":",correlation_coef)
        drug_train_val_pearson.append(correlation_coef)
        drug_train_val_spearman.append(spearman_correlation_coef)
    
    print("\n")
    print("test set"+"="*20)
    print("test set"+"="*20)
    
    
    ccl_test_dict = {}
    drug_test_dict = {}
    ccl_test_pearson=[]
    ccl_test_spearman=[]
    drug_test_pearson=[]
    drug_test_spearman=[]
    for idx,id in enumerate(id_test):
        ccl = ccl_name[id]
        if ccl not in ccl_test_dict: # ccl如果不再dict裡，就加上去
            ccl_test_dict[ccl] = [[],[]] #[[true],[pred]]
        ccl_test_dict[ccl][0].append(data_AUC[id])# ccl，就append AUC
        ccl_test_dict[ccl][1].append(test_output[idx])# ccl，就append predict value
        
        drug = drug_name[id]
        if drug not in drug_test_dict: # drug如果不再dict裡，就加上去
            drug_test_dict[drug] = [[],[]] #[[true],[pred]]
        drug_test_dict[drug][0].append(data_AUC[id])# drug，就append AUC
        drug_test_dict[drug][1].append(test_output[idx])# drug，就append predict value
    
    for key,value in ccl_test_dict.items():
        correlation_coef, p_value = pearsonr(value[0], value[1])
        correlation_coef=abs(correlation_coef)
        spearman_correlation_coef, p_value = spearmanr(value[0], value[1])
        spearman_correlation_coef = abs(spearman_correlation_coef)
        print(len(value[0]))
        print(key,":",correlation_coef)
        ccl_test_pearson.append(correlation_coef)
        ccl_test_spearman.append(spearman_correlation_coef)
    for key,value in drug_test_dict.items():
        correlation_coef, p_value = pearsonr(value[0], value[1])
        correlation_coef=abs(correlation_coef)
        spearman_correlation_coef, p_value = spearmanr(value[0], value[1])
        spearman_correlation_coef = abs(spearman_correlation_coef)
        #print(len(value[0]))
        #print(key,":",correlation_coef)
        drug_test_pearson.append(correlation_coef)
        drug_test_spearman.append(spearman_correlation_coef)