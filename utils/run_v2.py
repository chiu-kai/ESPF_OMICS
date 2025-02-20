# run_v2.py
# inputfiles 可變動

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.init as init
from sklearn.model_selection import KFold
import copy
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from utils.load_data import load_ccl, load_AUC_matrix, load_drug
from utils.data_repeat_to_instance import repeat_data
from utils.Mut_Drug_Model import Mut_Drug_Model
from utils.split_data_id import split_id
from utils.create_dataloader import create_dataset
from utils.train import train, evaluation
from utils.correlation import correlation_func
from utils.plot import loss_curve, correlation_density
from utils.tools import get_data_value_range


def run_v2(*args,hyperparameters,filename):
    # Unpack the hyperparameters dictionary
    model_name = hyperparameters["model_name"]
    splitType = hyperparameters["splitType"]
    valueMultiply = hyperparameters["valueMultiply"]
    mut_encode_dim = hyperparameters["mut_encode_dim"]
    drug_encode_dim = hyperparameters["drug_encode_dim"]
    mut_encode_dim_save = hyperparameters["mut_encode_dim_save"]
    drug_encode_dim_save = hyperparameters["drug_encode_dim_save"]
    activation_func = hyperparameters["activation_func"]
    activation_func_final = hyperparameters["activation_func_final"]
    dense_layer_dim = hyperparameters["dense_layer_dim"]
    batch_size = hyperparameters["batch_size"]
    num_epoch = hyperparameters["num_epoch"]
    patience = hyperparameters["patience"]
    warmup_iters = hyperparameters["warmup_iters"]
    Decrease_percent = hyperparameters["Decrease_percent"]
    continuous = hyperparameters["continuous"]
    learning_rate = hyperparameters["learning_rate"]
    criterion = hyperparameters["criterion"]
    hyperparameter_print = hyperparameters["hyperparameter_print"]
    hyperparametersave = hyperparameters["hyperparametersave"]


    device = (torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))
    print(f"Training on device {device}.")

    #load data
    data_drug, drug_names  = load_drug("./MACCS(Secondary_Screen_treatment_info)_union.txt")
    data_mut, gene_names_mut,ccl_names_mut  = load_ccl("./CCLE_matched_PRISMauc_PDAC_TCGA_binary_mutation.txt")
    data_AUC_matrix, drug_names_AUC, ccl_names_AUC = load_AUC_matrix(splitType,"/root/Winnie/no_Imputation_PRISM_Repurposing_Secondary_Screen_data/Drug_sensitivity_AUC_(PRISM_Repurposing_Secondary_Screen)_subsetted.csv")# splitType = "byCCCL" or "byDrug" 決定AUCmatrix要不要轉置
    print("\n\nDatasets successfully loaded.")
    print((np.shape(data_drug)))
    print((np.shape(data_mut)))
    print((np.shape(data_AUC_matrix)))
    print(np.shape(data_AUC_matrix))

    num_ccl = data_mut.shape[0]
    num_drug = data_drug.shape[0]

    # data repeat to instance
    # first change data type to torch.float32 and transfer to GPU, then repeat_to_instance byCCL/byDrug
    data_mut, data_drug, data_AUC = repeat_data(data_mut, data_drug, data_AUC_matrix, splitType= splitType, num_ccl=num_ccl, num_drug=num_drug, device=device) 

    data_AUC = data_AUC*valueMultiply
    data_AUC.shape

    print(data_AUC[1])
    print(f'{data_AUC[8].item():.8f}')

    # randomly split by CCL
    # 90% for training(10% for validation) and 10% for testing
    id_unrepeat_train, id_unrepeat_val, id_unrepeat_test, id_unrepeat_train_val, id_train, id_val, id_test= split_id(num_ccl,num_drug,splitType='byCCL',repeat=True,kFold=False)


    #create dataset
    seed=42
    torch.manual_seed(seed)
    train_dataset = create_dataset(data_mut, data_drug, data_AUC, id = id_train, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = create_dataset(data_mut, data_drug, data_AUC, id = id_val, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = create_dataset(data_mut, data_drug, data_AUC, id = id_test, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #train
    # Init the neural network  
    torch.manual_seed(seed)
    model = Mut_Drug_Model(mut_encode_dim, drug_encode_dim, activation_func, activation_func_final, dense_layer_dim, device,
                        num_mut_features=data_mut.shape[1], num_drug_features=data_drug.shape[1],
                        TCGA_pretrain_weight_path= "/root/Winnie/PDAC/results/Encoder_tcga_mut_33_22_14_best_loss_0.0078.pt").to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)# Initialize optimizer

    best_epoch, best_weight, best_val_loss, train_epoch_loss_list, val_epoch_loss_list,_ = train( model, activation_func_final,
        optimizer,      batch_size,      num_epoch,      patience,      warmup_iters,      Decrease_percent,    continuous,
        learning_rate,      criterion,      valueMultiply,      train_loader,      val_loader,
        device,      seed=42, kfoldCV = None)

    print("best Epoch : ",best_epoch,"best_val_loss : ",best_val_loss," batch_size : ",batch_size,
            "learning_rate : ",learning_rate," warmup_iters :" ,warmup_iters  ," with Decrease_percent : ",Decrease_percent )

    # Saving the model weughts
    save_path = f'/root/Winnie/PDAC/results/{filename}_BestValLoss_{best_val_loss}_{hyperparametersave}.pt'
    torch.save(best_weight, save_path)


    # Evaluation
    #Evaluation on best fold best split id (train, val) with best_fold_best_weight 
    train_dataset = TensorDataset(data_mut[id_train],data_drug[id_train], data_AUC[id_train])
    train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=False)

    # Evaluation on the train set
    model.load_state_dict(best_weight)  
    model = model.to(device=device)
    epoch = None
    train_loss, train_targets, train_outputs = evaluation(model, activation_func_final, epoch, num_epoch, val_epoch_loss_list, criterion, valueMultiply, train_loader, device, correlation='train', kfoldCV = None)
    # Evaluation on the validation set
    model.load_state_dict(best_weight)  
    model = model.to(device=device)
    val_loss, val_targets, val_outputs = evaluation(model, activation_func_final, epoch, num_epoch, val_epoch_loss_list, criterion, valueMultiply, val_loader, device, correlation='val', kfoldCV = None)
    # Evaluation on the test set
    model.load_state_dict(best_weight)
    model = model.to(device=device)
    test_loss, test_targets, test_outputs = evaluation(model, activation_func_final, epoch, num_epoch, val_epoch_loss_list, criterion, valueMultiply, test_loader, device, correlation='test', kfoldCV = None)


    # Correlation
    train_pearson, train_spearman = correlation_func(splitType, data_AUC_matrix,ccl_names_AUC,drug_names_AUC,id_unrepeat_train,train_targets,train_outputs)
    print("\n")
    print("val set"+"="*20)
    print("val set"+"="*20)
    val_pearson, val_spearman = correlation_func(splitType, data_AUC_matrix,ccl_names_AUC,drug_names_AUC,id_unrepeat_val,val_targets,val_outputs)
    print("\n")
    print("test set"+"="*20)
    print("test set"+"="*20)
    test_pearson, test_spearman = correlation_func(splitType, data_AUC_matrix,ccl_names_AUC,drug_names_AUC,id_unrepeat_test,test_targets,test_outputs)

    #plot correlation_density
    correlation_density(model_name,train_pearson,val_pearson,test_pearson,train_spearman,val_spearman,test_spearman)

    # plot GroundTruth AUC and predicted AUC distribution
    import seaborn as sns
    import matplotlib.pyplot as plt
    predicted_AUC = train_outputs + val_outputs + test_outputs
    predicted_AUC = np.concatenate(predicted_AUC).tolist()
    print(predicted_AUC[:10])
    print(np.array(predicted_AUC).shape)
    GroundTruth_AUC = train_targets + val_targets + test_targets
    GroundTruth_AUC = np.concatenate(GroundTruth_AUC).tolist()
    print(np.array(GroundTruth_AUC).shape)
    print(GroundTruth_AUC[:10])

    fig=plt.figure(figsize=(6, 5))
    sns.kdeplot(GroundTruth_AUC, color='red', label='GroundTruth_AUC')
    sns.kdeplot(predicted_AUC, color='blue', linestyle='dashed', label='predicted_AUC')
    # Set the x-axis label to 'Density'
    plt.xlabel('AUC values',fontsize=15)
    # Set the y-axis label to 'Pearson\'s Correlation Coefficient Value'
    plt.ylabel('Density',fontsize=15)
    # Set the title of the plot
    plt.title('Density Plot of AUC values',fontsize=15)
    plt.legend(fontsize=12)
    plt.show()
    fig.savefig('Density Plot of AUC values')

    # data range
    get_data_value_range(GroundTruth_AUC,"GroundTruth_AUC")
    get_data_value_range(predicted_AUC,"predicted_AUC")

    # print hyperparameter and result
    print(filename)
    print(hyperparameter_print)
    print('best epoch: ',best_epoch)

    print(f'Evaluation Training Loss: {train_loss:.6f}')
    print(f'Evaluation validation Loss: {val_loss:.6f}')
    print(f'Evaluation Test Loss: {test_loss:.6f}')
    
    print(f"Mean train_pearson PDAC {model_name}: ",np.mean(train_pearson))
    print(f"Median train_pearson PDAC {model_name}: ",np.median(train_pearson))
    print(f"Mode train_pearson PDAC {model_name}: ",stats.mode(np.round(train_pearson,2))[0],",count=",stats.mode(np.round(train_pearson,2))[1])

    print(f"Mean val_pearson PDAC {model_name}: ",np.mean(val_pearson))
    print(f"Median val_pearson PDAC {model_name}: ",np.median(val_pearson))
    print(f"Mode val_pearson PDAC {model_name}: ",stats.mode(np.round(val_pearson,2))[0],",count=",stats.mode(np.round(val_pearson,2))[1])

    print(f"Mean test_pearson PDAC {model_name}: ",np.mean(test_pearson))
    print(f"Median test_pearson PDAC {model_name}: ",np.median(test_pearson))
    print(f"Mode test_pearson PDAC {model_name}: ",stats.mode(np.round(test_pearson,2))[0],",count=",stats.mode(np.round(test_pearson,2))[1])

    print(f"Mean train_spearman PDAC {model_name}: ",np.mean(train_spearman))
    print(f"Median train_spearman PDAC {model_name}: ",np.median(train_spearman))
    print(f"Mode train_spearman PDAC {model_name}: ",stats.mode(np.round(train_spearman,2))[0],",count=",stats.mode(np.round(train_spearman,2))[1])

    print(f"Mean val_spearman PDAC {model_name}: ",np.mean(val_spearman))
    print(f"Median val_spearman PDAC {model_name}: ",np.median(val_spearman))
    print(f"Mode val_spearman PDAC {model_name}: ",stats.mode(np.round(val_spearman,2))[0],",count=",stats.mode(np.round(val_spearman,2))[1])

    print(f"Mean test_spearman PDAC {model_name}: ",np.mean(test_spearman))
    print(f"Median test_spearman PDAC {model_name}: ",np.median(test_spearman))
    print(f"Mode test_spearman PDAC {model_name}: ",stats.mode(np.round(test_spearman,2))[0],",count=",stats.mode(np.round(test_spearman,2))[1])

    print(f"Mean Median Mode train_pearson PDAC {model_name}: ",np.mean(train_pearson)," ",np.median(train_pearson)," ",stats.mode(np.round(train_pearson,2)))
    print(f"Mean Median Mode val_pearson PDAC {model_name}: ",np.mean(val_pearson)," ",np.median(val_pearson)," ",stats.mode(np.round(val_pearson,2)))
    print(f"Mean Median Mode test_pearson PDAC {model_name}: ",np.mean(test_pearson)," ",np.median(test_pearson)," ",stats.mode(np.round(test_pearson,2)))
    print(f"Mean Median Mode train_spearman PDAC {model_name}: ",np.mean(train_spearman)," ",np.median(train_spearman)," ",stats.mode(np.round(train_spearman,2)))
    print(f"Mean Median Mode val_spearman PDAC {model_name}: ",np.mean(val_spearman)," ",np.median(val_spearman)," ",stats.mode(np.round(val_spearman,2)))
    print(f"Mean Median Mode test_spearman PDAC {model_name}: ",np.mean(test_spearman)," ",np.median(test_spearman)," ",stats.mode(np.round(test_spearman,2)))


    return train_epoch_loss_list, val_epoch_loss_list, best_epoch, best_val_loss