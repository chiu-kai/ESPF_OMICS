# config.py

# 2649gene

# PDAC
import torch.nn as nn

model_name = "Mut_Drug_Model"
splitType= 'byCCL'
AUCmultiply=1
AUCtransform = None # "-log2"
data_scaler = None # StandardScaler()/MinMaxScaler()

mut_encode_dim =[33,22,14] 
drug_encode_dim =[116,81,56] #
mut_encode_dim_save = f'(33_22_14)'
drug_encode_dim_save= f'(116_81_56)'
activation_func = nn.ReLU()  # ReLU activation function
activation_func_final = nn.Sigmoid()# ReLU() , sigmoid()
dense_layer_dim = 800
batch_size = 1000
num_epoch = 200 # for k fold CV 
patience = 20
warmup_iters = 150
Decrease_percent = 0.99
learning_rate=1e-07
criterion = nn.MSELoss() #MAELoss


# ADAE pytorch
filename     = "new_ADAE_exp"
# hyperparameters
random_seed = 42
data_transform = None # "-log2"
data_scaler = None # StandardScaler()/MinMaxScaler()
class_balancing = 18 # ccl sample * 20 
feature_encode_dim = [1500,450,100] # [500,200,50] # [1000,200,50]
feature_encode_dim_save="exp(1500_450_100)"
activat_fun = nn.ReLU()
activa_f_final = nn.Sigmoid()
classifier_dim = feature_encode_dim[2]
output_dim = 1 # positive case's logit value
criterionAE = nn.MSELoss()
criterionADV = nn.BCELoss()
ae_lr = 1e-4
adv_lr = 1e-4
dropout_rate = 0.1
lambda_val = 0.1
threshold = 0.5
num_epochs = 100
iters = 100
batch_size = 256
patience = 15
ADAEpatience =30


hyperparameter_print = f' CCLE_class_balancing ={class_balancing}\n AUCmultiply ={AUCmultiply}\n AUCtransform ={AUCtransform}\n data_scaler ={data_scaler}\n feature_encode_dim ={feature_encode_dim}\n activation_func = {activat_fun}\n activation_func_final = {activa_f_final}\n classifier_dim = {feature_encode_dim[2]}\n criterionAE = {criterionAE}\n criterionADV = {criterionADV}\n output_dim = {output_dim}\n ae_lr = {ae_lr}\n adv_lr = {adv_lr}\n dropout_rate = {dropout_rate}\n lambda_val = {lambda_val}\n threshold = {threshold}\n num_epochs = {num_epochs}\n iters ={iters}\n batch_size = {batch_size}\n patience ={patience}\n ADAEpatience ={ADAEpatience}'

hyperparameter_save = f'classBalanc{class_balancing}scaler{data_scaler}FeaEncoDim{feature_encode_dim_save}ActivF{activat_fun}ActivFFinl{activa_f_final}ClassifierDim{feature_encode_dim[2]}AE-LR{ae_lr}ADV-LR{adv_lr}DropOR{dropout_rate}lambda{lambda_val}NumEpo{num_epochs}iter{iters}BatchSiz{batch_size}patien{patience}ADAEpatien{ADAEpatience}'


# ADAE tf2
#hyperparameters
seed = 42
data_transform = None # "-log2"
data_scaler = None # StandardScaler()/MinMaxScaler()
class_balancing = 19
feature_encode_dim = [500,250,100]
feature_encode_dim_save = 'mut(500_250_100)'
activation_func = 'relu'
activation_func_final = 'sigmoid'
AE_loss = 'mse'
ADV_loss = 'binary_crossentropy'
optimizer='adam'
lambda_val = 0.1
num_epochs = 200

hyperparameter_save = f'CCLclasBalan{class_balancing}scaler{data_scaler}FeaEncoDim{feature_encode_dim_save}activfun{activation_func}activfunfinl{activation_func_final}LambdaVal{lambda_val}NumEpochs{num_epochs}'



#hyperparameter
model_name = "Mut_Drug_Model"
splitType= 'byCCL'
valueMultiply=1

mut_encode_dim =[33,22,14] 
drug_encode_dim =[116,81,56] #
mut_encode_dim_save = f'(33_22_14)'
drug_encode_dim_save= f'(116_81_56)'
activation_func = nn.ReLU()  # ReLU activation function
activation_func_final = nn.Sigmoid()# sigmoid()
dense_layer_dim = 200 # MLPDim
batch_size = 1000
num_epoch = 200 # for k fold CV 
patience = 20
warmup_iters = 150
Decrease_percent = 0.99
continuous = True
learning_rate=1e-05
criterion = nn.MSELoss()
# loss need to be devided with valueMultiply^2 if lossfunction is MSE, if is L1 than devide with valueMultiply
# NNoutput need to be multiplied  with valueMultiply if activatefunction is sigmoid(), if ReLU() don't multiply

hyperparameter_print = f' model_name ={model_name}\n splitType ={splitType}\n valueMultiply ={valueMultiply}\n mut_encode_dim ={mut_encode_dim}\n drug_encode_dim ={drug_encode_dim}\n activation_func = {activation_func}\n activation_func_final = {activation_func_final}\n dense_layer_dim = {dense_layer_dim}\n batch_size = {batch_size}\n num_epoch = {num_epoch}\n patience = {patience}\n warmup_iters = {warmup_iters}\n Decrease_percent = {Decrease_percent}\n continuous ={continuous}\n learning_rate = {learning_rate}\n criterion ={criterion}'
hyperparametersave = f'data*{valueMultiply}_{splitType}_Mut{mut_encode_dim_save}Drug{drug_encode_dim_save}ActFun{activation_func}ActFunFinl{activation_func_final}MLPDim{dense_layer_dim}BatcSiz{batch_size}Epoc{num_epoch}Patien{patience}warmup{warmup_iters}Decre%{Decrease_percent}LR{learning_rate}{criterion}'
hyperparameters = {
    "model_name": model_name,
    "splitType": splitType,
    "valueMultiply": valueMultiply,
    "mut_encode_dim": mut_encode_dim,
    "drug_encode_dim": drug_encode_dim,
    "activation_func": activation_func,  # ReLU activation function
    "activation_func_final": activation_func_final,  # Sigmoid activation function
    "dense_layer_dim": dense_layer_dim,
    "batch_size": batch_size,
    "num_epoch": num_epoch,  # for k fold CV
    "patience": patience,
    "warmup_iters": warmup_iters,
    "Decrease_percent": Decrease_percent,
    "continuous": continuous,
    "learning_rate": learning_rate,
    "criterion": criterion,
    "hyperparameter_print": hyperparameter_print,
    "hyperparametersave": hyperparametersave
}