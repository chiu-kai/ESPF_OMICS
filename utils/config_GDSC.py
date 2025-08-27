# pip install subword-nmt seaborn lifelines openpyxl matplotlib scikit-learn openTSNE
# pip install torchmetrics==1.2.0 pandas==2.1.4 numpy==1.26.4
# python3 ./main_kfold.py --config utils/config.py
import torch.nn as nn
from utils.Loss import Custom_LossFunction,Custom_Weighted_LossFunction,FocalLoss
from utils.Custom_Activation_Function import ScaledSigmoid, ReLU_clamp
from utils.Metrics import MetricsCalculator_nntorch

model_inference = True # False
if model_inference is True:
    cohort = "TCGA"
    geneNUM = "" # _4692genes tcgadata tcgalabel tcgadata_4692genes tcgalabel_4692genes

test = False #False, True: batch_size = 3, num_epoch = 2, full dataset
drug_df_path= "../data/GDSC/GDSC_drug_merge_pubchem_dropNA_MACCS.csv"
one_drug=None # gsk690693 trametinib erlotinib
ESPF_file = "./ESPF/subword_units_map_chembl_freq_1500.csv"
AUC_df_path_numerical = "../data/GDSC/lorio GDSC1 by whole balanced_high LNic50 downsample 17088 CCL552drug238cancerType18 stack.csv" # gdsc1+2_ccle_z-score　gdsc1+2_ccle_AUC
AUC_df_path = "../data/GDSC/lorio GDSC1 by whole balanced_high LNic50 downsample 17088 CCL552drug238cancerType18 stack.csv"
response_file = 'down_high'# down/up/imbalanced _ high/even/low
omics_files = {
    'Mut': "",
    'Exp': "../data/DAPL/share/ccle_uq1000_feature_sorted.csv", # "../data/CCLE/CCLE_exp_476samples_4692genes.txt",
    # Add more omics types and paths as needed
    }
omics_dict = {'Mut':0,'Exp':1,'CN':2, 'Eff':3, 'Dep':4, 'Met':5}
omics_data_dict = {}
omics_data_tensor_dict = {}
omics_numfeatures_dict = {}
omics_encode_dim_dict ={'Mut':[128,32],'Exp':[128,32],  # Dr.Chiu:exp[500,200,50]  [1000,100,50] 'Mut':[128,32],'Exp':[128,32], 'Mut':[1000,100,50],'Exp':[1000,100,50],
                        'CN':[100,50,30], 'Eff':[100,50,30], 'Dep':[100,50,30], 'Met':[100,50,30]}

TCGA_pretrain_weight_path_dict = None#{'Mut': "./results/Encoder_tcga_mut_1000_100_50_best_loss_0.0066.pt",
                                  #'Exp': "./results/Encoder_tcga_exp_128_32_best_loss_0.2182988.pt", # "./results/Encoder_tcga_exp_128_32_best_loss_0.2182988.pt", "./results/Encoder_tcga_exp_1000_100_50_best_loss_0.7.pt"
                                  # Add more omics types and paths as needed
                              # }
seed = 42

model_name = "Omics_DrugESPF_Model" # Omics_DrugESPF_Model  Omics_DCSA_Model
AUCtransform = None #"-log2"
# splitType= 'byCCL' # byCCL byDrug 
splitType= 'whole' # ModelID or drug_name or whole
response = "lnIC50"
#------------------graph-------------
drug_graph = False # False True
drug_graph_pool = "add"
DCSA = False # False True # Drug_Cell_SelfAttention
drug_pretrain_weight_path = '../data/DAPL/share/pretrain/drug_encoder.pth' 
#------------------------------------
kfoldCV = 5
include_omics = ['Exp']
DA_Folder = "VAEwC_1" # None VAE_w10SC VAEwC_1
if DA_Folder != 'None':
    omics_files['Exp'] = f"../data/DAPL/share/pretrain/{DA_Folder}/ccle_latent_results.pkl" #
max_drug_len=50 # 不夠補零補到50 / 超過取前50個subwords(index) !!!!須改方法!!!! 
drug_embedding_feature_size = 128 # graph:32; ESPF: 128
ESPF = False # False True
Drug_SelfAttention = False
n_layer = 1 # transformer layer number
pos_emb_type = 'sinusoidal' # 'learned' 'sinusoidal'
#需再修改-----------

intermediate_size =256 # graph:64; ESPF: 256 
num_attention_heads = 8      
attention_probs_dropout_prob = 0.1
hidden_dropout_prob = 0.1
classifier_drop = 0
GINconv_drop =0

if ESPF is True:
    drug_dim = max_drug_len* drug_embedding_feature_size #50*128
    drug_encode_dims =[drug_dim//4,drug_dim//16,drug_dim//64] #  
    dense_layer_dim = sum(omics_encode_dim_dict[omic_type][len(omics_encode_dim_dict[omic_type])-1] for omic_type in include_omics) + drug_encode_dims[-1] # MLPDim
elif ESPF is False:
    drug_encode_dims =[110,55,22] #MACCS166
    dense_layer_dim = sum(omics_encode_dim_dict[omic_type][len(omics_encode_dim_dict[omic_type])-1] for omic_type in include_omics) + drug_encode_dims[-1] # MLPDim
if model_name == "Omics_DCSA_Model":
    drug_encode_dims = None
    conc_dim = (max_drug_len+len(include_omics))*(drug_embedding_feature_size+num_attention_heads)+ (len(include_omics)*drug_embedding_feature_size)
    dense_layer_dim = [conc_dim, conc_dim//10, conc_dim//100, 1] #7064or3736
if model_name == "GIN_DCSA_model":
    if DCSA is True:
        drug_encode_dims = None
        conc_dim = (1+len(include_omics))*(drug_embedding_feature_size+num_attention_heads)+ (len(include_omics)*drug_embedding_feature_size)
        dense_layer_dim = [conc_dim, conc_dim//2, 1] # [conc_dim, conc_dim//2, conc_dim//2 , 1] 40*2+32=112,56,28,1
    elif DCSA is False:
        drug_encode_dims = None
        conc_dim = (1+len(include_omics))*drug_embedding_feature_size # 32*2=64
        dense_layer_dim = [conc_dim, conc_dim//2, 1] #[conc_dim, conc_dim, conc_dim, 1] 64
print("drug_encode_dims",drug_encode_dims)
print("dense_layer_dim",dense_layer_dim)
#需再修改-------------
TrackGradient = False # False True

activation_func = nn.ReLU()  # ReLU activation function # Leaky ReLu
activation_func_final = nn.Sigmoid() # ScaledSigmoid(scale=8) GroundT range ( 0 ~ scale ) # ReLU_clamp(max=8)
#nn.Sigmoid()or ReLU() or Linear/identity(when -log2AUC)
batch_size = 400
num_epoch = 800 # for k fold CV 
patience = 20
learning_rate=1e-05

warmup_lr = True # False True
decrese_epoch = 60
Decrease_percent = 1
continuous = True

CosineAnnealing_LR = False # False True
T_max = 3 # CosinesAnnealingLR step size
eta_min = 1e-06 # CosinesAnnealingLR minimum learning rate

criterion = Custom_LossFunction(loss_type="BCE", loss_lambda=1.0, regular_type=None, regular_lambda=1e-06) #nn.MSELoss()##nn.MSELoss()#
# criterion =  FocalLoss(loss_type="MSE", alpha=8.0, gamma=1.0, regular_type=None, regular_lambda=1e-05) # loss_type="MSE"/"MAE"
# criterion = FocalHuberLoss(loss_type="FocalHuberLoss",delta=0.2, alpha=0.3, gamma=2.0, regular_type=None, regular_lambda=1e-05)
if 'BCE' in criterion.loss_type : 
    metrics_type_set = ["Accuracy","AUROC", "AUPRC", "Sensitivity","Specificity", "Precision", "F1", "Youden", "F1_RecSpe", "F1_RecSpePre" ] 
    metric="Youden" # best_prob_threshold_metric
    best_prob_threshold=0.5
else:
    metrics_type_set = ["MSE", "R^2"] #"MSE","MAE"  None
    metric=None # best_prob_threshold_metric
    best_prob_threshold=None
metrics_calculator = MetricsCalculator_nntorch(types = metrics_type_set)
""" A customizable loss function class.
    Args:
        loss_type (str): The type of loss to use ("RMSE", "MSE", "MAE","BCE","MAE+BCE", "MAE+MSE", "MAE+RMSE")/("weighted_RMSE", "weighted_MSE", "weighted_MAE", "weighted_MAE+MSE", "weighted_MAE+RMSE").
        loss_lambda (float): The lambda weight for the additional loss (MSE or RMSE) if applicable. Default is MAE+ 1.0*(MSE or RMSE).
        regular_type (str): The type of regularization to use ("L1", "L2", "L1+L2"), or None for no regularization.
        regular_lambda (float): The lambda weight for regularization. Default is 1e-05.
        
        # Binary Cross Entropy Loss # already done sigmoid"""
hyperparameter_print = f'  metric ={metric}\n best_prob_threshold ={best_prob_threshold}\n cohort ={cohort}\n geneNUM={geneNUM}\n one_drug ={one_drug}\n drug_df_path ={drug_df_path}\n ESPF_file ={ESPF_file}\n AUC_df_path_numerical ={AUC_df_path_numerical}\n AUC_df_path ={AUC_df_path}\n omics_dict ={omics_dict}\n omics_files ={omics_files}\n TCGA_pretrain_weight_path_dict ={TCGA_pretrain_weight_path_dict}\n drug_pretrain_weight_path ={drug_pretrain_weight_path}\n seed ={seed}\n  model_name ={model_name}\n AUCtransform ={AUCtransform}\n splitType ={splitType}\n response ={response}\n drug_graph ={drug_graph}\n drug_graph_pool ={drug_graph_pool}\n DCSA ={DCSA}\n kfoldCV ={kfoldCV}\n omics_encode_dim ={[(omic_type,omics_encode_dim_dict[omic_type]) for omic_type in include_omics]}\n DA_Folder ={DA_Folder}\n max_drug_len ={max_drug_len}\n drug_embedding_feature_size ={drug_embedding_feature_size}\n ESPF ={ESPF}\n Drug_SelfAttention ={Drug_SelfAttention}\n n_layer ={n_layer}\n pos_emb_type ={pos_emb_type}\n intermediate_size ={intermediate_size}\n num_attention_heads ={num_attention_heads}\n attention_probs_dropout_prob ={attention_probs_dropout_prob}\n hidden_dropout_prob ={hidden_dropout_prob}\n classifier_drop ={classifier_drop}\n drug_encode_dims ={drug_encode_dims}\n dense_layer_dim = {dense_layer_dim}\n activation_func = {activation_func}\n activation_func_final = {activation_func_final}\n batch_size = {batch_size}\n num_epoch = {num_epoch}\n patience = {patience}\n decrese_epoch = {decrese_epoch}\n Decrease_percent = {Decrease_percent}\n continuous ={continuous}\n learning_rate = {learning_rate}\n criterion ={criterion}\n'



__translation_table__ = str.maketrans({
    "*": "",    "/": "",    ":": "-",    "%": "",
    "'": "",    "\"": "",    "[": "",    "]": "",
    ",": "" })

if drug_graph is True:
    hyperparameter_folder_part = (f"{model_name}_{splitType}_DCSA{DCSA}").translate(__translation_table__)
else:
    hyperparameter_folder_part = (f"{model_name}_{splitType}_ESPF{ESPF}_DrugSelfAtten{Drug_SelfAttention}").translate(__translation_table__)

if test is True:
    print("Running in test mode, using small batch size and few epochs for quick testing.")
    batch_size = 100
    num_epoch = 2
    kfoldCV = 2
