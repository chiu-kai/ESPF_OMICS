# pip install subword-nmt seaborn lifelines openpyxl matplotlib scikit-learn openTSNE
# pip install torchmetrics==1.2.0 pandas==2.1.4 numpy==1.26.4
# python3 ./main_kfold.py --config utils/config.py
import torch.nn as nn
from utils.Loss import Custom_LossFunction,Custom_Weighted_LossFunction,BCE_FocalLoss
from utils.Custom_Activation_Function import ScaledSigmoid, ReLU_clamp
from utils.Metrics import MetricsCalculator_nntorch

model_inference = True # False
if model_inference is True:
    cohort = "TCGA"
    geneNUM = "" # _4692genes tcgadata tcgalabel tcgadata_4692genes tcgalabel_4692genes

test = False #False, True: batch_size = 3, num_epoch = 2, full dataset

drug_df_path= "../data/GDSC/GDSC_drug_merge_pubchem_dropNA_MACCS.csv"
one_drug=None # gsk690693 trametinib erlotinib
ESPF_file = "./ESPF/(NAN) subword_units_map_chembl_freq_1500.csv" #(NAN) subword_units_map_chembl_freq_1500.csv
AUC_df_path_numerical = "../data/GDSC/GDSC2_fitted_dose_response_27Oct23 from GDSC MaxScreen threshold ModelID966 drug230 samples145655 down_balanced_combined.csv" # gdsc1+2_ccle_z-score　gdsc1+2_ccle_AUC
AUC_df_path = "../data/GDSC/GDSC2_fitted_dose_response_27Oct23 from GDSC MaxScreen threshold ModelID966 drug230 samples145655 down_balanced_combined.csv"
response_file = 'down_balanced_combined'# down/up _ high/even/low 、imbalanced
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

# TCGA pretrain weight load or not 
TCGA_pretrain_weight_path_dict = None #{'Mut': "./results/Encoder_tcga_mut_1000_100_50_best_loss_0.0066.pt",
                                  #'Exp': "./results/Encoder_tcga_exp_128_32_best_loss_0.2182988.pt", # "./results/Encoder_tcga_exp_128_32_best_loss_0.2182988.pt", "./results/Encoder_tcga_exp_1000_100_50_best_loss_0.7.pt"
                                  # Add more omics types and paths as needed
                              # }
seed = 42

model_name = "GIN_DCSA_model" # Omics_DrugESPF_Model  OmicsESPF_DCSA_Model GIN_DCSA_model
AUCtransform = None #"-log2"
# samples splitType= 'byCCL' # byCCL byDrug 
splitType= 'ModelID' # ModelID or drug_name or whole
response = "lnIC50" # 

kfoldCV = 5
include_omics = ['Exp']
DA_Folder = "None" # None  VAEwC_1  VAE_w10SC  VAE_w5SC  VAE_gFID  VAE_0  VAE
if DA_Folder != 'None':
    omics_files['Exp'] = f"../data/DAPL/share/pretrain/{DA_Folder}/ccle_latent_results.pkl" #

#------------------graph-------------
drug_graph = True # False True
drug_graph_pool = "add"
Graph_norm_type = 'graph' # 'batch' | 'graph' | 'none'
Graph_nlayers = 3 #
DCSA = False # False True # Drug_Cell_SelfAttention
drug_pretrain_weight_path = None # None / '../data/DAPL/share/pretrain/drug_encoder.pth'
drug_pretrain_freeze_emb_pth = '../data/Drug_pretrain/SCAGE/gdsc_pretrain_latent.pkl' # None
drug_pretrain_freeze_emb= "graph_token" # None / "graph_token" / "atom_mean"
drug_pretrain_freeze_emb_MLP = False # True / False

#------------------ESPF------------------
ESPF = False # False True
max_drug_len=50 # 不夠補零補到50 / 超過取前50個subwords(index) !!!!須改方法!!!! 
Drug_SelfAttention = False

#---------Attention-------
n_layer = 1 # transformer layer number # control both Drug_SelfAttention and DCSA
pos_emb_type = 'sinusoidal' # 'learned' 'sinusoidal'

drug_emb_dim = 512 # graph: 10/32/64/128/256/512 ; ESPF: 128
DCmatch_emb_dim = max(drug_emb_dim, omics_encode_dim_dict['Exp'][-1]) # for Drug_Cell_SelfAttention, match the dimension of drug and cell (omic) embeddings to make sure they are in the same feature space when calculate attention scores

intermediate_size = drug_emb_dim*2 # graph:64; ESPF: 256 
num_attention_heads = 8      
attention_probs_dropout_prob = 0.1
hidden_dropout_prob = 0.1
classifier_drop = 0
GINconv_drop =0

if ESPF is True:
    drug_dim = max_drug_len* drug_emb_dim #50*128
    drug_encode_dims =[drug_dim//4,drug_dim//16,drug_dim//64] #  
    dense_layer_dim = sum(omics_encode_dim_dict[omic_type][len(omics_encode_dim_dict[omic_type])-1] for omic_type in include_omics) + drug_encode_dims[-1] # MLPDim
elif ESPF is False:
    drug_encode_dims =[110,55,22] #MACCS166
    dense_layer_dim = sum(omics_encode_dim_dict[omic_type][len(omics_encode_dim_dict[omic_type])-1] for omic_type in include_omics) + drug_encode_dims[-1] # MLPDim
if model_name == "OmicsESPF_DCSA_Model":
    drug_encode_dims = None
    conc_dim = (max_drug_len+len(include_omics))*(drug_emb_dim+num_attention_heads)+ (len(include_omics)*drug_emb_dim)# + ccl emb skip connection
    dense_layer_dim = [conc_dim, conc_dim//10, conc_dim//100, 1] #7064or3736
if model_name == "GIN_DCSA_model":
    if DCSA is True:
        if drug_pretrain_freeze_emb_MLP is True:
            drug_encode_dims = [drug_emb_dim//2, drug_emb_dim//4] # 512->256->128
            DCmatch_emb_dim = max(drug_encode_dims[-1], omics_encode_dim_dict['Exp'][-1]) #128
            conc_dim = (1+len(include_omics))*(DCmatch_emb_dim+num_attention_heads)+ (len(include_omics)*DCmatch_emb_dim)# + ccl emb skip connection # (128+8)*2+128= 400
            dense_layer_dim = [conc_dim, conc_dim, conc_dim//2, 1] # 400,400,200,1
        else:
            drug_encode_dims = None # 512 / 64
            conc_dim = (1+len(include_omics))*(DCmatch_emb_dim+num_attention_heads)+ (len(include_omics)*DCmatch_emb_dim)# + ccl emb skip connection # (64+8)*2+64= 208 # (512+8)*2+512= 1040
            dense_layer_dim = [conc_dim, conc_dim//2, conc_dim//4, 1] # [208, 104,52, 1] #(32+8)*2+32=112,56,28,1 # 1040,520,260,1
    elif DCSA is False:
        if drug_pretrain_freeze_emb_MLP is True:
            drug_encode_dims = [drug_emb_dim//2, drug_emb_dim//4] # 512->256->128
            DCmatch_emb_dim = max(drug_encode_dims[-1], omics_encode_dim_dict['Exp'][-1])
            conc_dim = (1+len(include_omics))*DCmatch_emb_dim # 128*2=256
            dense_layer_dim = [conc_dim, conc_dim, conc_dim//2, 1] #[256, 256, 128, 1] # 
        else:
            drug_encode_dims = None
            conc_dim = (1+len(include_omics))*DCmatch_emb_dim # 512*2=1024
            dense_layer_dim = [conc_dim, conc_dim//2, conc_dim//4, 1] #[1024, 512, 256, 1]  # [128,64,32,1]
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

LR_Scheduler = "CyclicLR" # "decay_LR" / "CosineAnnealing_LR" / "CyclicLR" / None
# decay_LR 
decay_epoch = 60 # Decay learning rate at this epoch if decay_LR is True
decay_percent = 1 # Decay_percent = 1 means no decay
continuous = True # if True, decay learning rate every epochs after decay_epoch ; if False, decay learning rate only once at decay_epoch
# CosineAnnealing_LR 
T_max = 3                 # step size
eta_min = 1e-06           # minimum learning rate
# CyclicLR 
base_lr = 5e-06 # Initial learning rate which is the lower boundary in the cycle for each parameter group
max_lr = 5e-05 # Upper learning rate boundaries in the cycle for each parameter group
step_size_up = 8 # Number of training iterations in the increasing half of a cycle
step_size_down = 64 # Number of training iterations in the decreasing half of a cycle
cyclic_gamma = 0.992 
cycle_momentum=False #Adam optimizer does not use momentum, so we set cycle_momentum to False when using CyclicLR with Adam.
mode = "exp_range" # One of {triangular, triangular2, exp_range}. Default: 'triangular'.

criterion = Custom_LossFunction(loss_type="BCE", loss_lambda=1.0, regular_type=None, regular_lambda=1e-06) #nn.MSELoss()##nn.MSELoss()#
# criterion = BCE_FocalLoss(loss_type="BCE_Focal",alpha=0.75, gamma=3.0, reduction='mean', regular_type=None, regular_lambda=0.001)
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
hyperparameter_print = f'  metric ={metric}\n best_prob_threshold ={best_prob_threshold}\n cohort ={cohort}\n geneNUM={geneNUM}\n one_drug ={one_drug}\n drug_df_path ={drug_df_path}\n ESPF_file ={ESPF_file}\n AUC_df_path_numerical ={AUC_df_path_numerical}\n AUC_df_path ={AUC_df_path}\n omics_dict ={omics_dict}\n omics_files ={omics_files}\n TCGA_pretrain_weight_path_dict ={TCGA_pretrain_weight_path_dict}\n drug_pretrain_weight_path ={drug_pretrain_weight_path}\n drug_pretrain_freeze_emb_pth ={drug_pretrain_freeze_emb_pth}\n drug_pretrain_freeze_emb ={drug_pretrain_freeze_emb}\n seed ={seed}\n  model_name ={model_name}\n AUCtransform ={AUCtransform}\n splitType ={splitType}\n response ={response}\n drug_graph ={drug_graph}\n drug_graph_pool ={drug_graph_pool}\n Graph_norm_type ={Graph_norm_type}\n Graph_nlayers ={Graph_nlayers}\n DCSA ={DCSA}\n kfoldCV ={kfoldCV}\n omics_encode_dim ={[(omic_type,omics_encode_dim_dict[omic_type]) for omic_type in include_omics]}\n DA_Folder ={DA_Folder}\n drug_emb_dim ={drug_emb_dim}\n DCmatch_emb_dim ={DCmatch_emb_dim}\n ESPF ={ESPF}\n max_drug_len ={max_drug_len}\n Drug_SelfAttention ={Drug_SelfAttention}\n n_layer ={n_layer}\n pos_emb_type ={pos_emb_type}\n intermediate_size ={intermediate_size}\n num_attention_heads ={num_attention_heads}\n attention_probs_dropout_prob ={attention_probs_dropout_prob}\n hidden_dropout_prob ={hidden_dropout_prob}\n classifier_drop ={classifier_drop}\n GINconv_drop ={GINconv_drop}\n drug_encode_dims ={drug_encode_dims}\n dense_layer_dim = {dense_layer_dim}\n activation_func = {activation_func}\n activation_func_final = {activation_func_final}\n batch_size = {batch_size}\n num_epoch = {num_epoch}\n patience = {patience}\n learning_rate = {learning_rate}\n criterion ={criterion}\n LR_Scheduler ={LR_Scheduler}\n decay_epoch = {decay_epoch}\n decay_percent = {decay_percent}\n continuous ={continuous}\n T_max ={T_max}\n eta_min ={eta_min}\n base_lr ={base_lr}\n max_lr ={max_lr}\n step_size_up ={step_size_up}\n step_size_down ={step_size_down}\n cyclic_gamma ={cyclic_gamma}\n cycle_momentum ={cycle_momentum}\n mode ={mode}\n TrackGradient = {TrackGradient} \n'

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
