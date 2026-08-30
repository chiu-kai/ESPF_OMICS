# Cancer Drug response prediction

## Installation

### Environment

* **OS:** Ubuntu 22.04
* **Container:** NVIDIA NGC GPU-optimized PyTorch container

  * NGC: [https://catalog.ngc.nvidia.com/?filters=&orderBy=weightPopularDESC&query=](https://catalog.ngc.nvidia.com/?filters=&orderBy=weightPopularDESC&query=)
  * **Image:** `nvcr.io/nvidia/pytorch:20.08-py3`

### Install Python Dependencies

```bash
pip install subword-nmt seaborn lifelines openpyxl matplotlib \
    scikit-learn openTSNE torchmetrics==1.2.0 \
    pandas==2.1.4 numpy==1.26.4

pip install torch-geometric==2.3.1 \
    hickle==5.0.2 \
    networkx==2.6.3 \
    rdkit-pypi==2023.3.1b1
```

---

## Training

step1: set all the hyperparameters in utils/config.py

step2: Run the following command to train and inference the model:

```bash
python3 ./main_kfold_instance.py --config utils/config.py
```

---

## Inference

Run inference only process using specific weight path:

```bash
python3 ./main_kfold_instance_inference.py \
    --config utils/config_GDSC.py \
    --path "2026-0503-0736_BF3_BCE_test_loss0.4556333_BestValEpo63_filedown_balanced_combined_GIN_DCSA_model_ModelID_DCSAFalse_Exp1426_nlayer1_DA-None" \
    --threshold "0.3868679702281952" \
    --BF "3"
```

---

## Dataset

Place the dataset directory **one level above** the project directory.
```text
root/
├── CDR_prediction/
│   ├── main_kfold_instance.py
│   ├── main_kfold_instance_inference.py
│   └── ...
└── data/
```

```text
data/                                  # 外部資料根目錄（../data/）
├── GDSC/
│   ├── GDSC_drug_merge_pubchem_dropNA_MACCS.csv
│   ├── GDSC2_fitted_dose_response_27Oct23 from GDSC MaxScreen threshold ModelID963 drug221 samples103840 balanced_by_cell_gap5.csv   
│   ├── GDSC2_fitted_dose_response_27Oct23 from GDSC MaxScreen threshold ModelID966 drug230 samples145655 down_balanced_combined.csv
│   ├── GDSC2_fitted_dose_response_27Oct23 from GDSC MaxScreen threshold ModelID678 drug230 samples142188 balanced_high.csv
│   ├── GDSC2_fitted_dose_response_27Oct23 from GDSC MaxScreen threshold ModelID678 drug230 samples142188 balanced_even.csv
│   ├── GDSC2_fitted_dose_response_27Oct23 from GDSC MaxScreen threshold ModelID678 drug230 samples142188 balanced_low.csv
│   ├── GDSC2_fitted_dose_response_27Oct23 from GDSC MaxScreen threshold ModelID678 drug230 samples142188 upsampling balanced.csv
│   └── GDSC2_fitted_dose_response_27Oct23 from GDSC.xlsx
├── CCLE/
│   ├── Expression_Public_23Q4_subsetted.csv
│   └── CCLE_exp_476samples_4692genes.txt
│
├── DAPL/share/
│   ├── ccle_uq1000_feature_sorted.csv       # CCLE 1426 exp
│   ├── pretrain_tcga.csv                    # TCGA 1426 exp (all TCGA samples)
│   ├── xena_sample_info_df.csv              # TCGA sample info
│   ├── pretrain/
│   │   ├── drug_encoder.pth                 # GIN drug pretrained weights
│   │   └── {DA_Folder}/                     # VAEwC_1 | VAE_w10SC | VAE_w5SC | VAE_gFID | VAE_0 | VAE | VAEwC_1 exp_11-param_058
│   │       ├── ccle_latent_results.pkl      # CCLE embedding
│   │       ├── tcga_latent_results.pkl      # TCGA embedding
│   │       ├── tcga_latent_results_DAPL_rmdup.csv      # TCGA embedding match "tcga_latent_results.pkl" and "TCGA_drug_response_from_DAPL.csv" 
│   │       ├── tcga_latent_results_TransDRP_rmdup.csv  # TCGA embedding match "tcga_latent_results.pkl" and "
│   │       └── tcga_latent_results_DiSyn_rmdup.csv     # TCGA embedding match "tcga_latent_results.pkl" and "DiSyn TCGA drug response match exp file samples.csv"
│   ├── TCGA_fromDAPL/
│   │   ├── TCGA_drug_response_from_DAPL.csv  # DAPL TCGA response label
│   │   └── TCGA_EXP1426_from_DAPL.csv        # DAPL TCGA 1426 exp
│   ├── PDTC_fromDAPL/
│   │   ├── PDX_drug_response_from_DAPL.csv   # DAPL PDX response label
│   │   └── pdtc_uq1000_feature.csv           # DAPL PDX 1426 exp
├── TCGA/
│   ├── DiSyn TCGA drug response match exp file samples.csv   # DiSyn TCGA response label
│   ├── TCGA DiSyn samples EXP1426.csv                        # DiSyn TCGA 1426 exp 
│   ├── TransDRP TCGA drug response samples.csv               # TransDRP TCGA response label
│   ├── TCGA TransDRP samples EXP1426 rmdup.csv               # TransDRP TCGA 1426 exp 
│   ├── TCGA DeepCDR samples.csv                              # DeepCDR TCGA response label
│   └── TCGA DeepCDR samples EXP1426.csv                      # DeepCDR TCGA 1426 exp

│
└── no_Imputation_PRISM_Repurposing_Secondary_Screen_data/    
    ├── MACCS(Secondary_Screen_treatment_info)_union_NOrepeat.csv                                         # PRISM drug df
    ├── Drug_sensitivity_AUC_(PRISM_Repurposing_Secondary_Screen)_subsetted_NOrepeat.csv                  # PRISM AUDRC response label
    └── Drug_sensitivity_AUC_(PRISM_Repurposing_Secondary_Screen)_subsetted_NOrepeat_instance_format.csv  # PRISM AUDRC response label
```


