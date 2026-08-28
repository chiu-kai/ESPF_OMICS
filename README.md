# ESPF and Omics

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

## Dataset

Place the dataset directory **one level above** the project directory.

For example:

```text
project_root/
├── ESPF_and_Omics/
│   ├── main_kfold_instance.py
│   ├── main_kfold_instance_inference.py
│   └── ...
└── data/
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
