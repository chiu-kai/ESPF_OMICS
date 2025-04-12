# main_kfold_GDSC_3Class.py
# pip install subword-nmt seaborn lifelines openpyxl matplotlib scikit-learn openTSNE
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import copy
from scipy import stats
import gc
import os
import importlib.util

from utils.ESPF_drug2emb import drug2emb_encoder
from utils.Model import Omics_DrugESPF_Model, Omics_DCSA_Model
from utils.split_data_id import split_id,repeat_func
from utils.create_dataloader import OmicsDrugDataset
from utils.train import train, evaluation
from utils.correlation import correlation_func
from utils.plot import loss_curve, correlation_density, Confusion_Matrix_plot
from utils.tools import set_seed