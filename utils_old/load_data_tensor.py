import numpy as np
import torch
def load_ccl(filename):
    data = []
    row_names = []
    lines = open(filename).readlines()
    column_names = lines[0].replace('\n', '').split('\t')[1:]
    dx = 1
    for line in lines[dx:451]:
        values = line.replace('\n', '').split('\t')
        row = str.upper(values[0])
        row_names.append(row)
        data.append(values[1:])
    data = torch.tensor(data, dtype=torch.float32, device=device)
    return data, column_names, row_names

def load_drug(filename):
    data = []
    row_names = []
    lines = open(filename).readlines()
    column_names = lines[0].replace('\n', '').split('\t')[1:]
    dx = 1
    for line in lines[dx:1451]:
        values = line.replace('\n', '').split('\t')
        row = str.upper(values[0])
        row_names.append(row)
        data.append(values[1:])
    data = torch.tensor(data, dtype=torch.float32, device=device)
    return data, column_names, row_names
    
def load_AUC(filename):
    data = []
    lines = open(filename).readlines()
    dx = 1
    for line in lines[dx:451]:
        values = line.replace('\n', '').split('\t')
        row = str.upper(values[0])
        row_names.append(row)
        data.append(values[1:1451])
    data = torch.tensor(data, dtype=torch.float32, device=device)
    return data





