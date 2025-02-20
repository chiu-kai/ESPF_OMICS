import numpy as np

def load_ccl(filename):
    data = []
    row_names = []
    lines = open(filename).readlines()
    if "csv" in filename:
        column_names = lines[0].replace('\n', '').split(',')[1:] # num_drug+1
    elif "txt" in filename:
        column_names = lines[0].replace('\n', '').split('\t')[1:] # num_drug+1
    dx = 1
    for line in lines[dx:]: # (num_ccl+1)
        if "csv" in filename:
            values = line.replace('\n', '').split(',')
        elif "txt" in filename:
            values = line.replace('\n', '').split('\t')
        row = str.upper(values[0])
        row_names.append(row)
        data.append(values[1:])
    data = np.array(data, dtype='float32')
    return data, column_names, row_names

def load_drug_matrix(filename):# for winnie file # matrix
    data = []
    row_names = []
    lines = open(filename).readlines()
    column_names = lines[0].replace('\n', '').split('\t')[1:]
    dx = 1
    for line in lines[dx:]:
        values = line.replace('\n', '').split('\t')
        row = str.upper(values[0])
        row_names.append(row)
        data.append(values[1:])
    data = np.array(data, dtype='float32')
    return data, column_names, row_names
    
def load_drug(filename): # for PDAC MACCS166 # with other info in file
    data = [] # MACCS166
    row_names = [] # drug_names
    lines = open(filename).readlines()
    
    for line in lines[1:]:
        values = line.replace('\n', '').split('\t')
        MACCS = [int(item) for item in (values[9].replace('"', '').split(','))]
        row_names.append(values[0])
        data.append(MACCS)
    data = np.array(data, dtype='float32')
    return data, row_names
    
def load_AUC_matrix(splitType,filename):
    # splitType = "byCCCL" or "byDrug" 決定AUCmatrix要不要轉置
    """
    Load an AUC matrix from a CSV file and process it based on the split type.
    Args:
        - splitType (str): Type of split. Options are "byCCCL" or "byDrug".
        - filename (str): Path to the CSV file containing AUC data.
    Returns:
        - Tuple[np.ndarray, List[str], List[str]]: A tuple containing:
            - A NumPy array of the AUC data.
            - A list of column names.
            - A list of row names.
    """
    data = []
    row_names = []
    lines = open(filename).readlines()
    if "csv" in filename:
        column_names = lines[0].replace('\n', '').split(',')[1:] # num_drug+1
    elif "txt" in filename:
        column_names = lines[0].replace('\n', '').split('\t')[1:] # num_drug+1
    for line in lines[1:]: # (num_ccl+1)
        if "csv" in filename:
            values = line.replace('\n', '').split(',')
        elif "txt" in filename:
            values = line.replace('\n', '').split('\t')
        row = str.upper(values[0])
        row_names.append(row)
        data.append(values[1:])# num_drug+1
    filled_nan_data = [[float(x) if x else np.nan for x in row] for row in data]
    filled_nan_data = np.array(filled_nan_data)#, dtype='float32'
    if splitType == "byCCL":
        pass # No modification needed
    elif splitType == "byDrug":
        filled_nan_data = filled_nan_data.T
    else:
        raise ValueError("Invalid splitType. Choose either 'byCCCL' or 'byDrug'.")
    return filled_nan_data, column_names, row_names

def load_AUC_instance(filename):
    # for FPS_auc_unadj_ordered.txt
    instance_AUC = []
    lines = open(filename).readlines()
    dx = 1
    for line in lines[dx:]:
        AUC = float(line.replace('\n', '').split('\t')[1])
        instance_AUC.append(AUC)
    instance_AUC = np.array(instance_AUC, dtype='float32')
    return instance_AUC



