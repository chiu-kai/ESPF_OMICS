# def get_data_value_range
# see memory usage # def get_vram_usage()

# read huge maf file
'''
import pandas as pd
# Specify the file path
filename = "/root/data/TCGA _rawdata_maf/mc3.v0.2.8.PUBLIC.maf"
maf_df = pd.read_csv(filename, sep='\t', comment='#', low_memory=False)
# View the first few rows
print(maf_df.head())
for i in maf_df.iloc[0]:
    print(i)
'''    

import numpy as np
import pandas as pd
def get_data_value_range(data, name='', file=None): 
    data = pd.DataFrame(data)
    data=data.values.ravel()
    data = data[~np.isnan(data)]
    
    mean_data = np.mean(data)
    median_data = np.median(data)
    #std_data = np.std(data)
    std_data = np.std(data)
    range_data = np.ptp(data)
    min_data = np.min(data)
    max_data = np.max(data)
    
    TorF = np.array_equal(data, data.astype(bool))# True if data is binary
    print(name," : ", file=file)
    print(f"Range: {range_data:.8f}", file=file)
    print(f"Minimum: {min_data:.8f}", file=file)
    print(f"Maximum: {max_data:.8f}", file=file)
    print(f"Mean: {mean_data:.8f}", file=file)
    print(f"Median: {median_data:.8f}", file=file)
    print(f"Standard Deviation: {std_data:.8f}", file=file)
    print("binary data:",TorF, file=file)
    print("-------------------------------------", file=file)


# see memory usage
import psutil
import subprocess
# Function to get VRAM usage and percentage
def get_vram_usage():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    output = result.stdout.decode().strip().split('\n')
    used_memory, total_memory = map(int, output[0].split(','))
    vram_usage_mb = used_memory  
    vram_percent = (used_memory / total_memory) * 100

    # Get RAM usage and percentage
    ram_usage_mb = psutil.virtual_memory().used/(1024*1024)# Convert to MB
    ram_percent = psutil.virtual_memory().percent

    print("RAM Usage:", ram_usage_mb, "MB")
    print("RAM Usage Percentage:", ram_percent, "%")
    # Get VRAM usage and percentage
    print("VRAM Usage:", vram_usage_mb, "MB")
    print("VRAM Usage Percentage:", vram_percent, "%")
    return None






# 將缺少的gene加上去，全部補零，因為NN input feature的維度是固定的2649
'''
cancer_type_matched_genenames_TCGA2649_df = cancer_type_df[matched_genenames] # 2337genes
# fill_value=0 to missing gene names(columns)
columns_list = (ccle_df.columns).tolist() # 2649genes
cancer_type_matched_genenames_TCGA2649_df = cancer_type_matched_genenames_TCGA2649_df.reindex(columns=columns_list, fill_value=0)
print(cancer_type_matched_genenames_TCGA2649_df.shape)
'''

# convert_mutation_toBinary.ipynb
#convert PDAC MAF mutation info to binary encoding which 1 represents gene having mutation on CDS
'''
# find out unique samples in PDAC_unique_642_Quadruplet.csv
import pandas as pd

PDAC_df = pd.read_csv("./PDAC_unique_642_Quadruplet.csv", sep=',') # With index_col=0, the Patient_ID column will be used as the row index

# Identify duplicated rows
duplicated_mask = PDAC_df.duplicated(subset=['Patient_ID', 'GeneName', 'CDSmut', 'ChemoRegimen'],keep=False)

# Separate unique and duplicated rows
PDAC_df_unique_rows = PDAC_df[~duplicated_mask | duplicated_mask & ~PDAC_df.duplicated(subset=['Patient_ID', 'GeneName', 'CDSmut', 'ChemoRegimen'], keep='first')]
# ~duplicated_mask:只出現一次
# duplicated_mask = PDAC_df.duplicated(,keep=False):出現超過一次
# ~PDAC_df.duplicated( ,keep='first') : 重複的就保留第一個
'''
# use PDAC_df_unique_rows dataframe to generate PDAC_unique_383_Pair_binary_mutation_withoutCNV.csv
'''
# Initialize data structures
patient_ids = set()
gene_names = set()
patient_gene_data = {}
co=0
uni_count=0
splic_count=0
cnv=0
for idx, row in PDAC_df_unique_rows.iterrows():
     if row[26]=="Unique":#0<idx<=sample_count 
          uni_count+=1
          print(row[26])
          patient_id=str(row[0])
          gene_name=row[19].strip() # .strip()去除頭尾的空白格
          print(gene_name)
          AAmut= str(row[20]).strip() 
          # AAmut = [mut.split(' ')[0] for mut in AAmut] #  p.I655V 後面常常會有空格，會造成誤判
          print(AAmut)
          CDSmut= row[21].strip() 
          if "Copy" not in CDSmut: # exclude cope number variable 
               if AAmut and (("?" not in AAmut) and ("nan" not in AAmut))  : # if 有protein change # exclude Splice_Site 
                    co+=1
                    patient_ids.add(patient_id)
                    gene_names.add(gene_name)
                    # Initialize dictionary for each patient
                    if patient_id not in patient_gene_data:
                         patient_gene_data[patient_id] = {}
                    # Mark the presence of the gene mutation
                    patient_gene_data[patient_id][gene_name] = 1
               else:
                    splic_count+=1
                    print("Splice_Site" )
                    print('AAmut',AAmut,"CDSmut",CDSmut,row[27])
                    print(idx)
          else:
               cnv+=1
               print("*"*30)
               print("CNV",CDSmut,'AAmut',AAmut)

print('co',co)
print("uni_count",uni_count)
print("splic_count",splic_count)
print('cnv',cnv)
# Step 2: Prepare the unique Patient IDs and Gene Names
patient_ids = sorted(patient_ids, key=int) # sort by str會是106最小，因為開頭是1
gene_names = sorted(gene_names)

print(patient_ids)
print(len(patient_ids))
print(gene_names)
print(len(gene_names))

print([i for i in patient_gene_data['90']])

# Step 3: Write the output CSV file
outputfilename = "./PDAC_unique_383_Pair_binary_mutation_withoutCNV.csv"

with open(outputfilename, mode='w', newline='') as outfile:
    writer = csv.writer(outfile)
    # Write the header
    header = ['Patient_ID'] + gene_names
    writer.writerow(header)
    # Write the data rows
    for patient_id in patient_ids:
        row = [patient_id]
        for gene_name in gene_names:
            if gene_name in patient_gene_data[patient_id]:
                row.append(1)
            else:
                row.append(0)
        writer.writerow(row)
print("CSV file has been written successfully.")
'''
# convert TCGA MAF mutation info to binary encoding which 1 represents gene having mutation on CDS
'''
cancer_type = "BRCA"
import csv
inputfilename =f"./TCGA _rawdata_maf/mut-{cancer_type}.csv"
outputfilename = f"./TCGA _rawdata_maf/mut-{cancer_type}_binary_excludeSpliceSite.csv"
# Initialize data structures
Tumor_Sample_ids = set()
gene_names = set()
patient_gene_data = {}
with open(inputfilename, 'r') as inputfile:
    reader = csv.reader(inputfile)# Create a CSV reader object
    for idx,row in enumerate(reader):
        if 0<idx:
            Tumor_Sample_id=row[17] # Tumor_Sample_Barcode
            # HGVSp_Short=row[38]# HGVSp_Short 是否影響Protein #SpliceSite as 1
            HGVSp=row[37]# HGVSp 是否影響Protein
            gene_name = row[2] #Hugo_Symbol
            # if "=" not in HGVSp_Short and HGVSp_Short != "NA":#並非'='或"NA" #SpliceSite as 1
            if "=" not in HGVSp and HGVSp != "NA":#並非'='或"NA"#SpliceSite as 0
                Tumor_Sample_ids.add(Tumor_Sample_id)
                gene_names.add(gene_name)
                # Initialize dictionary for each patient
                if Tumor_Sample_id not in patient_gene_data:
                    patient_gene_data[Tumor_Sample_id] = {}
                # Mark the presence of the gene mutation
                patient_gene_data[Tumor_Sample_id][gene_name] = 1
# Step 2: Prepare the unique Patient IDs and Gene Names
import re
def alphanumeric_sort_key(s):# sort Tumor_Sample_ids by the alphabet and number
    # Split the string into a list of alternating numeric and non-numeric parts
    parts = re.split('([0-9]+)', s)
    # Convert numeric parts to integers for proper numerical sorting
    return [int(part) if part.isdigit() else part for part in parts]

Tumor_Sample_ids = sorted(Tumor_Sample_ids, key=alphanumeric_sort_key)
gene_names = sorted(gene_names)
# print(Tumor_Sample_ids)
# print(gene_names)            
#print([i for i in patient_gene_data['0023']])
# Step 3: Write the output CSV file
with open(outputfilename, mode='w', newline='') as outfile:
    writer = csv.writer(outfile)
    # Write the header
    header = ['Tumor_Sample_id'] + gene_names
    writer.writerow(header)
    # Write the data rows
    for Tumor_Sample_id in Tumor_Sample_ids:
        row = [Tumor_Sample_id]
        for gene_name in gene_names:
            if gene_name in patient_gene_data[Tumor_Sample_id]:
                row.append(1)
            else:
                row.append(0)
        writer.writerow(row)
print("CSV file has been written successfully.")


inputfilename = f"./TCGA _rawdata_maf/mut-{cancer_type}_binary_excludeSpliceSite.csv"
with open(inputfilename, 'r') as inputfile:
    reader = csv.reader(inputfile)# Create a CSV reader object
    instance_count=0
    tumer_sample_count=0
    for idx,row in enumerate(reader): 
        if idx>0:
            instance_count+=row.count('1')
            tumer_sample_count+=1
print(len(row)-1)#gene count
print(instance_count)
print(tumer_sample_count)
'''



'''
# load CCLE data
data_exp,  exp_gene_names, ccl_names_exp = load_drug("/home/40523001/Winnie/data/match_ccle_expression_ordered.txt")

# load tcga data
tcga_exp_data, tumer_names, tcga_exp_genenames = load_ccl("/home/40523001/Winnie/data/tcga_exp_data_paired_with_ccl.txt")

# modify gene names of ccle
ccl_exp_genenames = [name.split(' ')[0] for name in exp_gene_names]

# get matched expression genenames of tcga and ccle
matched_exp_genenames = sorted(set(tcga_exp_genenames) & set(ccl_exp_genenames))
print("tcga_exp_genenames:",np.array(tcga_exp_genenames).shape)
print("ccl_exp_genenames:",np.array(ccl_exp_genenames).shape)
print("matched genenames in alphabetical order:",np.array(matched_exp_genenames).shape)

 
        
# use dataframe 
import pandas as pd
tcga_df = pd.read_csv("/home/40523001/Winnie/data/tcga_exp_data_paired_with_ccl.txt", sep='\t', index_col=0)
ccl_df = pd.read_csv("/home/40523001/Winnie/data/match_ccle_expression_ordered.txt", sep='\t', index_col=0)

# modify gene names of ccle
ccl_df.columns = [col.split(' ')[0] for col in ccl_df.columns]

# get ccle filtered dataframe by wanted list of column names
matched_exp_genenames_ccl_df = ccl_df[matched_exp_genenames]
# get tcga filtered dataframe by wanted list of column names
matched_exp_genenames_tcga_df = tcga_df.T[matched_exp_genenames]

# write tcga dataframe to txt file
matched_exp_genenames_tcga_df.index =  tumer_names # row names
file_path = "/home/40523001/Winnie/data/TCGA_exp_matched_genenames_CCLE.txt"
# Save the DataFrame to a text file
matched_exp_genenames_tcga_df.to_csv(file_path, sep='\t', index_label='tumer_names', header=True) # header=True:column name

# write ccle dataframe to txt file
matched_exp_genenames_ccl_df.index =  ccl_names_exp # row names
file_path = "/home/40523001/Winnie/data/CCLE_exp_matched_genenames_TCGA.txt"
# Save the DataFrame to a text file
matched_exp_genenames_tcga_df.to_csv(file_path, sep='\t', index_label='ccl_names_exp', header=True) # header=True:column name



data_exp = np.array(matched_exp_genenames_ccl_df)
matched_exp_genenames = matched_exp_genenames_ccl_df.columns.tolist()
print(type(data_exp))
print(np.shape(data_exp))
print(type(matched_exp_genenames))
print(np.shape(matched_exp_genenames))
'''


