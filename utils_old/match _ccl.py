#for Train_byCCL_mut_drug_exp.ipynb
import pandas as pd
import numpy as np
# Load data
ccle_expression = pd.read_csv("/home/40523001/Adversarial Deconfounding/data/CCLE_expression.csv")
data_mut = pd.read_csv("/home/40523001/Winnie/data/ccle_matched_tcga_mutation_ordered.txt", delimiter='\t')
data_AUC = pd.read_csv("/home/40523001/Winnie/data/[new]AUC_imp_filtered_ordered.txt", delimiter='\t')
same_cell_lines = list(set(ccle_expression.iloc[:, 0]).intersection(set(data_mut.iloc[:, 0])))

# Filter CCLE_expression to include only common cell lines
new_ccle_expression = ccle_expression[ccle_expression.iloc[:, 0].isin(same_cell_lines)]
new_data_mut = data_mut[data_mut.iloc[:, 0].isin(same_cell_lines)]
new_data_AUC= data_AUC[data_AUC.iloc[:, 0].isin(same_cell_lines)]
# Sort the new_ccle_expression DataFrame based on the first column
new_ccle_expression = new_ccle_expression.sort_values(by=new_ccle_expression.columns[0])
print(new_data_AUC.shape)

# Save the new CCLE_expression dataset
new_ccle_expression.to_csv('/home/40523001/Winnie/data/match_ccle_expression_ordered.txt', sep='\t', index=False)
new_data_mut.to_csv('/home/40523001/Winnie/data/match_ccle_matched_tcga_mutation_ordered.txt', sep='\t', index=False)
new_data_AUC.to_csv('/home/40523001/Winnie/data/match_AUC_imp_filtered_ordered.txt', sep='\t', index=False)
