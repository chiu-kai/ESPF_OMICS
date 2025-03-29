import csv
from rdkit import Chem
from rdkit.Chem import MACCSkeys

"""
將有單個或多個smiles的藥都是取第一個smile去轉MACCS
"""
import csv
from rdkit import Chem
from rdkit.Chem import MACCSkeys

inputfilename ='../../data/DAPL/share/GDSC_drug_merge_pubchem_dropNA.csv'
outputfilename = '../../data/DAPL/share/GDSC_drug_merge_pubchem_dropNA_MACCS.csv'

with open(outputfilename, 'w', newline='') as outputfile: # Open a new CSV file in write mode
    writer = csv.writer(outputfile) # Create a CSV writer object
    
    with open(inputfilename, 'r') as file:
        reader = csv.reader(file)# Create a CSV reader object
        header=['DRUG_NAME', 'PATHWAY_NAME', 'synonyms', 'pathway_name', 'target', 'pubchem', 'Dataset', 'name', 'SMILES','MACCS166bits']
        writer.writerow(header)# Write the header row
        for idx,row in enumerate(reader):
            if idx>0:
                smile = row[8]
                if "," in smile:
                    smile = row[8].split(',')[0]
                fps = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smile))# get 166 ExplicitBitVect object
                bitfeatures = [int(bit) for bit in fps][1:]
                out = row + [', '.join(str(x) for x in bitfeatures)]
                writer.writerow(out)# Write a row to the CSV file



"""
將有多個smiles的藥中MACCS不一致的的藥挑出來另外存
"""
import csv
from rdkit import Chem
from rdkit.Chem import MACCSkeys

def all_strings_same(ls):
    first_string = ls[0].split(',')
    for string in ls[1:]:
        diff_bit_indices=[]
        string =string.split(',')
        diff_bit_indices = [i for i in range(len(string)) if string[i] != first_string[i]]
        if diff_bit_indices:
            print("mismatch")
            print(f'total {len(diff_bit_indices)} bits are different at {diff_bit_indices}')
            return False  
    print("match")
    return True

inputfilename ='./no_Imputation_PRISM_Repurposing_Secondary_Screen_data/smiles(Secondary_Screen_treatment_info).csv'
outputfilename = './no_Imputation_PRISM_Repurposing_Secondary_Screen_data/2MACCS(Secondary_Screen_treatment_info).csv'
output_mismatch_filename = './no_Imputation_PRISM_Repurposing_Secondary_Screen_data/mismatch_MACCS.csv'

mismatch_MACCS=[]

with open(outputfilename, 'w', newline='') as outputfile: # Open a new CSV file in write mode
    writer = csv.writer(outputfile) # Create a CSV writer object
    
    with open(inputfilename, 'r') as inputfile:
        reader = csv.reader(inputfile)# Create a CSV reader object
        header=['', 'Name', 'BRD_ID', 'name', 'moa', 'target', 'disease.area', 'indication', 'smiles', 'phase','MACCS166bits']
        writer.writerow(header)# Write the header row
        for idx,row in enumerate(reader):
            if idx>0:
                print(idx)
                smile = row[8] # type(smile) <class 'str'>
                #print(smile)
                if smile != "":
                    if "," in smile:
                        smileS = row[8].split(',') # type(smileS) <class 'list'>
                        non_empty_smileS_ls = [item for item in smileS if item.strip()] # Filter out empty strings
                        if not non_empty_smileS_ls: # if the list contains only empty strings
                            print("no smile in ls but have ',' ")
                        bitfeaturesList=[]
                        print(non_empty_smileS_ls)
                        for i in non_empty_smileS_ls: # type(i) <class 'str'>
                            fps = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(i))# get 166 ExplicitBitVect object
                            bitfeatures = [int(bit) for bit in fps][1:] # 1~166
                            bitfeaturesList += [', '.join(str(x) for x in bitfeatures)]# ["smile1","smile2"]
                        if all_strings_same(bitfeaturesList) is True:
                            out = row + [bitfeaturesList[0]]
                            writer.writerow(out)
                        else:
                            mismatch_MACCS.append(row + [bitfeaturesList])
                    else:
                        fps = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smile))# get 166 ExplicitBitVect object
                        bitfeatures = [int(bit) for bit in fps][1:]
                        out = row + [', '.join(str(x) for x in bitfeatures)]
                        writer.writerow(out)# Write a row to the CSV file
                else: 
                    print("smile is empty")


with open(output_mismatch_filename, 'w', newline='') as outputfile: # Open a new CSV file in write mode
    writer = csv.writer(outputfile) # Create a CSV writer object
    header=['', 'Name', 'BRD_ID', 'name', 'moa', 'target', 'disease.area', 'indication', 'smiles', 'phase','MACCS166bits']
    writer.writerow(header)# Write the header row
    for i in mismatch_MACCS:
        writer.writerow(i)