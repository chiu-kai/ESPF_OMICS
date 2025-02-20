# ESPF_drug2emb.py
import codecs
from subword_nmt.apply_bpe import BPE
import numpy as np

def drug2emb_encoder(smile,vocab_path,sub_csv, max_d=50): # drug_smiles['smiles']
    
    bpe_codes_drug = codecs.open(vocab_path) # token
    # dbpe: initiate BPE function 
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='') # merges=-1 : no limit on the number of merge operations, and BPE will continue merging subword units until it can no longer find any meaningful merges. 
                                                        #In other words, it will keep merging until the training process converges.
                                                        #separator='' : uses an empty string ('') as a separator between subword units.
    idx2word_d = sub_csv['index'].values # index : substructure EX:CC
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))# zip substructure with it's substructure character length. EX:(CC,2)

    t1 = dbpe.process_line(smile).split() # BPE tokenizes the input SMILES # t1=vocabulary set # ex:[ 'Nc1nc(', 'O)', 'c2nc(', 'Br)', 'n(', '[C@@H]3' ]
        # t1: list of subwords of a word
    try:
        i1 = np.asarray([words2idx_d[subword ] for subword  in t1]) # i1: subword在sub_csv file中的index ; i: t1中的subword
    except:# 如果t1是空值，或t1中有subword沒有在words2idx_d字典裡
        i1 = np.array([0])
    l = len(i1)
    if l < max_d: # subword list 長度小於50
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0) # 補零補到50，前面是subword在sub_csv file中的index
        input_mask = ([1] * l) + ([0] * (max_d - l)) # i的mask
    else: # subword list 長度大於等於50
        i = i1[:max_d] # 取前50個subwords(index) # !!!!須改方法!!!!
        input_mask = [1] * max_d # i的mask
    return i, np.asarray(input_mask)