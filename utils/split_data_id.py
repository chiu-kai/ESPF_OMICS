# split_data_id
import numpy as np
import random

#啓唐學長的
def data_split(IC, split_base, seed):
    unique_ids = data[split_base].unique()
    
    train_ids, val_test_ids = train_test_split(unique_ids, test_size=0.2, random_state=seed)
    val_ids, test_ids = train_test_split(val_test_ids, test_size=0.5, random_state=seed)
    
    train_set = data[data[split_base].isin(train_ids)]
    val_set = data[data[split_base].isin(val_ids)]
    test_set = data[data[split_base].isin(test_ids)]
    return train_set, val_set, test_set
# data = pd.read_csv(rpath+'/sorted_IC50_82833_580_170.csv')
# train_set, val_set, test_set = data_split(data, 'DepMap_ID', args)


class split():
    def __init__(self):
        self.test = 1
    def repeat():
        return 
    def kfold():
        return 
    def byCCL():
        return 
    def byDrug():  
        return 


# version3
def repeat_func(id_unrepeat, repeatNum, setname=''):
    id_dataset = np.arange(0, repeatNum) + id_unrepeat[0]*repeatNum   
    for y in id_unrepeat:
        id_dataset = np.union1d(id_dataset, np.arange(0, repeatNum) + y*repeatNum)
    print(f"id_{setname}.shape",id_dataset.shape) 
    return id_dataset

def split_id(num_ccl,num_drug , splitType, kfoldCV ,repeat):
    random.seed(42)
    if splitType in ['byCCL', 'ModelID']:
        sampleNum = num_ccl
        repeatNum = num_drug
    elif splitType in ['byDrug', 'drug_name']:
        sampleNum = num_drug
        repeatNum = num_ccl
        
    all_unrepeat_ids = list(range(0, sampleNum))
    id_unrepeat_train = random.sample(all_unrepeat_ids, int(sampleNum*0.80))
    id_unrepeat_val_test = list(set(all_unrepeat_ids) - set(id_unrepeat_train))
    id_unrepeat_val = random.sample(id_unrepeat_val_test, int(0.50 * len(id_unrepeat_val_test)))
    id_unrepeat_test = list(set(id_unrepeat_val_test) - set(id_unrepeat_val))
    id_unrepeat_train_val = list(set(all_unrepeat_ids) - set(id_unrepeat_test))

    print("id_unrepeat_train",np.shape(id_unrepeat_train))
    print("id_unrepeat_val",np.shape(id_unrepeat_val))
    print("id_unrepeat_test",np.shape(id_unrepeat_test))
    print("id_unrepeat_train_val",np.shape(id_unrepeat_train_val))

    if kfoldCV > 1: # for kfold test, and train_val split by kfold.split function   
        return id_unrepeat_test, id_unrepeat_train_val
    else :
        if kfoldCV == 1 and (repeat is True): 
            id_train = repeat_func(id_unrepeat_train, repeatNum, setname='train')
            id_val = repeat_func(id_unrepeat_val, repeatNum, setname='val')
            id_test = repeat_func(id_unrepeat_test, repeatNum, setname='test')
            return id_unrepeat_train, id_unrepeat_val, id_unrepeat_test, id_unrepeat_train_val, id_train, id_val, id_test #, id_train_val
        elif repeat is False: # for only mut or only exp (no drug)
            return id_unrepeat_train, id_unrepeat_val, id_unrepeat_test, id_unrepeat_train_val
        else:
            print("Wrong assignment for repeat")

# version2
'''
def repeat_func(id_unrepeat, repeatNum, setname=''):
    id_dataset = np.arange(0, repeatNum) + id_unrepeat[0]*repeatNum   
    for y in id_unrepeat:
        id_dataset = np.union1d(id_dataset, np.arange(0, repeatNum) + y*repeatNum)
    print(f"id_{setname}.shape",id_dataset.shape) 
    return id_dataset

def split_id(num_ccl,num_drug , splitType ,repeat, kFold):
    random.seed(42)
    if splitType in ['byCCL', 'ModelID']:
        all_unrepeat_ids = list(range(0, num_ccl))
        id_unrepeat_test = random.sample(all_unrepeat_ids, int(num_ccl*0.10))
        id_unrepeat_train_val = list(set(all_unrepeat_ids) - set(id_unrepeat_test))

        id_unrepeat_val = random.sample(id_unrepeat_train_val, int(0.10 * len(id_unrepeat_train_val)))
        id_unrepeat_train = list(set(id_unrepeat_train_val) - set(id_unrepeat_val))
        
        print("id_unrepeat_train",np.shape(id_unrepeat_train))
        print("id_unrepeat_val",np.shape(id_unrepeat_val))
        print("id_unrepeat_test",np.shape(id_unrepeat_test))
        print("id_unrepeat_train_val",np.shape(id_unrepeat_train_val))

        if kFold is True:# for kfold test, and train_val split by kfold.split function   
            return id_unrepeat_test, id_unrepeat_train_val
        else : 
            if repeat is True:
                id_train = repeat_func(id_unrepeat_train, num_drug, setname='train')
                id_val = repeat_func(id_unrepeat_val, num_drug, setname='val')
                id_test = repeat_func(id_unrepeat_test, num_drug, setname='test')
                return id_unrepeat_train, id_unrepeat_val, id_unrepeat_test, id_unrepeat_train_val, id_train, id_val, id_test #, id_train_val
            else: # for only mut or only exp    
                return id_unrepeat_train, id_unrepeat_val, id_unrepeat_test, id_unrepeat_train_val
    elif splitType in ['byDrug', 'drug_name']:
        all_unrepeat_ids = list(range(0, num_drug))
        id_unrepeat_test = random.sample(all_unrepeat_ids, int(num_drug*0.10))
        id_unrepeat_train_val = list(set(all_unrepeat_ids) - set(id_unrepeat_test))
        
        id_unrepeat_val = random.sample(id_unrepeat_train_val, int(0.10 * len(id_unrepeat_train_val)))
        id_unrepeat_train = list(set(id_unrepeat_train_val) - set(id_unrepeat_val))
        
        print("id_unrepeat_train",np.shape(id_unrepeat_train))
        print("id_unrepeat_val",np.shape(id_unrepeat_val))
        print("id_unrepeat_test",np.shape(id_unrepeat_test))
        print("id_unrepeat_train_val",np.shape(id_unrepeat_train_val))

        if kFold is True: # for kfold test, and train_val split by kfold.split function   
            return id_unrepeat_test, id_unrepeat_train_val
        else :
            if repeat is True:
                id_train = repeat_func(id_unrepeat_train, num_ccl, setname='train')
                id_val = repeat_func(id_unrepeat_val, num_ccl, setname='val')
                id_test = repeat_func(id_unrepeat_test, num_ccl, setname='test')
                return id_unrepeat_train, id_unrepeat_val, id_unrepeat_test, id_unrepeat_train_val, id_train, id_val, id_test #, id_train_val
            else: # for only mut or only exp    
                return id_unrepeat_train, id_unrepeat_val, id_unrepeat_test, id_unrepeat_train_val
'''


# version1

def split_byCCL(num_ccl,num_drug,repeat):
    import numpy as np
    random.seed(42)
    all_cell_ids = list(range(0, num_ccl))
    id_cell_test = random.sample(all_cell_ids, int(num_ccl*0.10))
    id_cell_train_val = list(set(all_cell_ids) - set(id_cell_test))
    
    id_cell_val = random.sample(id_cell_train_val, int(0.10 * len(id_cell_train_val)))
    id_cell_train = list(set(id_cell_train_val) - set(id_cell_val))
    
    print("id_cell_train",np.shape(id_cell_train))
    print("id_cell_val",np.shape(id_cell_val))
    print("id_cell_test",np.shape(id_cell_test))
    print("id_cell_train_val",np.shape(id_cell_train_val))
    # prepare sample indices (selected CCLs x 1450Drug)
    id_train = np.arange(0, num_drug) + id_cell_train[0]*num_drug
    for y in id_cell_train:
        id_train = np.union1d(id_train, np.arange(0, num_drug) + y*num_drug)
    print("id_train.shape",id_train.shape) 
    
    id_val = np.arange(0, num_drug) + id_cell_val[0]*num_drug
    for y in id_cell_val:
        id_val = np.union1d(id_val, np.arange(0, num_drug) + y*num_drug)
    print("id_val.shape",id_val.shape) 
    
    id_test = np.arange(0, num_drug) + id_cell_test[0] * num_drug
    for y in id_cell_test:
        id_test = np.union1d(id_test, np.arange(0, num_drug) + y*num_drug)
    print("id_test.shape",id_test.shape) 
    
    id_train_val = list(set(id_train).union(id_val))
    print("id_train_val.shape",np.shape(id_train_val))

    if repeat is True:
        return id_cell_train, id_cell_val, id_cell_test, id_cell_train_val, id_train, id_val, id_test, id_train_val
    else: # for only mut or only exp    
        return id_cell_train, id_cell_val, id_cell_test, id_cell_train_val

def split_byDrug(num_ccl,num_drug,repeat,kfold):
    import numpy as np
    random.seed(42)
    all_drug_ids = list(range(0, num_drug))
    id_drug_test = random.sample(all_drug_ids, int(num_drug*0.10))
    id_drug_train_val = list(set(all_drug_ids) - set(id_drug_test))
    
    id_drug_val = random.sample(id_drug_train_val, int(0.10 * len(id_drug_train_val)))
    id_drug_train = list(set(id_drug_train_val) - set(id_drug_val))
    
    print("id_drug_train",np.shape(id_drug_train))
    print("id_drug_val",np.shape(id_drug_val))
    print("id_drug_test",np.shape(id_drug_test))
    print("id_drug_train_val",np.shape(id_drug_train_val))
    
    # prepare sample indices (selected CCLs x 1450Drug)
    id_train = np.arange(0, num_ccl) + id_drug_train[0]*num_ccl
    for y in id_drug_train:
        id_train = np.union1d(id_train, np.arange(0, num_ccl) + y*num_ccl)
    print("id_train.shape",id_train.shape) 
    
    id_validation = np.arange(0, num_ccl) + id_drug_val[0]*num_ccl
    for y in id_drug_val:
        id_validation = np.union1d(id_validation, np.arange(0, num_ccl) + y*num_ccl)
    print("id_validation.shape",id_validation.shape) 
    
    id_test = np.arange(0, num_ccl) + id_drug_test[0] * num_ccl
    for y in id_drug_test:
        id_test = np.union1d(id_test, np.arange(0, num_ccl) + y*num_ccl)
    print("id_test.shape",id_test.shape) 
    
    # id_train_val = list(set(id_train).union(id_validation))
    # print("id_train_val.shape",np.shape(id_train_val)) 
    
    if repeat is True:
        return id_drug_train, id_drug_val, id_drug_test, id_drug_train_val, id_train, id_validation, id_test#, id_train_val
    else: # for only drug  
        return id_drug_train, id_drug_val, id_drug_test#, id_drug_train_val