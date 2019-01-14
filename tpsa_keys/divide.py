import pickle
import numpy as np
#these data will be save 
id_to_smiles = dict()
id_to_condition1 = dict()
id_to_condition2 = dict()
def normalize(v, max_v, min_v):
    v = min(max_v, v)
    v = max(min_v, v)
    return (v-min_v)/(max_v-min_v)

with open('data.txt') as f : #this file contains information of id and corresponding smiles and scaffold
    lines = f.read().split('\n')[:-1]
    lines = [l.split('\t') for l in lines]
    max_condition = 150
    min_condition = 0
    for l in lines[:]:
        id_to_smiles[l[0]] = [l[1], l[2]] #l[1] smiles of whole molecule, l[2] : smiles of scaffold
        c1 = normalize(float(l[3]), max_condition, min_condition)
        c2 = normalize(float(l[4]), max_condition, min_condition)
        id_to_condition1[l[0]] = [c1, 1-c1] #l[1] smiles of whole molecule, l[2] : smiles of scaffold
        id_to_condition2[l[0]] = [c2, 1-c2] #l[1] smiles of whole molecule, l[2] : smiles of scaffold


#divide keys into train and test
common_keys = [k for k in id_to_smiles.keys() if k in id_to_condition1.keys() and k in id_to_condition2.keys()]
train_keys = common_keys[:len(common_keys)*3//4]
test_keys = common_keys[len(common_keys)*3//4:]

#divide train_keys and test_keys into active and inactive 
train_active_keys = [k for k in train_keys if id_to_condition1[k][0]>0.5]
train_inactive_keys = [k for k in train_keys if id_to_condition1[k][0]<0.5]
test_active_keys = [k for k in test_keys if id_to_condition1[k][0]>0.5]
test_inactive_keys = [k for k in test_keys if id_to_condition1[k][0]<0.5]
print ('train active keys : ', len(train_active_keys))
print ('train inactive keys : ', len(train_inactive_keys))
print ('test active keys : ', len(test_active_keys))
print ('test inactive keys : ', len(test_inactive_keys))


#save keys and dictionary
import pickle
with open('train_active_keys.pkl', 'wb') as f:
    pickle.dump(train_active_keys, f)
with open('train_inactive_keys.pkl', 'wb') as f:
    pickle.dump(train_inactive_keys, f)
with open('test_active_keys.pkl', 'wb') as f:
    pickle.dump(test_active_keys, f)
with open('test_inactive_keys.pkl', 'wb') as f:
    pickle.dump(test_inactive_keys, f)
with open('id_to_smiles.pkl', 'wb') as f:
    pickle.dump(id_to_smiles, f)    
with open('id_to_condition1.pkl', 'wb') as f:
    pickle.dump(id_to_condition1, f)    
with open('id_to_condition2.pkl', 'wb') as f:
    pickle.dump(id_to_condition2, f)    
