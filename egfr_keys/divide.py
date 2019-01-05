import pickle

#these data will be save 
id_to_smiles = dict()
id_to_whole_activity = dict()
id_to_scaffold_activity = dict()

with open('smiles_id_scaffold.txt') as f : #this file contains information of id and corresponding smiles and scaffold
    lines = f.read().split('\n')[:-1]
    lines = [l.split() for l in lines]

    for l in lines:
        id_to_smiles[l[1]] = [l[0], l[2]] #l[0] smiles of whole molecule, l[2] : smiles of scaffold


with open('affinity_whole.txt') as f:
    lines = f.read().split('\n')[:-1]
    for l in lines:
        l = l.split()
        id_to_whole_activity[l[0].split('_')[1]] = [float(l[3]), float(l[4])]

with open('affinity_scaffold.txt') as f:
    lines = f.read().split('\n')[:-1]
    for l in lines:
        l = l.split()
        id_to_scaffold_activity[l[0].split('_')[1]] = [float(l[3]), float(l[4])]

#divide keys into train and test
common_keys = [k for k in id_to_smiles.keys() if k in id_to_whole_activity.keys() and k in id_to_scaffold_activity.keys()]
train_keys = common_keys[:len(common_keys)*3//4]
test_keys = common_keys[len(common_keys)*3//4:]

#divide train_keys and test_keys into active and inactive 
train_active_keys = [k for k in train_keys if id_to_whole_activity[k][0]>0.5]
train_inactive_keys = [k for k in train_keys if id_to_whole_activity[k][0]<0.5]
test_active_keys = [k for k in test_keys if id_to_whole_activity[k][0]>0.5]
test_inactive_keys = [k for k in test_keys if id_to_whole_activity[k][0]<0.5]
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
with open('id_to_whole_activity.pkl', 'wb') as f:
    pickle.dump(id_to_whole_activity, f)    
with open('id_to_scaffold_activity.pkl', 'wb') as f:
    pickle.dump(id_to_scaffold_activity, f)    
