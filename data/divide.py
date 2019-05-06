"""\
Prepare training and test sets, divided by active and inactive.
Command:
    $ python divide.py data.txt max min
"""
import pickle
import sys

data_txt, max_condition, min_condition = sys.argv[1:]
max_condition = float(max_condition)
min_condition = float(min_condition)

# Data to be saved:
id_to_condition1 = dict()
id_to_condition2 = dict()

def normalize(v, max_v, min_v):
    v = min(max_v, v)
    v = max(min_v, v)
    return (v-min_v)/(max_v-min_v)

with open(data_txt) as f:
    lines = f.read().split('\n')[:-1]
    lines = [l.split() for l in lines]
    for l in lines:
        id_to_condition1[l[0]] = normalize(float(l[1]), max_condition, min_condition)
        id_to_condition2[l[0]] = normalize(float(l[2]), max_condition, min_condition)

#divide keys into train and test
#common_keys = [k for k in id_to_smiles.keys() if k in id_to_condition1.keys() and k in id_to_condition2.keys()]
common_keys = list(id_to_condition1.keys())
train_keys = common_keys[:len(common_keys)*3//4]
test_keys = common_keys[len(common_keys)*3//4:]

#divide train_keys and test_keys into active and inactive 
train_active_keys = [k for k in train_keys if id_to_condition1[k] > 0.5]
train_inactive_keys = [k for k in train_keys if id_to_condition1[k] < 0.5]
test_active_keys = [k for k in test_keys if id_to_condition1[k] > 0.5]
test_inactive_keys = [k for k in test_keys if id_to_condition1[k] < 0.5]
print ('train active keys : ', len(train_active_keys))
print ('train inactive keys : ', len(train_inactive_keys))
print ('test active keys : ', len(test_active_keys))
print ('test inactive keys : ', len(test_inactive_keys))

#save keys and dictionary
with open('train_active_keys.pkl', 'wb') as f:
    pickle.dump(train_active_keys, f)
with open('train_inactive_keys.pkl', 'wb') as f:
    pickle.dump(train_inactive_keys, f)
with open('test_active_keys.pkl', 'wb') as f:
    pickle.dump(test_active_keys, f)
with open('test_inactive_keys.pkl', 'wb') as f:
    pickle.dump(test_inactive_keys, f)
