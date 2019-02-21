"""\
Prepare training and test sets, divided by active and inactive.
Command:
    $ python divide.py

NOTE that this script is slightly different from `../divide.py`.
"""
import pickle

# Data to be saved
id_to_condition1 = dict()
id_to_condition2 = dict()

with open('data_whole.txt') as f:
    for l in f:
        l = l.split()
        id_to_condition1[l[0].split('_')[1]] = float(l[3])

with open('data_scaffold.txt') as f:
    for l in f:
        l = l.split()
        id_to_condition2[l[0].split('_')[1]] = float(l[3])

#divide keys into train and test
common_keys = list( set(id_to_condition1.keys()) & set(id_to_condition2.keys()) )
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
