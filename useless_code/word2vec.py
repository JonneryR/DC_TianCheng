import pandas as pd
import numpy as np
from gensim.models import word2vec


train_maxlist = pd.read_csv('./feature/onehot_fea_train.csv',low_memory= False)
test_maxlist = pd.read_csv('./feature/onehot_fea_test.csv',low_memory= False)
data_maxlist = pd.concat([train_maxlist,test_maxlist],axis = 0)
op_device_list = ['op_version','op_device1','op_device2','op_device_code1','op_device_code2','op_device_code3']
op_ip_mac_list = ['op_mac1','op_mac2','op_ip1','op_ip2','op_wifi','op_geo_code','op_ip1_sub','op_ip2_sub']
tr_device_list = ['tr_device_code1','tr_device_code2','tr_device_code3','tr_device1','tr_device2']
tr_ip_mac_list = ['tr_ip1','tr_ip1_sub','tr_mac1','tr_geo_code']
tr_user_merchant = ['tr_merchant','tr_code1','tr_code2','tr_trans_type1','tr_trans_type2','tr_market_code','tr_market_type','tr_acc_id1','tr_acc_id2','tr_acc_id3']
data_maxlist['op_device'] = data_maxlist['op_version'].astype(str) +',' \
                + data_maxlist['op_device1'].astype(str) + ',' + data_maxlist['op_device2'].astype(str) + ',' \
                + data_maxlist['op_device_code1'].astype(str) + ',' + data_maxlist['op_device_code2'].astype(str) + ',' \
                + data_maxlist['op_device_code3'].astype(str) 
data_maxlist['op_ip_mac'] = data_maxlist['op_ip1'].astype(str) +',' \
                + data_maxlist['op_ip2'].astype(str) + ',' + data_maxlist['op_ip1_sub'].astype(str) + ',' \
                + data_maxlist['op_ip2_sub'].astype(str) + ',' + data_maxlist['op_mac1'].astype(str) + ',' \
                + data_maxlist['op_mac2'].astype(str) + ',' + data_maxlist['op_wifi'].astype(str) + ',' + data_maxlist['op_geo_code'].astype(str)
device = list(data_maxlist['op_device'].values)
ip_mac = list(data_maxlist['op_ip_mac'].values)

sentence_device = [s.split(',') for s in device]
sentence_ip_mac = [s.split(',') for s in ip_mac]
sentence_op = sentence_device + sentence_ip_mac
num_features = 20    # Word vector dimensionality
min_word_count = 5   # Minimum word count
num_workers = 16       # Number of threads to run in parallel
context = 2          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words


model_op = word2vec.Word2Vec(sentence_op, workers=num_workers,size=num_features, 
                          min_count = min_word_count, window = context, 
                          sg = 1, sample = downsampling)
op_array = np.zeros((len(data_maxlist),0))
for item in op_device_list+op_ip_mac_list:
    arr_list = list(data_maxlist[item].astype(str).values)
    arr = []
    example = arr_list[5]
    size1 = model_op[example].shape[0]
    for ele in arr_list:
        try:
            arr.append(model_op[ele])
        except KeyError:
            arr.append(np.random.normal(model_op[example].mean(), model_op[example].std(), (size1)))
            
    arr = np.array(arr)
    op_array = np.concatenate((op_array,arr),axis = 1)
    
    
data_maxlist['tr_device'] = data_maxlist['tr_device1'].astype(str) + ',' + data_maxlist['tr_device2'].astype(str) + ',' \
                + data_maxlist['tr_device_code1'].astype(str) + ',' + data_maxlist['tr_device_code2'].astype(str) + ',' \
                + data_maxlist['tr_device_code3'].astype(str) 
data_maxlist['tr_ip_mac'] = data_maxlist['tr_ip1'].astype(str) +',' \
                + data_maxlist['tr_ip1_sub'].astype(str) + ',' \
                + data_maxlist['tr_mac1'].astype(str) + ',' \
                + data_maxlist['tr_geo_code'].astype(str)

data_maxlist['tr_user_merchant'] = data_maxlist['tr_merchant'].astype(str) +',' + data_maxlist['tr_code1'].astype(str) +',' \
                    + data_maxlist['tr_code2'].astype(str) +',' + data_maxlist['tr_trans_type1'].astype(str) +','  \
                    + data_maxlist['tr_trans_type2'].astype(str)+',' + data_maxlist['tr_market_code'].astype(str)+','\
                    + data_maxlist['tr_market_type'].astype(str)+',' + data_maxlist['tr_acc_id1'].astype(str)+',' \
                    + data_maxlist['tr_acc_id2'].astype(str)+',' + data_maxlist['tr_acc_id3'].astype(str)



device = list(data_maxlist['tr_device'].values)
ip_mac = list(data_maxlist['tr_ip_mac'].values)
user_merchant = list(data_maxlist['tr_user_merchant'].values)

sentence_device = [s.split(',') for s in device]
sentence_ip_mac = [s.split(',') for s in ip_mac]
sentence_user_merchant = [s.split(',') for s in user_merchant]
sentence_tr = sentence_device + sentence_ip_mac + sentence_user_merchant
model_tr = word2vec.Word2Vec(sentence_tr, workers=num_workers,size=num_features, 
                          min_count = min_word_count, window = context, 
                          sg = 1, sample = downsampling)


tr_array = np.zeros((len(data_maxlist),0))
for item in tr_device_list+tr_ip_mac_list + tr_user_merchant:
    arr_list = list(data_maxlist[item].astype(str).values)
    arr = []
    example = data_maxlist['tr_device_code2'][0]
    size1 = model_tr['BKL-AL00'].shape[0]
    for ele in arr_list:
        try:
            arr.append(model_tr[ele])
        except KeyError:
            arr.append(np.random.normal(model_tr['BKL-AL00'].mean(), model_tr['BKL-AL00'].std(), (size1)))
    arr = np.array(arr)
    tr_array = np.concatenate((tr_array,arr),axis = 1)
    

final_array = np.concatenate((tr_array,op_array),axis = 1)
train_array = final_array[:len(train_maxlist)]
test_array = final_array[len(train_maxlist):]
np.save('./feature/train_id_w2v',train_array)
np.save('./feature/test_id_w2v',test_array)




'''
##get 5 kinds of word2vec

op_train = pd.read_csv('./data/operation_train_new.csv',low_memory= False)
trans_train = pd.read_csv('./data/transaction_train_new.csv',low_memory= False)

op_test = pd.read_csv('./data/test_operation_round2.csv',low_memory= False)
trans_test = pd.read_csv('./data/test_transaction_round2.csv',low_memory= False)
y = pd.read_csv('./data/tag_train_new.csv',low_memory= False)
sub = pd.read_csv('./data/sub.csv',low_memory= False)


op = pd.concat([op_train,op_test],axis = 0)
trans = pd.concat([trans_train,trans_test],axis = 0)
op['device'] = op['version'].astype(str) +',' \
                + op['device1'].astype(str) + ',' + op['device2'].astype(str) + ',' \
                + op['device_code1'].astype(str) + ',' + op['device_code2'].astype(str) + ',' \
                + op['device_code3'].astype(str) 
op['ip_mac'] = op['ip1'].astype(str) +',' \
                + op['ip2'].astype(str) + ',' + op['ip1_sub'].astype(str) + ',' \
                + op['ip2_sub'].astype(str) + ',' + op['mac1'].astype(str) + ',' \
                + op['mac2'].astype(str) + ',' + op['wifi'].astype(str) + ',' + op['geo_code'].astype(str)
device = list(op['device'].values)
ip_mac = list(op['ip_mac'].values)

sentence_device = [s.split(',') for s in device]
sentence_ip_mac = [s.split(',') for s in ip_mac]
num_features = 20    # Word vector dimensionality
min_word_count = 5   # Minimum word count
num_workers = 16       # Number of threads to run in parallel
context = 2          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words


model_device = word2vec.Word2Vec(sentence_device, workers=num_workers,size=num_features, 
                          min_count = min_word_count, window = context, 
                          sg = 1, sample = downsampling)
model_device.save('./feature/model_device')
model_ip_mac = word2vec.Word2Vec(sentence_ip_mac, workers=num_workers,size=num_features, 
                          min_count = min_word_count, window = context, 
                          sg = 1, sample = downsampling)
model_ip_mac.save('./feature/model_ip_mac')

trans['tr_device'] = trans['device1'].astype(str) + ',' + trans['device2'].astype(str) + ',' \
                + trans['device_code1'].astype(str) + ',' + trans['device_code2'].astype(str) + ',' \
                + trans['device_code3'].astype(str) 

trans['tr_ip_mac'] = trans['ip1'].astype(str) +',' \
                + trans['ip1_sub'].astype(str) + ',' \
                + trans['mac1'].astype(str) + ',' \
                + trans['geo_code'].astype(str)

trans['user_merchant'] = trans['merchant'].astype(str) +',' + trans['code1'].astype(str) +',' \
                    + trans['code2'].astype(str) +',' + trans['trans_type1'].astype(str) +','  \
                    + trans['trans_type2'].astype(str)+',' + trans['market_code'].astype(str)+','\
                    + trans['market_type'].astype(str)+',' + trans['acc_id1'].astype(str)+',' \
                    + trans['acc_id2'].astype(str)+',' + trans['acc_id3'].astype(str)

tr_device = list(trans['tr_device'].values)
tr_ip_mac = list(trans['tr_ip_mac'].values)
user_merchant = list(trans['user_merchant'].values)

sentence_tr_device = [s.split(',') for s in tr_device]
sentence_tr_ip_mac = [s.split(',') for s in tr_ip_mac]
sentence_user_merchant = [s.split(',') for s in user_merchant]



model_tr_device = word2vec.Word2Vec(sentence_tr_device, workers=num_workers,size=num_features, 
                          min_count = min_word_count, window = context, 
                          sg = 1, sample = downsampling)
model_tr_device.save('./feature/model_tr_device')

model_tr_ip_mac = word2vec.Word2Vec(sentence_tr_ip_mac, workers=num_workers,size=num_features, 
                          min_count = min_word_count, window = context, 
                          sg = 1, sample = downsampling)
model_tr_ip_mac.save('./feature/model_tr_ip_mac')

model_user_merchant = word2vec.Word2Vec(sentence_user_merchant, workers=num_workers,size=num_features, 
                          min_count = min_word_count, window = context, 
                          sg = 1, sample = downsampling)
model_user_merchant.save('./feature/model_user_merchant')
'''