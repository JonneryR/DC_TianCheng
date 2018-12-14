#encoding=utf8
import lightgbm as lgb
import pandas as pd
import numpy as np
import gc,time,datetime
from util import *
from sklearn.model_selection import KFold, cross_val_score, train_test_split, StratifiedKFold
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import log_loss


op_train = pd.read_csv('./data/operation_train_new.csv',low_memory= False)
trans_train = pd.read_csv('./data/transaction_train_new.csv',low_memory= False)

op_test = pd.read_csv('./data/test_operation_round2.csv',low_memory= False)
trans_test = pd.read_csv('./data/test_transaction_round2.csv',low_memory= False)
label = pd.read_csv('./data/tag_train_new.csv',low_memory= False)
sub = pd.read_csv('./data/sub.csv',low_memory= False)
y = label.drop(['Tag'], axis = 1)
sub = sub.drop(['Tag'], axis = 1)


###
for datas in [op_train,op_test]:
	datas.drop(['ip2','ip2_sub'],axis = 1,inplace = True)

for datas in [trans_train,trans_test]:
	datas.drop(['code1','code2'],axis = 1,inplace = True)
	
###without doing drop_duplicates, get hour and hourseg
for datas in [op_train,trans_train,op_test,trans_test]:
	datas['hour'] = pd.to_datetime(datas['time']).apply(lambda x:x.hour)
	datas['hourSeg'] = datas['hour'].apply(hourgetSeg)
	datas['device2'] = datas['device2'].astype(str).apply(deal_make)


###这里给他了一个相对时间这样好算时间差
result = timestamp(op_train)
result = timestamp(trans_train)
result = timestamp2(op_test)
result = timestamp2(trans_test)

'''
train = get_feature(op_train,trans_train,y).fillna(-1)
test = get_feature(op_test,trans_test,sub).fillna(-1)

train.to_csv('./feature/main_fea_train.csv')
test.to_csv('./feature/main_fea_test.csv')
'''


train_cross = get_cross(op_train,trans_train,y).fillna(-1)
test_cross = get_cross(op_test,trans_test,sub).fillna(-1)

np.save('./feature/train_cross',train_cross.values)
np.save('./feature/test_cross',test_cross.values)

train_ratio = get_ratio(op_train,trans_train,y).fillna(-1)
test_ratio = get_ratio(op_test,trans_test,sub).fillna(-1)

np.save('./feature/train_ratio',train_ratio.values)
np.save('./feature/test_ratio',test_ratio.values)


train_merchant = get_merchant_feature(trans_train,y).fillna(-1)
test_merchant = get_merchant_feature(trans_test,sub).fillna(-1)

train_merchant.to_csv('./feature/merchant_fea_train.csv')
test_merchant.to_csv('./feature/merchant_fea_test.csv')



train_time = get_time_features(op_train,trans_train,y).fillna(-1)
test_time = get_time_features(op_test,trans_test,sub).fillna(-1)

train_time.to_csv('./feature/time_fea_train.csv')
test_time.to_csv('./feature/time_fea_test.csv')

train_maxlist = get_most_list(op_train,trans_train,y).fillna(-1)
test_maxlist = get_most_list(op_test,trans_test,sub).fillna(-1)

train_maxlist.to_csv('./feature/onehot_fea_train.csv')
test_maxlist.to_csv('./feature/onehot_fea_test.csv')

train_new = get_new_feature(op_train,trans_train,y).fillna(-1)
test_new = get_new_feature(op_test,trans_test,sub).fillna(-1)

train_new.to_csv('./feature/new_main_fea_train.csv')
test_new.to_csv('./feature/new_main_fea_test.csv')

tr_train,tr_test = get_onehot_features(trans_train,trans_test,y,sub)
op_train,op_test = get_onehot_features(op_train,op_test,y,sub)

train_countvector =  sparse.hstack((tr_train, op_train), 'csr','int')
test_countvector = sparse.hstack((tr_test, op_test), 'csr','int')

sparse.save_npz('./feature/train_countvector.npz', train_countvector) 
sparse.save_npz('./feature/test_countvector.npz', test_countvector) 


train_all_time_fea = get_all_time_fea(op_train,trans_train,y).fillna(-1)
test_all_time_fea = get_all_time_fea(op_test,trans_test,sub).fillna(-1)

np.save('./feature/train_all_time_fea',train_all_time_fea.values)
np.save('./feature/test_all_time_fea',test_all_time_fea.values)

train_time_cross_nunique = get_time_cross_nunique(op_train,trans_train,y).fillna(-1)
test_time_cross_nunique = get_time_cross_nunique(op_test,trans_test,sub).fillna(-1)

np.save('./feature/train_time_cross_nunique',train_time_cross_nunique.values)
np.save('./feature/test_time_cross_nunique',test_time_cross_nunique.values)


print('All features get!!!')