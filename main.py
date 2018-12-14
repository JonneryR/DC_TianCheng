#encoding=utf8
import lightgbm as lgb
import pandas as pd
import numpy as np
import gc,time,datetime
from util import *
from sklearn.model_selection import KFold, cross_val_score, train_test_split, StratifiedKFold
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import log_loss
label = pd.read_csv('./data/tag_train_new.csv',low_memory = False)
sub = pd.read_csv('./data/sub.csv',low_memory= False)
label = label.Tag.values
sub = sub[['UID']]

train = pd.read_csv('./feature/new_main_fea_train.csv')
test = pd.read_csv('./feature/new_main_fea_test.csv')

train_countvector = sparse.load_npz('./feature/train_countvector.npz')
test_countvector = sparse.load_npz('./feature/test_countvector.npz')

train_merchant = pd.read_csv('./feature/merchant_fea_train.csv')
test_merchant = pd.read_csv('./feature/merchant_fea_test.csv')

train_time = pd.read_csv('./feature/time_fea_train.csv')
test_time = pd.read_csv('./feature/time_fea_test.csv')

train_maxlist = pd.read_csv('./feature/onehot_fea_train.csv')
test_maxlist = pd.read_csv('./feature/onehot_fea_test.csv')

train_npmerchant = np.load('./feature/train_uid_merchant.npy')
test_npmerchant = np.load('./feature/test_uid_merchant.npy')

'''works very well to get rid of it
train_w2v = np.load('./feature/train_id_w2v.npy')
test_w2v = np.load('./feature/test_id_w2v.npy')
'''
##train_cross = np.load('./feature/train_cross.npy')
##test_cross = np.load('./feature/test_cross.npy')

'''
train_cross_nunique = np.load('./feature/train_cross_nunique.npy')
test_cross_nunique = np.load('./feature/test_cross_nunique.npy')
'''

train_time_cross_nunique = np.load('./feature/train_time_cross_nunique.npy')
test_time_cross_nunique = np.load('./feature/test_time_cross_nunique.npy')


train_ratio = np.load('./feature/train_ratio.npy')
test_ratio = np.load('./feature/test_ratio.npy')

'''
train_stack = np.load('./feature/train_stack.npy')
test_stack = np.load('./feature/test_stack.npy')
'''

train_all_time_fea = np.load('./feature/train_all_time_fea.npy')
test_all_time_fea = np.load('./feature/test_all_time_fea.npy')

train_conver_rate = np.load('./feature/train_conver_rate.npy')
test_conver_rate = np.load('./feature/test_conver_rate.npy')

data = pd.concat([train_maxlist,test_maxlist],axis = 0)
base_train_csr = sparse.csr_matrix((len(train_maxlist), 0))
base_test_csr = sparse.csr_matrix((len(test_maxlist), 0))

enc = OneHotEncoder()
for feature in data.columns:
	if (data[feature].dtype == 'object') and (feature!= 'time'):
	    enc.fit(data[feature].astype(str).values.reshape(-1, 1))
	    base_train_csr = sparse.hstack((base_train_csr, enc.transform(train_maxlist[feature].astype(str).values.reshape(-1, 1))), 'csr','bool')
	    base_test_csr = sparse.hstack((base_test_csr, enc.transform(test_maxlist[feature].astype(str).values.reshape(-1, 1))),'csr','bool')

for fea in [train_time_cross_nunique,train_ratio,train_countvector,train,train_merchant,train_npmerchant,train_all_time_fea,train_time,train_conver_rate]:#,
	base_train_csr = sparse.hstack((base_train_csr, fea), 'csr', 'float32')
	gc.collect()
for fea in [test_time_cross_nunique,test_ratio,test_countvector,test,test_merchant,test_npmerchant,test_all_time_fea,test_time,test_conver_rate]:#,
	base_test_csr = sparse.hstack((base_test_csr, fea), 'csr', 'float32')
	gc.collect()

for random_seed in [1000,2018,3096,4088,5000,6666,7668,8888]:
	result_lgb = lgb_seed_cv(base_train_csr,label,base_test_csr,sub,random_seed)
	result_xgb = xgb_seed_cv(base_train_csr,label,base_test_csr,sub,random_seed)
print('data shape::','train.shape:',base_train_csr.shape,'test.shape',base_test_csr.shape)



'''
sub['Tag'] = (result_lgb['Tag'] + result_xgb['Tag'])/2
sub.to_csv("./submit/xgb+lgb.csv", index=False)
'''