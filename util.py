#encoding=utf8
import lightgbm as lgb
import pandas as pd
import numpy as np
import gc,time,datetime
from sklearn.model_selection import KFold, cross_val_score, train_test_split, StratifiedKFold
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from xgboost import XGBClassifier
import catboost as cb
def deal_make(data):
    if 'VIVO' in data:
        return 'OPPO'
    elif 'IPHONE' in data:
        return 'APPLE'
    elif 'OPPO' in data:
        return 'OPPO'
    elif 'PAC' in data:
        return 'OPPO'
    elif 'MI' in data:
        return 'XiaoMi'
    elif 'REDMI' in data:
        return 'XiaoMi'
    elif 'HUAWEI' in data:
        return 'HUAWEI'
    elif 'MHA' in data:
        return 'HUAWEI'
    elif 'PRA' in data:
        return 'HUAWEI'
    elif 'FRD' in data:
        return 'HUAWEI'
    elif 'ALP' in data:
        return 'HUAWEI'
    elif 'Apple' in data:
        return 'APPLE'
    elif 'Lenovo' in data:
        return 'Lenovo'
    elif 'iPad' in data:
        return 'APPLE'
    elif 'Meizu' in data:
        return 'Meizu'
    elif 'SAMSUNG' in data:
        return 'SAMSUNG'
    elif 'samsung' in data:
        return 'SAMSUNG'
    elif 'Philips' in data:
        return 'Philips'
    elif 'Hisense' in data:
        return 'Hisense'
    else:
        return data

op_delete_col = ['device_code3','mac1','ip2','ip2_sub']
tr_delete_col = ['code1','code2','acc_id2','acc_id3','market_code','market_type']
def delete_col(data,delete_cols):
    new_data = data.copy()
    for col in delete_cols:
        new_data = new_data.drop([col],axis = 1)
    gc.collect()
    return new_data
'''
train_op = delete_col(train_op,op_delete_col)
train_tr = delete_col(train_tr,tr_delete_col)
test_op = delete_col(test_op,op_delete_col)
test_tr = delete_col(test_tr,tr_delete_col)
'''

def hourgetSeg(x):
    if x >=0 and x<4:
        return 1
    elif x>=4 and x<8:
        return 2
    elif x>=8 and x<12:
        return 3
    elif x>=12 and x<16:
        return 4
    elif x>=16 and x<20:
        return 5
    elif x>=20 and x<24:
        return 6


def max_list(data):
    lt = []
    for item in data:
        lt.append(item)
    temp = 0
    for i in lt:
        if lt.count(i) > temp:
            max_str = i
            temp = lt.count(i)
    return max_str

def get_change_count(data):
    new_list = []
    count = 0
    for item in data:
        new_list.append(item)
    for i in range(len(new_list)-1):
        if(new_list[i]!=new_list[i+1]):
            count +=1
    return count


def get_time_mean(data):
    time_list = []
    for item in data:
        time_list.append(item)
    time_delta = []
    if(len(time_list)>1):
        for i in range(len(time_list)-1):
            time_delta.append((time_list[i+1]-time_list[i]))
        time_delta = np.array(time_delta)
        return np.mean(time_delta)
    else:
        return 0

def get_time_var(data):
    time_list = []
    for item in data:
        time_list.append(item)
    time_delta = []
    if(len(time_list)>1):
        for i in range(len(time_list)-1):
            time_delta.append((time_list[i+1]-time_list[i]))
        time_delta = np.array(time_delta)
        return np.var(time_delta)
    else:
        return 0

def get_10min_count(data):
    time_list = []
    if(len(data)>=1):
        for item in data:
            if(item + 600*1000 > max(data)):
                time_list.append(item)
        return len(time_list)
    else:
        return 0
    
def get_5min_count(data):
    time_list = []
    if(len(data)>=1):
        for item in data:
            if(item + 300*1000 > max(data)):
                time_list.append(item)
        return len(time_list)
    else:
        return 0

'''
def get_feature(op,trans,label):
    op = op.groupby(['UID']).size().reset_index().rename(columns = {0:'UID_op_size'})
    tr = trans.groupby(['UID']).size().reset_index().rename(columns = {0:'UID_tr_size'})
    label = label.merge(op,on = ['UID'],how = 'left').fillna(0)
    label = label.merge(tr,on = ['UID'],how = 'left').fillna(0)
    for feature in op.columns[2:]:
        label =feat_count(label,op,['UID'],feature,'op')
        label =feat_nunique(label,op,['UID'],feature,'op')
    
    for feature in trans.columns[2:]:
        if trans[feature].dtype == 'object':
            label =feat_count(label,trans,['UID'],feature)
            label =feat_nunique(label,trans,['UID'],feature)
        else:
            label =feat_count(label,trans,['UID'],feature)
            label =feat_nunique(label,trans,['UID'],feature)
            label =feat_max(label,trans,['UID'],feature)
            label =feat_min(label,trans,['UID'],feature)
            label =feat_sum(label,trans,['UID'],feature)
            label =feat_mean(label,trans,['UID'],feature)
            label =feat_std(label,trans,['UID'],feature)
    label = label.drop(['UID'],axis = 1)
    gc.collect()
    return label
'''
def get_cross(op,trans,label):
    op_cross_1 = ['mode','version','os','device1','device_code1']
    op_cross_2 = ['ip1','mac1','device_code3','device2','mac2','wifi','geo_code','ip1_sub']
    op_cross_feature = []
    for feat_1 in op_cross_1:
        for feat_2 in op_cross_2:
            col_name = "cross_" + feat_1 + "_and_" + feat_2
            op_cross_feature.append(col_name)
            op[col_name] = op[feat_1].astype(str).values + '_' + op[feat_2].astype(str).values
    for col in op_cross_feature:
        temp = op[['UID',col]].merge(op.groupby([col]).size().reset_index().rename(columns = {0:'size_%s'%col}),on =col, how = 'left')[['UID','size_%s'%col]]
        temp1 = temp.groupby(['UID'])['size_%s'%col].mean().reset_index()
        label = label.merge(temp1, on = 'UID',how = 'left')
        temp1 = temp.groupby(['UID'])['size_%s'%col].sum().reset_index()
        label = label.merge(temp1, on = 'UID',how = 'left')
        temp1 = temp.groupby(['UID'])['size_%s'%col].max().reset_index()
        label = label.merge(temp1, on = 'UID',how = 'left')
        temp1 = temp.groupby(['UID'])['size_%s'%col].min().reset_index()
        label = label.merge(temp1, on = 'UID',how = 'left')
        gc.collect()
        del temp,temp1
            
    tr_cross_1 = ['channel','amt_src1','merchant','trans_type1','device1','device_code1']
    tr_cross_2 = ['acc_id1','trans_type2','device_code3','device2','mac1','ip1','market_code','market_type','ip1_sub','amt_src2']
    tr_cross_feature = []
    for feat_1 in tr_cross_1:
        for feat_2 in tr_cross_2:
            col_name = "cross_" + feat_1 + "_and_" + feat_2
            tr_cross_feature.append(col_name)
            trans[col_name] = trans[feat_1].astype(str).values + '_' + trans[feat_2].astype(str).values
    for col in tr_cross_feature:
        temp = trans[['UID',col]].merge(trans.groupby([col]).size().reset_index().rename(columns = {0:'size_%s'%col}),on =col, how = 'left')[['UID','size_%s'%col]]
        temp1 = temp.groupby(['UID'])['size_%s'%col].mean().reset_index()
        label = label.merge(temp1, on = 'UID',how = 'left')
        temp1 = temp.groupby(['UID'])['size_%s'%col].sum().reset_index()
        label = label.merge(temp1, on = 'UID',how = 'left')
        temp1 = temp.groupby(['UID'])['size_%s'%col].max().reset_index()
        label = label.merge(temp1, on = 'UID',how = 'left')
        temp1 = temp.groupby(['UID'])['size_%s'%col].min().reset_index()
        label = label.merge(temp1, on = 'UID',how = 'left')
        gc.collect()
        del temp,temp1
    print('cross done!')
    label = label.drop(['UID'],axis = 1)
    return label

def get_time_cross_nunique(op,trans,label):
    op_cross_1 = ['mode','version','os','device1','device_code1']
    op_cross_2 = ['ip1','mac1','device_code3','device2','mac2','wifi','geo_code','ip1_sub']
    op_cross_feature = []
    for feat_1 in op_cross_1:
        for feat_2 in op_cross_2:
            col_name = "cross_" + feat_1 + "_and_" + feat_2
            op_cross_feature.append(col_name)
            op[col_name] = op[feat_1].astype(str).values + '_' + op[feat_2].astype(str).values
    for col in op_cross_feature:
        temp = op[['UID',col]].merge(op.groupby([col]).size().reset_index().rename(columns = {0:'size_%s'%col}),on =col, how = 'left')[['UID','size_%s'%col]]
        temp1 = temp.groupby(['UID'])['size_%s'%col].mean().reset_index()
        label = label.merge(temp1, on = 'UID',how = 'left')
        temp1 = temp.groupby(['UID'])['size_%s'%col].sum().reset_index()
        label = label.merge(temp1, on = 'UID',how = 'left')
        temp1 = temp.groupby(['UID'])['size_%s'%col].max().reset_index()
        label = label.merge(temp1, on = 'UID',how = 'left')
        temp1 = temp.groupby(['UID'])['size_%s'%col].min().reset_index()
        label = label.merge(temp1, on = 'UID',how = 'left')
        
        label = label.merge(op.groupby(['UID'])[col].count().reset_index(),on = 'UID',how = 'left')
        label = label.merge(op.groupby(['UID'])[col].nunique().reset_index(),on = 'UID',how = 'left')
        
        for fea in ['day','hour','hourSeg']:
            temp = op[['UID',col]].merge(op.groupby([col,fea]).size().reset_index().rename(columns = {0:'size_%s'%col}),on =col, how = 'left')[['UID','size_%s'%col]]
            temp1 = temp.groupby(['UID'])['size_%s'%col].mean().reset_index()
            label = label.merge(temp1, on = 'UID',how = 'left')
            temp1 = temp.groupby(['UID'])['size_%s'%col].sum().reset_index()
            label = label.merge(temp1, on = 'UID',how = 'left')
            temp1 = temp.groupby(['UID'])['size_%s'%col].max().reset_index()
            label = label.merge(temp1, on = 'UID',how = 'left')
            temp1 = temp.groupby(['UID'])['size_%s'%col].min().reset_index()
            label = label.merge(temp1, on = 'UID',how = 'left')

        gc.collect()
        del temp,temp1
            
    tr_cross_1 = ['channel','amt_src1','merchant','trans_type1','device1','device_code1']
    tr_cross_2 = ['acc_id1','trans_type2','device_code3','device2','mac1','ip1','market_code','market_type','ip1_sub','amt_src2']
    tr_cross_feature = []
    for feat_1 in tr_cross_1:
        for feat_2 in tr_cross_2:
            col_name = "cross_" + feat_1 + "_and_" + feat_2
            tr_cross_feature.append(col_name)
            trans[col_name] = trans[feat_1].astype(str).values + '_' + trans[feat_2].astype(str).values
    for col in tr_cross_feature:
        temp = trans[['UID',col]].merge(trans.groupby([col]).size().reset_index().rename(columns = {0:'size_%s'%col}),on =col, how = 'left')[['UID','size_%s'%col]]
        temp1 = temp.groupby(['UID'])['size_%s'%col].mean().reset_index()
        label = label.merge(temp1, on = 'UID',how = 'left')
        temp1 = temp.groupby(['UID'])['size_%s'%col].sum().reset_index()
        label = label.merge(temp1, on = 'UID',how = 'left')
        temp1 = temp.groupby(['UID'])['size_%s'%col].max().reset_index()
        label = label.merge(temp1, on = 'UID',how = 'left')
        temp1 = temp.groupby(['UID'])['size_%s'%col].min().reset_index()
        label = label.merge(temp1, on = 'UID',how = 'left')
        
        label = label.merge(trans.groupby(['UID'])[col].count().reset_index(),on = 'UID',how = 'left')
        label = label.merge(trans.groupby(['UID'])[col].nunique().reset_index(),on = 'UID',how = 'left')
        
        for fea in ['day','hour','hourSeg']:
            temp = trans[['UID',col]].merge(trans.groupby([col,fea]).size().reset_index().rename(columns = {0:'size_%s'%col}),on =col, how = 'left')[['UID','size_%s'%col]]
            temp1 = temp.groupby(['UID'])['size_%s'%col].mean().reset_index()
            label = label.merge(temp1, on = 'UID',how = 'left')
            temp1 = temp.groupby(['UID'])['size_%s'%col].sum().reset_index()
            label = label.merge(temp1, on = 'UID',how = 'left')
            temp1 = temp.groupby(['UID'])['size_%s'%col].max().reset_index()
            label = label.merge(temp1, on = 'UID',how = 'left')
            temp1 = temp.groupby(['UID'])['size_%s'%col].min().reset_index()
            label = label.merge(temp1, on = 'UID',how = 'left')
        
        
        gc.collect()
        del temp,temp1
    label = label.drop(['UID'],axis = 1)
    print('time_cross_nunique done!')
    return label
    



def get_cross_nunique(op,trans,label):
    op_cross_1 = ['mode','version','os','device1','device_code1']
    op_cross_2 = ['ip1','mac1','device_code3','device2','mac2','wifi','geo_code','ip1_sub']
    op_cross_feature = []
    for feat_1 in op_cross_1:
        for feat_2 in op_cross_2:
            col_name = "cross_" + feat_1 + "_and_" + feat_2
            op_cross_feature.append(col_name)
            op[col_name] = op[feat_1].astype(str).values + '_' + op[feat_2].astype(str).values
    for col in op_cross_feature:
        temp = op[['UID',col]].merge(op.groupby([col]).size().reset_index().rename(columns = {0:'size_%s'%col}),on =col, how = 'left')[['UID','size_%s'%col]]
        temp1 = temp.groupby(['UID'])['size_%s'%col].mean().reset_index()
        label = label.merge(temp1, on = 'UID',how = 'left')
        temp1 = temp.groupby(['UID'])['size_%s'%col].sum().reset_index()
        label = label.merge(temp1, on = 'UID',how = 'left')
        temp1 = temp.groupby(['UID'])['size_%s'%col].max().reset_index()
        label = label.merge(temp1, on = 'UID',how = 'left')
        temp1 = temp.groupby(['UID'])['size_%s'%col].min().reset_index()
        label = label.merge(temp1, on = 'UID',how = 'left')
        
        label = label.merge(op.groupby(['UID'])[col].count().reset_index(),on = 'UID',how = 'left')
        label = label.merge(op.groupby(['UID'])[col].nunique().reset_index(),on = 'UID',how = 'left')
        gc.collect()
        del temp,temp1
            
    tr_cross_1 = ['channel','amt_src1','merchant','trans_type1','device1','device_code1']
    tr_cross_2 = ['acc_id1','trans_type2','device_code3','device2','mac1','ip1','market_code','market_type','ip1_sub','amt_src2']
    tr_cross_feature = []
    for feat_1 in tr_cross_1:
        for feat_2 in tr_cross_2:
            col_name = "cross_" + feat_1 + "_and_" + feat_2
            tr_cross_feature.append(col_name)
            trans[col_name] = trans[feat_1].astype(str).values + '_' + trans[feat_2].astype(str).values
    for col in tr_cross_feature:
        temp = trans[['UID',col]].merge(trans.groupby([col]).size().reset_index().rename(columns = {0:'size_%s'%col}),on =col, how = 'left')[['UID','size_%s'%col]]
        temp1 = temp.groupby(['UID'])['size_%s'%col].mean().reset_index()
        label = label.merge(temp1, on = 'UID',how = 'left')
        temp1 = temp.groupby(['UID'])['size_%s'%col].sum().reset_index()
        label = label.merge(temp1, on = 'UID',how = 'left')
        temp1 = temp.groupby(['UID'])['size_%s'%col].max().reset_index()
        label = label.merge(temp1, on = 'UID',how = 'left')
        temp1 = temp.groupby(['UID'])['size_%s'%col].min().reset_index()
        label = label.merge(temp1, on = 'UID',how = 'left')
        
        
        label = label.merge(trans.groupby(['UID'])[col].count().reset_index(),on = 'UID',how = 'left')
        label = label.merge(trans.groupby(['UID'])[col].nunique().reset_index(),on = 'UID',how = 'left')
        
        gc.collect()
        del temp,temp1
    label = label.drop(['UID'],axis = 1)
    print('cross_nunique done!')
    return label
    

def get_ratio(op,trans,label):
    op_feature = ['mode','version','os','device1','device_code1','ip1','mac1',
                  'device_code3','device2','mac2','wifi','geo_code','ip1_sub',]
    for i in range(len(op_feature)-1):
        for j in range(i+1,len(op_feature)):
            temp = op.groupby(['UID'])[op_feature[i],op_feature[j]].size().reset_index().rename(columns = {0:'count_i_j'})
            temp1 = op.groupby(['UID'])[op_feature[i]].size().reset_index()
            temp2 = op.groupby(['UID'])[op_feature[j]].size().reset_index()
            temp['ratio_' + op_feature[i] + '_of_' + op_feature[j]] = temp['count_i_j'] / temp2[op_feature[j]]
            temp['ratio_' + op_feature[j] + '_of_' + op_feature[i]] = temp['count_i_j'] / temp1[op_feature[i]]
            label = label.merge(temp,on = 'UID',how = 'left')
            del temp,temp1,temp2
            gc.collect()
            
    tr_feature = ['channel','amt_src1','merchant','trans_type1','device1','device_code1',
                 'acc_id1','trans_type2','device_code3','device2','mac1','ip1','market_code',
                  'market_type','ip1_sub','amt_src2','acc_id2','acc_id3']
    
    for i in range(len(tr_feature)-1):
        for j in range(i+1,len(tr_feature)):
            temp = trans.groupby(['UID'])[tr_feature[i],tr_feature[j]].size().reset_index().rename(columns = {0:'count_i_j'})
            temp1 = trans.groupby(['UID'])[tr_feature[i]].size().reset_index()
            temp2 = trans.groupby(['UID'])[tr_feature[j]].size().reset_index()
            temp['ratio_' + tr_feature[i] + '_of_' + tr_feature[j]] = temp['count_i_j'] / temp2[tr_feature[j]]
            temp['ratio_' + tr_feature[j] + '_of_' + tr_feature[i]] = temp['count_i_j'] / temp1[tr_feature[i]]
            label = label.merge(temp,on = 'UID',how = 'left')
            del temp,temp1,temp2
            gc.collect()
    label = label.drop(['UID'],axis = 1)
    print('ratio done!')
    return label


def get_most_list(op,tr,label):
    for item in op.columns :
        if item!='UID':
            result = op.groupby(['UID'])[item].apply(max_list).reset_index().rename(columns = {item:'op_%s'%item})
            label = label.merge(result, on = ['UID'], how = 'left')
    for item in tr.columns :
        if item!='UID':
            result = tr.groupby(['UID'])[item].apply(max_list).reset_index().rename(columns = {item:'tr_%s'%item})
            label = label.merge(result, on = ['UID'], how = 'left')
    label = label.drop(['UID'],axis = 1)
    gc.collect()
    print('Done!')
    return label


def get_time_features(train_op,train_tr,label):
    a = train_tr.groupby(['UID','day'])['timestamp'].apply(get_time_var).reset_index()
    a.columns = ['UID','day'] + ['time_var']
    b = train_tr.groupby(['UID','day'])['timestamp'].apply(get_time_mean).reset_index()
    b.columns = ['UID','day'] + ['time_var']
    c = train_tr.groupby(['UID','day'])['timestamp'].apply(get_10min_count).reset_index()
    c.columns = ['UID','day'] + ['10min_count']
    d = train_tr.groupby(['UID','day'])['timestamp'].apply(get_5min_count).reset_index()
    d.columns = ['UID','day'] + ['5min_count']
    for item in [a,b,c,d]:
        result = item.groupby(['UID'])[item.columns[2]].mean().reset_index()
        label = label.merge(result,on = ['UID'],how = 'left').fillna(0)
        result = item.groupby(['UID'])[item.columns[2]].min().reset_index()
        label = label.merge(result,on = ['UID'],how = 'left').fillna(0)
        result = item.groupby(['UID'])[item.columns[2]].max().reset_index()
        label = label.merge(result,on = ['UID'],how = 'left').fillna(0)
        result = item.groupby(['UID'])[item.columns[2]].sum().reset_index()
        label = label.merge(result,on = ['UID'],how = 'left').fillna(0)
        
    a = train_op.groupby(['UID','day'])['timestamp'].apply(get_time_var).reset_index()
    a.columns = ['UID','day'] + ['time_var']
    b = train_op.groupby(['UID','day'])['timestamp'].apply(get_time_mean).reset_index()
    b.columns = ['UID','day'] + ['time_var']
    c = train_op.groupby(['UID','day'])['timestamp'].apply(get_10min_count).reset_index()
    c.columns = ['UID','day'] + ['10min_count']
    d = train_op.groupby(['UID','day'])['timestamp'].apply(get_5min_count).reset_index()
    d.columns = ['UID','day'] + ['5min_count']
    for item in [a,b,c,d]:
        result = item.groupby(['UID'])[item.columns[2]].mean().reset_index()
        label = label.merge(result,on = ['UID'],how = 'left').fillna(0)
        result = item.groupby(['UID'])[item.columns[2]].min().reset_index()
        label = label.merge(result,on = ['UID'],how = 'left').fillna(0)
        result = item.groupby(['UID'])[item.columns[2]].max().reset_index()
        label = label.merge(result,on = ['UID'],how = 'left').fillna(0)
        result = item.groupby(['UID'])[item.columns[2]].sum().reset_index()
        label = label.merge(result,on = ['UID'],how = 'left').fillna(0)

    label = label.drop(['UID'],axis = 1)
    gc.collect()
    print('Done!')
    return label

def get_all_time_fea(op,trans,label):
    op_cross_1 = ['mode','version','os','device1','device_code1']
    op_cross_2 = ['ip1','mac1','device_code3','device2','mac2','wifi','geo_code','ip1_sub']
    time_fea = ['day','hour','hourSeg']
    for fea1 in time_fea:
        for fea2 in op_cross_1+op_cross_2:
            temp1 = op.groupby(['UID',fea1])[fea2].count().reset_index()
            label = label.merge(temp1.groupby('UID')[fea2].sum().reset_index(), on = 'UID',how = 'left')
            label = label.merge(temp1.groupby('UID')[fea2].mean().reset_index(), on = 'UID',how = 'left')
            label = label.merge(temp1.groupby('UID')[fea2].max().reset_index(), on = 'UID',how = 'left')
            label = label.merge(temp1.groupby('UID')[fea2].min().reset_index(), on = 'UID',how = 'left')
            label = label.merge(temp1.groupby('UID')[fea2].count().reset_index(), on = 'UID',how = 'left')
            label = label.merge(temp1.groupby('UID')[fea2].nunique().reset_index(), on = 'UID',how = 'left')
            label = label.merge(temp1.groupby('UID')[fea2].std().reset_index(), on = 'UID',how = 'left')
            
            temp2 = op.groupby(['UID',fea1])[fea2].nunique().reset_index()
            label = label.merge(temp2.groupby('UID')[fea2].sum().reset_index(), on = 'UID',how = 'left')
            label = label.merge(temp2.groupby('UID')[fea2].mean().reset_index(), on = 'UID',how = 'left')
            label = label.merge(temp2.groupby('UID')[fea2].max().reset_index(), on = 'UID',how = 'left')
            label = label.merge(temp2.groupby('UID')[fea2].min().reset_index(), on = 'UID',how = 'left')
            label = label.merge(temp2.groupby('UID')[fea2].count().reset_index(), on = 'UID',how = 'left')
            label = label.merge(temp2.groupby('UID')[fea2].nunique().reset_index(), on = 'UID',how = 'left')
            label = label.merge(temp2.groupby('UID')[fea2].std().reset_index(), on = 'UID',how = 'left')
            
            gc.collect()
            del temp1,temp2
            
    tr_cross_1 = ['channel','amt_src1','merchant','trans_type1','device1','device_code1']
    tr_cross_2 = ['acc_id1','trans_type2','device_code3','device2','mac1','ip1','market_code','market_type','ip1_sub','amt_src2']

    for fea1 in time_fea:
        for fea2 in tr_cross_1+tr_cross_2:
            temp1 = trans.groupby(['UID',fea1])[fea2].count().reset_index()
            label = label.merge(temp1.groupby('UID')[fea2].sum().reset_index(), on = 'UID',how = 'left')
            label = label.merge(temp1.groupby('UID')[fea2].mean().reset_index(), on = 'UID',how = 'left')
            label = label.merge(temp1.groupby('UID')[fea2].max().reset_index(), on = 'UID',how = 'left')
            label = label.merge(temp1.groupby('UID')[fea2].min().reset_index(), on = 'UID',how = 'left')
            label = label.merge(temp1.groupby('UID')[fea2].count().reset_index(), on = 'UID',how = 'left')
            label = label.merge(temp1.groupby('UID')[fea2].nunique().reset_index(), on = 'UID',how = 'left')
            label = label.merge(temp1.groupby('UID')[fea2].std().reset_index(), on = 'UID',how = 'left')

            
            temp2 = trans.groupby(['UID',fea1])[fea2].nunique().reset_index()
            label = label.merge(temp2.groupby('UID')[fea2].sum().reset_index(), on = 'UID',how = 'left')
            label = label.merge(temp2.groupby('UID')[fea2].mean().reset_index(), on = 'UID',how = 'left')
            label = label.merge(temp2.groupby('UID')[fea2].max().reset_index(), on = 'UID',how = 'left')
            label = label.merge(temp2.groupby('UID')[fea2].min().reset_index(), on = 'UID',how = 'left')
            label = label.merge(temp2.groupby('UID')[fea2].count().reset_index(), on = 'UID',how = 'left')
            label = label.merge(temp2.groupby('UID')[fea2].nunique().reset_index(), on = 'UID',how = 'left')
            label = label.merge(temp2.groupby('UID')[fea2].std().reset_index(), on = 'UID',how = 'left')
            
            gc.collect()
            del temp1,temp2
    label = label.drop(['UID'],axis = 1)
    return label



def get_feature(op,trans,label):
    for feature in op.columns[:]:
        if feature not in ['day']:
            if feature != 'UID':
                label = label.merge(op.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
                label =label.merge(op.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
            for deliver in ['ip1','mac1','mac2','geo_code']:
                if feature not in deliver:
                    if feature != 'UID':
                        temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].count().reset_index(),on=deliver,how='left')[['UID',feature]] 
                        temp = temp.groupby('UID')[feature].sum().reset_index()
                        temp.columns = ['UID',feature+deliver]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
                        temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].nunique().reset_index(),on=deliver,how='left')[['UID',feature]] 
                        temp = temp.groupby('UID')[feature].sum().reset_index()
                        temp.columns = ['UID',feature+deliver]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
                    else:
                        temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].count().reset_index(),on=deliver,how='left')[['UID_x','UID_y']] 
                        temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        temp.columns = ['UID',feature+deliver]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
                        temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].nunique().reset_index(),on=deliver,how='left')[['UID_x','UID_y']] 
                        temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        temp.columns = ['UID',feature+deliver]
                        label =label.merge(temp,on='UID',how='left')
                        del temp

        else:
            print(feature)
            label =label.merge(op.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].max().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].min().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].sum().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].mean().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].std().reset_index(),on='UID',how='left')
            for deliver in ['ip1','mac1','mac2']:
                if feature not in deliver:
                    temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].count().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].sum().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].nunique().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].sum().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].max().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].min().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].sum().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].mean().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].std().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    
                    
                    
    for feature in trans.columns[1:]:
        if feature not in ['trans_amt','bal','day']:
            if feature != 'UID':
                label =label.merge(trans.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
                label =label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
            for deliver in ['merchant','ip1','mac1','geo_code',]:
                if feature not in deliver: 
                    if feature != 'UID':
                        temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].count().reset_index(),on=deliver,how='left')[['UID',feature]] 
                        temp = temp.groupby('UID')[feature].sum().reset_index()
                        temp.columns = ['UID',feature+deliver]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
                        temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].nunique().reset_index(),on=deliver,how='left')[['UID',feature]] 
                        temp = temp.groupby('UID')[feature].sum().reset_index()
                        temp.columns = ['UID',feature+deliver]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
                    else:
                        temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].count().reset_index(),on=deliver,how='left')[['UID_x','UID_y']] 
                        temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        temp.columns = ['UID',feature+deliver]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
                        temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].nunique().reset_index(),on=deliver,how='left')[['UID_x','UID_y']] 
                        temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        temp.columns = ['UID',feature+deliver]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
            #if feature in ['merchant','code2','acc_id1','market_code','market_code']:
            #    label[feature+'_z'] = 0 
            #    label[feature+'_z'] = label[feature+'_y']/label[feature+'_x']
        else:
            print(feature)
            label =label.merge(trans.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].max().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].min().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].sum().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].mean().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].std().reset_index(),on='UID',how='left')
            for deliver in ['merchant','ip1','mac1']:
                if feature not in deliver:
                    temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].count().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].sum().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].nunique().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].sum().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].max().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].min().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].sum().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].mean().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].std().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    
                    
    label = label.drop(['UID'],axis = 1)                
    print("Done")
    return label




def get_new_feature(op,trans,label):
    for feature in op.columns[:]:
        if feature not in ['day']:
            if feature != 'UID':
                label = label.merge(op.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
                label =label.merge(op.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
            for deliver in ['mode','ip1','ip1_sub','mac1','mac2','geo_code','device1','device_code1','device_code3']:
                if feature not in deliver:
                    if feature != 'UID':
                        temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].count().reset_index(),on=deliver,how='left')[['UID',feature]] 
                        temp = temp.groupby('UID')[feature].sum().reset_index()
                        temp.columns = ['UID',feature+deliver]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
                        temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].nunique().reset_index(),on=deliver,how='left')[['UID',feature]] 
                        temp = temp.groupby('UID')[feature].sum().reset_index()
                        temp.columns = ['UID',feature+deliver]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
                    else:
                        temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].count().reset_index(),on=deliver,how='left')[['UID_x','UID_y']] 
                        temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        temp.columns = ['UID',feature+deliver]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
                        temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].nunique().reset_index(),on=deliver,how='left')[['UID_x','UID_y']] 
                        temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        temp.columns = ['UID',feature+deliver]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
            gc.collect()

        else:
            print(feature)
            label =label.merge(op.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].max().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].min().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].sum().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].mean().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].std().reset_index(),on='UID',how='left')
            for deliver in ['ip1','ip1_sub','mac1','mac2','geo_code','device1','device_code1','device_code3','geo_code']:
                if feature not in deliver:
                    temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].count().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].sum().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].nunique().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].sum().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].max().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].min().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].sum().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].mean().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = op[['UID',deliver]].merge(op.groupby([deliver])[feature].std().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
            gc.collect()
                    
                    
    for feature in trans.columns[1:]:
        if feature not in ['trans_amt','bal','day']:
            if feature != 'UID':
                label =label.merge(trans.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
                label =label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
            for deliver in ['merchant','ip1','ip1_sub','mac1','geo_code','amt_src1','acc_id1','market_code','device_code1','device_code3']:
                if feature not in deliver: 
                    if feature != 'UID':
                        temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].count().reset_index(),on=deliver,how='left')[['UID',feature]] 
                        temp = temp.groupby('UID')[feature].sum().reset_index()
                        temp.columns = ['UID',feature+deliver]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
                        temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].nunique().reset_index(),on=deliver,how='left')[['UID',feature]] 
                        temp = temp.groupby('UID')[feature].sum().reset_index()
                        temp.columns = ['UID',feature+deliver]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
                    else:
                        temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].count().reset_index(),on=deliver,how='left')[['UID_x','UID_y']] 
                        temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        temp.columns = ['UID',feature+deliver]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
                        temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].nunique().reset_index(),on=deliver,how='left')[['UID_x','UID_y']] 
                        temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        temp.columns = ['UID',feature+deliver]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
            gc.collect()
            #if feature in ['merchant','code2','acc_id1','market_code','market_code']:
            #    label[feature+'_z'] = 0 
            
            #    label[feature+'_z'] = label[feature+'_y']/label[feature+'_x']
        else:
            print(feature)
            label =label.merge(trans.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].max().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].min().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].sum().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].mean().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].std().reset_index(),on='UID',how='left')
            for deliver in ['merchant','ip1','mac1','ip1_sub','geo_code','amt_src1','acc_id1','market_code','device_code1','device_code3']:
                if feature not in deliver:
                    temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].count().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].sum().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].nunique().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].sum().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].max().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].min().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].sum().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].mean().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = trans[['UID',deliver]].merge(trans.groupby([deliver])[feature].std().reset_index(),on=deliver,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+deliver]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
            gc.collect()        
                    
    label = label.drop(['UID'],axis = 1)                
    print("Done")
    return label




def get_merchant_feature(trans,label):
    
    tr = trans.groupby(['UID','merchant']).size().reset_index().rename(columns = {0:'UID_merchant_tr_size'})

    for feature in trans.columns[2:]:
        if feature!= 'merchant' and trans[feature].dtype == 'object':
            tr =feat_count(tr,trans,['UID','merchant'],feature)
            tr =feat_nunique(tr,trans,['UID','merchant'],feature)
        elif feature!= 'merchant':
            tr =feat_count(tr,trans,['UID','merchant'],feature)
            tr =feat_nunique(tr,trans,['UID','merchant'],feature)
            tr =feat_max(tr,trans,['UID','merchant'],feature)
            tr =feat_min(tr,trans,['UID','merchant'],feature)
            tr =feat_sum(tr,trans,['UID','merchant'],feature)
            tr =feat_mean(tr,trans,['UID','merchant'],feature)
            tr =feat_std(tr,trans,['UID','merchant'],feature)
            
    for feature in tr.columns[2:]:
        label =feat_count(label,tr,['UID'],feature)
        label =feat_nunique(label,tr,['UID'],feature)
        label =feat_max(label,tr,['UID'],feature)
        label =feat_min(label,tr,['UID'],feature)
        label =feat_sum(label,tr,['UID'],feature)
        label =feat_mean(label,tr,['UID'],feature)
        label =feat_std(label,tr,['UID'],feature)
        
    label = label.drop(['UID'],axis = 1)
    gc.collect()
    print('Done!')
    return label

##get_bool_hour_features:::
def get_hour_features(train,test,train_UID,test_UID):
    train1 = train[['UID','hour']]
    test1 = test[['UID','hour']]
    data = pd.concat([train,test],axis = 0,sort=False)
    base_train_csr = sparse.csr_matrix((len(train_UID), 0))
    base_predict_csr = sparse.csr_matrix((len(test_UID), 0))
    for feature in ['hour']:
        data['hour_isin_03'] = data['hour'].apply(lambda x: 1 if (x<=3) &(x>=0) else 0)
        data['hour_isin_46'] = data['hour'].apply(lambda x: 1 if (x<=6) &(x>=4) else 0)
        data['hour_isin_712'] = data['hour'].apply(lambda x: 1 if (x<=12) &(x>=7) else 0)
        data['hour_isin_1219'] = data['hour'].apply(lambda x: 1 if (x<=19) &(x>=12) else 0)
        data['hour_isin_2023'] = data['hour'].apply(lambda x: 1 if (x<=23) &(x>=20) else 0)
        
        train_process = train_UID.merge(data_process,on = ['UID'], how = 'left')
        test_process = test_UID.merge(data_process,on = ['UID'], how = 'left')
        cv.fit(data_process[feature])
        base_train_csr = sparse.hstack((base_train_csr, cv.transform(train_process[feature].astype(str))), 'csr', 'int')
        base_predict_csr = sparse.hstack((base_predict_csr, cv.transform(test_process[feature].astype(str))), 'csr', 'int')
        print(feature + ' finished!')
    gc.collect()
    print('Done!')
    return base_train_csr,base_predict_csr


def get_onehot_features(train,test,train_UID,test_UID):
    cv = CountVectorizer(min_df=5)
    data = pd.concat([train,test],axis = 0,sort = False)
    base_train_csr = sparse.csr_matrix((len(train_UID), 0))
    base_predict_csr = sparse.csr_matrix((len(test_UID), 0))
    features = list(train.select_dtypes(include=[np.object]).columns)
    for feature in features:
        data[feature] = data[feature].astype(str).fillna(str(-1))
        data_process = data.groupby(['UID'])[feature].apply(lambda x:','.join(x)).reset_index()
        train_process = train_UID.merge(data_process,on = ['UID'], how = 'left').fillna(str(-1))
        test_process = test_UID.merge(data_process,on = ['UID'], how = 'left').fillna(str(-1))
        cv.fit(data_process[feature])
        base_train_csr = sparse.hstack((base_train_csr, cv.transform(train_process[feature].astype(str))), 'csr', 'int')
        base_predict_csr = sparse.hstack((base_predict_csr, cv.transform(test_process[feature].astype(str))), 'csr', 'int')
        print(feature + ' finished!')
    gc.collect()
    print('Done!')
    return base_train_csr,base_predict_csr

##处理时间戳函数
def timestamp(data):
    arr = data['day'].astype(str) + ',' + data['time']
    brr = arr.apply(lambda x:'2018-04-' + x.split(',')[0] + ' ' + x.split(',')[1])
    data['timestamp'] = brr.apply(lambda x:int(time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S"))))
    print('Done!')
    return brr
def timestamp2(data):
    arr = data['day'].astype(str) + ',' + data['time']
    brr = arr.apply(lambda x:'2018-06-' + x.split(',')[0] + ' ' + x.split(',')[1])
    data['timestamp'] = brr.apply(lambda x:int(time.mktime(time.strptime(x,"%Y-%m-%d %H:%M:%S"))))
    print('Done!')
    return brr



def tpr_weight_funtion(y_true,y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3

def tpr_weight(y_predict,d_train):
    y_true = d_train.get_label().copy()
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return ('TPR_weight',0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3,True)

def lgb_crossvalidation(train,label,test,test_id,round = 5000):

    lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=100, reg_alpha=3, reg_lambda=5, max_depth=-1,
        n_estimators=round, objective='binary', subsample=0.9, colsample_bytree=0.77, subsample_freq=1, learning_rate=0.05,
        random_state=1000, n_jobs=16, min_child_weight=4, min_child_samples=5, min_split_gain=0)
    skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)
    best_score = []

    oof_preds = np.zeros(train.shape[0])
    sub_preds = np.zeros(test_id.shape[0])

    for index, (train_index, test_index) in enumerate(skf.split(train, label)):
        lgb_model.fit(train[train_index], label[train_index], verbose=50,
                      eval_set=[(train[train_index], label[train_index]),
                                (train[test_index], label[test_index])], early_stopping_rounds=100)
        best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
        print(best_score)
        oof_preds[test_index] = lgb_model.predict_proba(train[test_index], num_iteration=lgb_model.best_iteration_)[:,1]

        test_pred = lgb_model.predict_proba(test, num_iteration=lgb_model.best_iteration_)[:, 1]
        sub_preds += test_pred / 5
        #print('test mean:', test_pred.mean())
        #predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred

    m = tpr_weight_funtion(y_predict=oof_preds,y_true=label)
    print('final_tpr: ',m)
    sub = pd.read_csv('./data/sub.csv')
    sub['Tag'] = sub_preds
    print('mean: ',sub_preds.mean())
    sub.to_csv('./submit/lgb_baseline_%s.csv'%str(m),index=False)
    return sub



def lgb_seed_cv(train,label,test,test_id,seed,round = 5000):

    lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=100, reg_alpha=3, reg_lambda=5, max_depth=-1,
        n_estimators=round, objective='binary', subsample=0.9, colsample_bytree=0.77, subsample_freq=1, learning_rate=0.05,
        random_state=1000, n_jobs=16, min_child_weight=4, min_child_samples=5, min_split_gain=0)
    
    skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    best_score = []

    oof_preds = np.zeros(train.shape[0])
    sub_preds = np.zeros(test_id.shape[0])

    for index, (train_index, test_index) in enumerate(skf.split(train, label)):
        lgb_model.fit(train[train_index], label[train_index], verbose=50,
                      eval_set=[(train[train_index], label[train_index]),
                                (train[test_index], label[test_index])], early_stopping_rounds=100)
        best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
        print(best_score)
        oof_preds[test_index] = lgb_model.predict_proba(train[test_index], num_iteration=lgb_model.best_iteration_)[:,1]

        test_pred = lgb_model.predict_proba(test, num_iteration=lgb_model.best_iteration_)[:, 1]
        sub_preds += test_pred / 5
        #print('test mean:', test_pred.mean())
        #predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred

    m = tpr_weight_funtion(y_predict=oof_preds,y_true=label)
    print('final_tpr: ',m)
    sub = pd.read_csv('./data/sub.csv')
    sub['Tag'] = sub_preds
    print('mean: ',sub_preds.mean())
    sub.to_csv('./seed/lgb_baseline_%d.csv'%seed,index=False)
    return sub

def xgb_seed_cv(train,label,test,test_id,seed,round = 5000):
    
    xgb_model = XGBClassifier(boosting_type='gbdt', num_leaves=48, max_depth=8, 
                              learning_rate=0.05, n_estimators=round,subsample=0.8,
                              colsample_bytree=0.6, reg_alpha=3, reg_lambda=5, 
                              seed=1000, nthread=10,verbose=50)

    skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    best_score = []

    oof_preds = np.zeros(train.shape[0])
    sub_preds = np.zeros(test_id.shape[0])

    for index, (train_index, test_index) in enumerate(skf.split(train, label)):
        xgb_model.fit(train[train_index], label[train_index], verbose=50,eval_metric = 'logloss',
                      eval_set=[(train[train_index], label[train_index]),
                                (train[test_index], label[test_index])], early_stopping_rounds=100)
        
        
        oof_preds[test_index] = xgb_model.predict_proba(train[test_index], ntree_limit=xgb_model.best_iteration)[:,1]
        valid_loss = log_loss(label[test_index],oof_preds[test_index])
        best_score.append(valid_loss)
        print(best_score)
        test_pred = xgb_model.predict_proba(test, ntree_limit=xgb_model.best_iteration)[:, 1]
        sub_preds += test_pred / 5
        #print('test mean:', test_pred.mean())
        #predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred

    m = tpr_weight_funtion(y_predict=oof_preds,y_true=label)
    print('final_tpr: ',m)
    sub = pd.read_csv('./data/sub.csv')
    sub['Tag'] = sub_preds
    print('mean: ',sub_preds.mean())
    sub.to_csv('./seed/xgb_baseline_%d.csv'%seed,index=False)
    return sub



def xgb_crossvalidation(train,label,test,test_id,round = 5000):
    '''
    xgb_model = XGBClassifier(learning_rate=0.1,
                      n_estimators=round,         # 树的个数--1000棵树建立xgboost
                      max_depth=6,               # 树的深度
                      min_child_weight = 1,      # 叶子节点最小权重
                      gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                      subsample=0.8,             # 随机选择80%样本建立决策树
                      colsample_btree=0.8,       # 随机选择80%特征建立决策树
                      objective='binary:logistic', # 指定损失函数
                      reg_alpha = 3,
                      reg_lambda = 5,
                      scale_pos_weight=1,        # 解决样本个数不平衡的问题
                      random_state=1000            # 随机数
                      )
    '''
    xgb_model = XGBClassifier(boosting_type='gbdt', num_leaves=48, max_depth=8, 
                              learning_rate=0.05, n_estimators=round,subsample=0.8,
                              colsample_bytree=0.6, reg_alpha=3, reg_lambda=5, 
                              seed=1000, nthread=10,verbose=50)

    skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)
    best_score = []

    oof_preds = np.zeros(train.shape[0])
    sub_preds = np.zeros(test_id.shape[0])

    for index, (train_index, test_index) in enumerate(skf.split(train, label)):
        xgb_model.fit(train[train_index], label[train_index], verbose=50,eval_metric = 'logloss',
                      eval_set=[(train[train_index], label[train_index]),
                                (train[test_index], label[test_index])], early_stopping_rounds=100)
        
        
        oof_preds[test_index] = xgb_model.predict_proba(train[test_index], ntree_limit=xgb_model.best_iteration)[:,1]
        valid_loss = log_loss(label[test_index],oof_preds[test_index])
        best_score.append(valid_loss)
        print(best_score)
        test_pred = xgb_model.predict_proba(test, ntree_limit=xgb_model.best_iteration)[:, 1]
        sub_preds += test_pred / 5
        #print('test mean:', test_pred.mean())
        #predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred

    m = tpr_weight_funtion(y_predict=oof_preds,y_true=label)
    print('final_tpr: ',m)
    sub = pd.read_csv('./data/sub.csv')
    sub['Tag'] = sub_preds
    print('mean: ',sub_preds.mean())
    sub.to_csv('./submit/xgb_baseline_%s.csv'%str(m),index=False)
    return sub

def cgb_seed_cv(train,label,test,test_id,seed,round = 5000):
    cgb_model = cb.CatBoostClassifier(iterations=round, depth=8, learning_rate=0.05,
                                      reg_lambda = 5,random_seed=1000,
                                      loss_function='Logloss',logging_level='Verbose')
    
    skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    best_score = []

    oof_preds = np.zeros(train.shape[0])
    sub_preds = np.zeros(test_id.shape[0])

    for index, (train_index, test_index) in enumerate(skf.split(train, label)):
        cgb_model.fit(train[train_index], label[train_index], verbose=50,
                      eval_set=[(train[train_index], label[train_index]),
                                (train[test_index], label[test_index])], early_stopping_rounds=100)
        
        
        oof_preds[test_index] = cgb_model.predict_proba(train[test_index])[:,1]
        valid_loss = log_loss(label[test_index],oof_preds[test_index])
        best_score.append(valid_loss)
        print(best_score)
        test_pred = cgb_model.predict_proba(test)[:, 1]
        sub_preds += test_pred / 5
        #print('test mean:', test_pred.mean())
        #predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred

    m = tpr_weight_funtion(y_predict=oof_preds,y_true=label)
    print('final_tpr: ',m)
    sub = pd.read_csv('./data/sub.csv')
    sub['Tag'] = sub_preds
    print('mean: ',sub_preds.mean())
    sub.to_csv('./seed/cgb_baseline_%d.csv'%seed,index=False)
    return sub



def LGB_cross_version2(train_csr,label,predict_csr,UID_lists,round = 3000,n_folds = 5):
    res = UID_lists
    print('Training set and valid set has been splited completely!：',train_csr.shape,predict_csr.shape) 
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'num_leaves': 48,
        'metric': 'binary_logloss',
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'seed': 1000,
        'verbose': 50,
        'lambda_l1':3,
        'lambda_l2': 0.1,
        'max_depth': -1,
        'min_data_in_leaf':10,
        'min_sum_hessian_in_leaf':5
        #'max_bin': 425
        }
    skf = StratifiedKFold(n_splits=n_folds, random_state=2018, shuffle=True)
    baseloss = []
    loss = 0
    for i, (train_index, test_index) in enumerate(skf.split(train_csr,label)):

        x_train,y_train = train_csr[train_index],label[train_index]
        x_valid,y_valid = train_csr[test_index],label[test_index]
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train)

        lgb_model = lgb.train(params, lgb_train, num_boost_round= round,valid_sets = lgb_eval,
                               valid_names = 'valid',early_stopping_rounds=100,verbose_eval = 50)

        valid_pred = lgb_model.predict(x_valid, num_iteration=lgb_model.best_iteration)
        valid_loss = tpr_weight_funtion(y_valid,valid_pred)
        baseloss.append(valid_loss)
        loss += valid_loss
        test_pred= lgb_model.predict(predict_csr, num_iteration=lgb_model.best_iteration)
        print(baseloss)
        print('test mean:', test_pred.mean())
        res['Tag_%s' % str(i)] = test_pred
    print('tpr_weight:', baseloss, loss/n_folds)

    # 加权平均
    res['Tag'] = 0
    for i in range(n_folds):
        res['Tag'] += res['Tag_%s' % str(i)]
    res['Tag'] = res['Tag']/n_folds
    print('Final mean:',res['Tag'].mean())
    res[['UID', 'Tag']].to_csv("./submit/lgb_baseline.csv", index=False)
    return res

def LGB_cross_version1(train_csr,label,predict_csr,UID_lists,round = 3000,n_folds = 5):
    res = UID_lists.copy()
    print('Training set and valid set has been splited completely!：',train_csr.shape,predict_csr.shape) 
    '''yuan shi can shu
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        #'metric': {'binary_logloss'},
        'num_leaves': 48,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 2018,
        'verbose': 0,
        'lambda_l2': 0.1,
        'max_depth': -1,
        #'max_bin': 425,
        'device': 'gpu',
        'gpu_platform_id': 1, 
        'gpu_device_id' : 0
        }
    '''
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        'num_leaves': 100,
        'learning_rate': 0.05,
        'feature_fraction': 0.77,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'seed': 1000,
        'verbose': 50,
        'lambda_l1': 3,
        'lambda_l2': 5,
        'max_depth': -1,
        'min_child_weight':4,
        'min_data_in_leaf':5,
        #'max_bin': 425,
        'device': 'gpu',
        'gpu_platform_id': 1, 
        'gpu_device_id' : 0
        }
    skf = StratifiedKFold(n_splits=n_folds, random_state=2018, shuffle=True)
    trainloss = []
    validloss = []
    valid_loss = 0
    train_loss = 0
    for i, (train_index, test_index) in enumerate(skf.split(train_csr,label)):

        x_train,y_train = train_csr[train_index],label[train_index]
        x_valid,y_valid = train_csr[test_index],label[test_index]
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train)

        lgb_model = lgb.train(params, lgb_train, num_boost_round= round,valid_sets = lgb_eval,
                               valid_names = 'valid',feval = tpr_weight,early_stopping_rounds=100,verbose_eval = 50)
        
        train_pred = lgb_model.predict(x_train, num_iteration=lgb_model.best_iteration)
        valid_pred = lgb_model.predict(x_valid, num_iteration=lgb_model.best_iteration)
        loss1 = tpr_weight_funtion(y_train,train_pred)
        loss2 = tpr_weight_funtion(y_valid,valid_pred)
        trainloss.append(loss1)
        validloss.append(loss2)
        train_loss += loss1
        valid_loss += loss2
        test_pred= lgb_model.predict(predict_csr, num_iteration=lgb_model.best_iteration)
        print('train_loss:',loss1)
        print('valid_loss:',loss2)
        print('test mean:', test_pred.mean())
        res['Tag_%s' % str(i)] = test_pred
    print('train_tpr_weight:', trainloss, 'mean_tpr:',train_loss/n_folds)
    print('valid_tpr_weight:', validloss, 'mean_tpr:',valid_loss/n_folds)


def LGB_cross_version3(train,label,predict_csr,UID_lists,round = 5000,n_folds = 5):
    res = UID_lists.copy()
    print('Training set and valid set has been splited completely!：',train.shape,predict_csr.shape) 
    lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=100, reg_alpha=3, reg_lambda=5, max_depth=-1,
                                    n_estimators=round, objective='binary', subsample=0.9, colsample_bytree=0.77, subsample_freq=1, learning_rate=0.05,
                                    random_state=1000, n_jobs=16, min_child_weight=4, min_child_samples=5, min_split_gain=0)
    
    skf = StratifiedKFold(n_splits=n_folds, random_state=2018, shuffle=True)
    trainloss = []
    validloss = []
    valid_loss = 0
    train_loss = 0
    for i, (train_index, test_index) in enumerate(skf.split(train,label)):

        lgb_model.fit(train.iloc[train_index], label[train_index], verbose=50,
                      eval_set=[(train.iloc[train_index], label[train_index]),
                                (train.iloc[test_index], label[test_index])], early_stopping_rounds=100)

        train_pred = lgb_model.predict_proba(train.iloc[train_index], num_iteration=lgb_model.best_iteration_)[:,1]
        valid_pred = lgb_model.predict_proba(train.iloc[test_index], num_iteration=lgb_model.best_iteration_)[:,1]
        loss1 = tpr_weight_funtion(label[train_index],train_pred)
        loss2 = tpr_weight_funtion(label[test_index],valid_pred)
        trainloss.append(loss1)
        validloss.append(loss2)
        train_loss += loss1
        valid_loss += loss2
        test_pred= lgb_model.predict_proba(predict_csr, num_iteration=lgb_model.best_iteration_)[:,1]
        print('train_tpr:',loss1)
        print('valid_tpr:',loss2)
        print('test mean:', test_pred.mean())
        res['Tag_%s' % str(i)] = test_pred
    print('train_tpr_weight:', trainloss, 'mean_tpr:',train_loss/n_folds)
    print('valid_tpr_weight:', validloss, 'mean_tpr:',valid_loss/n_folds)

    # 加权平均
    res['Tag'] = 0
    for i in range(n_folds):
        res['Tag'] += res['Tag_%s' % str(i)]
    res['Tag'] = res['Tag']/n_folds
    print('Final mean:',res['Tag'].mean())
    res[['UID', 'Tag']].to_csv("./submit/lgb_baseline.csv", index=False)
    return res


def encode_onehot(df,column_name):
    feature_df=pd.get_dummies(df[column_name], prefix=column_name)
    all = pd.concat([df.drop([column_name], axis=1),feature_df], axis=1)
    return all

def encode_count(df,column_name):
    lbl = LabelEncoder()
    lbl.fit(list(df[column_name].values))
    df[column_name] = lbl.transform(list(df[column_name].values))
    return df

def merge_count(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].count()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_nunique(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].nunique()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_median(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].median()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_mean(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].mean()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_sum(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].sum()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_max(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].max()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_min(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].min()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df

def merge_std(df,columns,value,cname):
    add = pd.DataFrame(df.groupby(columns)[value].std()).reset_index()
    add.columns=columns+[cname]
    df=df.merge(add,on=columns,how="left")
    return df
    

def feat_count(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].count()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_count" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name+value+"_%s_count" % ("_".join(fe))]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_size(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].size()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_size" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name+value+"_%s_size" % ("_".join(fe))]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_nunique(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].nunique()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_nunique" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name+value+"_%s_nunique" % ("_".join(fe))]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_mean(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].mean()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_mean" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name+value+"_%s_mean" % ("_".join(fe))]
    df = df.merge(df_count, on=fe, how="left")
    return df

def feat_std(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].std()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_std" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name+value+"_%s_std" % ("_".join(fe))]
    df = df.merge(df_count, on=fe, how="left")
    return df

def feat_median(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].median()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_median" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name+value+"_%s_median" % ("_".join(fe))]
    df = df.merge(df_count, on=fe, how="left")
    return df

def feat_max(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].max()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_max" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name+value+"_%s_max" % ("_".join(fe))]
    df = df.merge(df_count, on=fe, how="left")
    return df

def feat_min(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].min()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_min" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name+value+"_%s_min" % ("_".join(fe))]
    df = df.merge(df_count, on=fe, how="left")
    return df

def feat_sum(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].sum()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_sum" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name+value+"_%s_sum" % ("_".join(fe))]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df


def feat_var(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].var()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_var" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name+value+"_%s_var" % ("_".join(fe))]
    df = df.merge(df_count, on=fe, how="left")
    return df

