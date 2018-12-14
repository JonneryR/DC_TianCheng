import pandas as pd
import numpy as np
from util import *
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
import torch as tr
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils as utils
import torch.nn.functional as F
import torch.optim as opt
import torch.utils.data
import os

trans_train = pd.read_csv('./data/transaction_train_new.csv',low_memory= False)
trans_test = pd.read_csv('./data/test_transaction_round2.csv',low_memory= False)
label = pd.read_csv('./data/tag_train_new.csv',low_memory= False)
sub = pd.read_csv('./data/sub.csv',low_memory= False)
y = label.drop(['Tag'], axis = 1)
sub = sub.drop(['Tag'], axis = 1)

data_trans = pd.concat([trans_train,trans_test],axis = 0)
merchant = pd.concat([trans_train[['merchant']],trans_test[['merchant']]],axis = 0)
result = merchant.groupby(['merchant']).size().reset_index().rename(columns = {0:'merchant_times'})
new_result = result[result['merchant_times'] > 10]
new_result = new_result.merge(data_trans, on= 'merchant',how = 'left')
data = new_result[['UID','merchant','trans_type1','trans_type2','trans_amt','code1','acc_id1','device1','mac1','ip1','ip1_sub']]

for col in data.columns:
    if (data[col].dtype == 'object') and (col!='UID'):
        data = encode_count(data,col)

train = data.drop(['merchant','UID'],axis = 1).fillna(-1)
label = data['merchant'].values

if(os.path.exists('./feature/merchant_np.npy')):

	merchant_weight = np.load('./feature/merchant_np.npy')
	for item in ['merchant']:
	        
	    result = data.groupby(['UID'])[item].apply(max_list).reset_index().rename(columns = {item:'arr_%s'%item}).fillna(0)
	    y = y.merge(result[['UID','arr_%s'%item]], on = ['UID'], how = 'left').fillna(0)
	    sub = sub.merge(result[['UID','arr_%s'%item]], on = ['UID'], how = 'left').fillna(0)


	 ##pay attention :: astype(int)!!!!!!!!
	for dat in [y,sub]:
		dat['new_merchant'] = dat['arr_merchant'].astype(int).apply(lambda x:merchant_weight[x])
	for i in range(100):
	    y['new_merchant_%d'%i] = y['new_merchant'].apply(lambda x:x[i])
	    sub['new_merchant_%d'%i] = sub['new_merchant'].apply(lambda x:x[i])

	train = y.drop(['UID','arr_merchant','new_merchant'],axis = 1).values
	test = sub.drop(['UID','arr_merchant','new_merchant'],axis = 1).values

	print(train.shape,test.shape)
	np.save('./feature/train_uid_merchant',train)
	np.save('./feature/test_uid_merchant',test)


else:

	class MLP(nn.Module):
	    def __init__(self):
	        super(MLP,self).__init__()
	        self.fc1 = nn.Linear(9,100)
	        self.fc2 = nn.Linear(100,3173)
	    def forward(self,x):
	        out1 = F.relu(self.fc1(x))
	        out2 = F.relu(self.fc2(out1))
	        return out2

	device = tr.device('cuda')
	learning_rate = 0.01
	model = MLP().to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = tr.optim.Adam(model.parameters(), lr=learning_rate)
	batch_size = 100
	train_tr = tr.from_numpy(train.values).float().to(device)
	label_tr = tr.from_numpy(label).long().to(device)
	train_dataset = torch.utils.data.TensorDataset(train_tr,label_tr)
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
	                                           batch_size=batch_size, 
	                                           shuffle=True)


	num_epochs = 10
	for epoch in range(num_epochs):
	    for i, (trains, labels) in enumerate(train_loader):
	        trains = trains.to(device)
	        labels = labels.to(device)
	        
	        # Forward pass
	        outputs = model(trains)
	        loss = criterion(outputs, labels)
	        
	        # Backward and optimize
	        optimizer.zero_grad()
	        loss.backward()
	        optimizer.step()
	    print('Epoch %d'%epoch + 'finished!')


	params=model.state_dict() 
	merchant_weight = params['fc2.weight'].cpu().numpy() 
	np.save('./feature/merchant_np',merchant_weight)



	for item in ['merchant']:
	        
	    result = data.groupby(['UID'])[item].apply(max_list).reset_index().rename(columns = {item:'arr_%s'%item}).fillna(0)
	    y = y.merge(result[['UID','arr_%s'%item]], on = ['UID'], how = 'left').fillna(0)
	    sub = sub.merge(result[['UID','arr_%s'%item]], on = ['UID'], how = 'left').fillna(0)


	 ##pay attention :: astype(int)!!!!!!!!
	for dat in [y,sub]:
		dat['new_merchant'] = dat['arr_merchant'].astype(int).apply(lambda x:merchant_weight[x])
	
	for i in range(100):
	    y['new_merchant_%d'%i] = y['new_merchant'].apply(lambda x:x[i])
	    sub['new_merchant_%d'%i] = sub['new_merchant'].apply(lambda x:x[i])

	train = y.drop(['UID','merchant','new_merchant'],axis = 1).values
	test = sub.drop(['UID','merchant','new_merchant'],axis = 1).values

	print(train.shape,test.shape)
	np.save('./feature/train_uid_merchant',train)
	np.save('./feature/test_uid_merchant',test)
