import pandas as pd

sub = pd.read_csv('./data/sub.csv')
sub['Tag'] = 0

for seed in [1000,2018,3096,4088,5000,6666,7668,8888]:
	sub['Tag'] += pd.read_csv('./seed/lgb_baseline_%d.csv'%seed)
	sub['Tag'] += pd.read_csv('./seed/xgb_baseline_%d.csv'%seed)
	sub['Tag'] += pd.read_csv('./seed/cgb_baseline_%d.csv'%seed)

sub['Tag'] = sub['Tag']/24
sub.to_csv('./seed/final_submit.csv',index = False)