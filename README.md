# DC_TianCheng

1.46/2139  
复赛A榜0.47，B榜0.44010    
2.运行顺序：  
python get_feature.py  
python merchant_id.py  
python main.py  
python cat_main.py  
python bagging.py  

3.特征工程:  
get_cross 交叉特征  
get_ratio 转化率特征  
get_merchant_feature 商户统计特征  
get_time_features，get_all_time_fea 时间统计特征  
get_most_list  id出现最多次数特征，统计之后要做一个onehot  
get_onehot_features  统计id countvector特征  
get_time_cross_nunique 对交叉特征做time上的统计  
也试过对onehot和countvector部分做stacking特征，线下有千分点的提升，线上崩了。  

4.关于bagging:  
主要是lgb，xgb和catboost的加权，lgb权重可以设置大一些。

特征还是主要是以用户为主体  
感觉需要看下前排大佬的开源，好好学习，还是太菜了。