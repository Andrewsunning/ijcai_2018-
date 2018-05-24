import pandas as pd
import numpy as np
import time
import warnings
np.set_printoptions(suppress=True)
warnings.filterwarnings('ignore')

# 删除暂时没有用到的字段
def drop_col(df_data, is_train=False):
    drop_list = ['item_id', 'item_property_list', 'item_city_id', 'user_id', 'context_id', 'predict_category_property', 'shop_id']
    df_data.drop(drop_list, axis=1, inplace=True)
    return  df_data

# 定义判断子串是否在字符串中的函数
def sub_judge(x, sub_string):
    if sub_string in x:
        return True
    else:
        return False

# 参数i:{0,1},表示要提取item_category_list字段的第i个类别
def split_category_list(category_str, i):
    category_list = category_str.split(';')
    if i == 0:
        return category_list[i]
    elif i == 1:
        return category_list[i]
    else:
        try:
            return category_list[i]
        except IndexError:
            return '-1'  # 如果不存在第3个子类，则返回0

#用user_occupation_id、item_brand_id，item_category_0，item_category_1，item_category_2来填充性别缺失值
def set_missing_gender(df):
    gender_df = df[['user_gender_id', 'user_occupation_id', 'item_brand_id', 'item_category_0', 'item_category_1']]
    known_gender = gender_df[gender_df.user_gender_id.notnull()]
    unknown_gender = gender_df[gender_df.user_gender_id.isnull()]

    # 取出标签字段user_gender_id
    y = known_gender.loc[:, 'user_gender_id']

    # 特征属性值
    X = known_gender.loc[:, 'user_occupation_id': 'item_category_1']

    # 用RandomForestClassifier训练模型
    rfc = RandomForestClassifier(random_state=0, n_estimators=1000, n_jobs=2)
    rfc.fit(X, y)

    # 用模型预测性别缺失值
    predict_gender = rfc.predict(unknown_gender.loc[:, 'user_occupation_id': 'item_category_1'])  ### ::两个冒号表示什么意思？？？
    df.loc[(df.user_gender_id.isnull()), 'user_gender_id'] = predict_gender
    return df, rfc
        
# 利用随机森林模型填充用户性别缺失值
from sklearn.ensemble import RandomForestClassifier
def null_duplicates_process(df_data, is_train=False):
    # 删除重复值
    df_data.drop_duplicates(inplace=True)

    #处理缺失值
    # 将-1转换成np.nan
    dict_nan = {
        # '-1':np.NaN,
        -1:np.NaN
    }
    df_data.replace(dict_nan, inplace=True)

    # user_age_level用众数填充
    df_data.user_age_level.fillna(df_data.user_age_level.mode()[0], inplace=True)
    # user_occupation_id用众数填充
    df_data.user_occupation_id.fillna(df_data.user_occupation_id.mode()[0], inplace=True)
    # user_star_level用均值填充
    df_data.user_star_level.fillna(df_data.user_star_level.mean(), inplace=True)
    # item_sales_level用众数填充
    df_data.item_sales_level.fillna(df_data.item_sales_level.mode()[0], inplace=True)
    # shop_review_positive_rate用平均数填充
    df_data.shop_review_positive_rate.fillna(df_data.shop_review_positive_rate.mean(), inplace=True)
    # shop_score_service用平均数填充
    df_data.shop_score_service.fillna(df_data.shop_score_service.mean(), inplace=True)
    # shop_score_delivery用平均数填充
    df_data.shop_score_delivery.fillna(df_data.shop_score_delivery.mean(), inplace=True)
    # shop_score_description用平均数填充
    df_data.shop_score_description.fillna(df_data.shop_score_description.mean(), inplace=True)
    
    for i in range(3):
        df_data.loc[:, 'item_category_' + str(i)] = df_data.item_category_list.apply(split_category_list, args=(i,))

    ## 出错了，因为预测性别时需要用到item_brand_id字段，而该字段有缺失值，所以fit()函数会报错
    df_data.item_brand_id.fillna(method='ffill', inplace=True)

    df_data,rfc = set_missing_gender(df_data)

    #前向填充缺失值,fillna()默认按列填充
    df_data.fillna(method='ffill', axis=0, inplace=True)
    
    return df_data

def split_category_list(category_str, i):
    category_list = category_str.split(';')
    if i==0:
        return category_list[i]
    elif i==1:
        return category_list[i]
    else:
        try:
            return category_list[i]
        except IndexError:
            return '-1'

def category_process(category_2, category_3):
    if category_3 == '-1':
        return category_2
    else:
        return category_3

def category(df_data, is_train=False):
    for i in range(3):
        df_data.loc[:,'item_category_'+str(i)] = df_data.item_category_list.apply(split_category_list, args=(i,))
        
    df_data.loc[:,'property'] = list(map(lambda x,y:category_process(x, y), df_data.item_category_1, df_data.item_category_2))

    dummies = pd.get_dummies(df_data.property, prefix='property')

    #删除 'item_category_0', 'item_category_1'，‘item_category_2’, 'item_category_list', 'property'五列
    df_data.drop(['item_category_0', 'item_category_1', 'item_category_2', 'item_category_list', 'property'], axis=1, inplace=True)

    cat_df = pd.concat([df_data, dummies], axis=1)
    return cat_df


def convert_thousand(x, thousands):
    return int(x-thousands)

def jian_thousand(df_data, is_train=False):
    df_data.loc[:,'user_occupation_id'] = df_data.user_occupation_id.apply(convert_thousand, args=(2000,))
    df_data.loc[:,'user_age_level'] = df_data.user_age_level.apply(convert_thousand, args=(1000,))
    df_data.loc[:,'user_star_level'] = df_data.user_star_level.apply(convert_thousand, args=(3000,))
    df_data.loc[:,'context_page_id'] = df_data.context_page_id.apply(convert_thousand, args=(4000,))
    df_data.loc[:,'shop_star_level'] = df_data.shop_star_level.apply(convert_thousand, args=(5000,))
    return df_data


# 定义user_gender_id、user_age_level两个字段的拼接函数
def cat_age_gender(age_level, gender_id):
    return age_level+'+'+gender_id

def gender_age(df_data, is_train=False):
    # 将user_gender_id、user_age_level字段转换为字符串类型，便于后续拼接
    df_data.user_gender_id = df_data.user_gender_id.astype('str')
    df_data.user_age_level = df_data.user_age_level.astype('str')

    
    df_data.loc[:,'cat_age_gender'] = list(map(lambda x,y : cat_age_gender(x,y), df_data.user_age_level, df_data.user_gender_id))

    dummies = pd.get_dummies(df_data['cat_age_gender'], prefix='cat_age_gender')
    df_data.drop(['user_gender_id', 'user_age_level', 'cat_age_gender'], axis=1, inplace=True)

    cat_df_with_dummies = pd.concat([df_data, dummies], axis=1)
    return cat_df_with_dummies


#bins = [1.5194085487e+15, 2.27620937822e+18, 4.53175049861e+18, 6.8381763414e+18, 9.22239633696e+18]
# 按照上面的bins对item_brand_id字段进行分组
# 具体分为0-9组
def  convert_item_brand_id(x):
    if 1519408548701639.0<=x<9.876508624198538e+17:
        return 0
    elif 9.876508624198538e+17<=x<=1.8519578876377582e+18:
        return 1
    elif 1.8519578876377582e+18<=x<=2.756163007387907e+18:
        return 2
    elif 2.756163007387907e+18<=x<=3.72245477335059e+18:
        return 3
    elif 3.72245477335059e+18<=x<=4.5317504986144256e+18:
        return 4
    elif 4.5317504986144256e+18<=x<=5.431663708056886e+18:
        return 5
    elif 5.431663708056886e+18<=x<=6.330734742760423e+18:
        return 6
    elif 6.330734742760423e+18<=x<=7.336137537402504e+18:
        return 7
    elif 7.336137537402504e+18<=x<=8.284144144506873e+18:
        return 8
    else:
        return 9

def item_brand_id(df_data, is_train=False):
    df_data['item_brand_id'] = df_data.item_brand_id.apply(convert_item_brand_id)
    df_temp = pd.get_dummies(df_data.item_brand_id, prefix='item_brand_id')
    cat_df = pd.concat([df_data, df_temp], axis=1)
    cat_df.drop('item_brand_id', inplace=True, axis=1)
    return cat_df

def user_occupation_id(df_data, is_train=False):
    # dummies_Gender = pd.get_dummies(df_data['user_gender_id'], prefix= 'user_gender_id')
    dummies_Occupation = pd.get_dummies(df_data['user_occupation_id'], prefix= 'user_occupation_id')
    df = pd.DataFrame.copy(pd.concat([df_data, dummies_Occupation], axis=1))
    df.drop(['user_occupation_id'], inplace=True, axis=1)
    return df


##定义int64时间戳->str(年月日)的转换函数
def ser_convert_int(x):
    time_local = time.localtime(x)
    return time.strftime("%H:%M:%S", time_local)

#时间离散化
# 0->表示交易发生在'00:00:00'到'3:59:59'
# 1->表示交易发生在'06:00:00'到'11:59:59'
# 2->表示交易发生在'12:00:00'到'17:59:59'
# 3->表示交易发生在'18:00:00'到'23:59:59'
def time_intervel(x):
    hour,minute,second = x.split(':')
    hour,minute,second = int(hour),int(minute),int(second)
    if hour < 6:
        return 0
    elif hour < 12:
        return 1
    elif hour < 18:
        return 2
    else:
        return 3

def context_timestamp(df_data, is_train=False):
    df_data.context_timestamp = df_data.context_timestamp.astype('int64')
    df_data.context_timestamp = df_data.context_timestamp.apply(ser_convert_int)

    df_data.loc[:,'time_intervel'] = df_data.context_timestamp.apply(time_intervel)
    df_data.drop('context_timestamp', axis=1, inplace=True)
    return df_data


import sklearn.preprocessing as preprocessing
# 定义幅度缩放函数
# scaled_list:要进行幅度缩放的字段
# X:要进行幅度缩放的DataFrame
def scale_func(scaled_list, X):
    scaler = preprocessing.StandardScaler()
    for item in scaled_list:
        scale_param = scaler.fit(X[[item]])
        X.loc[:,'scaled_'+item] = scaler.fit_transform(X[[item]], scale_param)
    X.drop(scaled_list, inplace=True, axis=1)
    return X


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
# 将sklearn数据处理与GridSearchCV合起来
def LogisticGridCV(data, clf, param_grid, scoring):
    # 特征和标记区别开来
    X = data.filter(regex='item.*|user.*|shop.*|context_page_id|time_intervel|property.*|cat_age.*')
    Y = data.loc[:, 'is_trade']
    X = scale_func(list(X.columns), X)
    # 参数stratify=y表示按y进行分层切分，保证训练集和测试集中的转化率相同
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=22, stratify=Y)
    # 以损失函数为性能指标进行调优
    ### 问题：为什么‘scoring='neg_log_loss'损失值要转化成负值呢？？？
    grid_clf = GridSearchCV(clf, param_grid, n_jobs=2, scoring=scoring)
    grid_clf.fit(X_train, Y_train)
    print("训练集最优性能分数为：{}".format(grid_clf.best_score_))
    best_iter = grid_clf.best_estimator_
    pred_prob_X_test = best_iter.predict_proba(X_test)
    print("模型验证集损失函数为：{}".format(log_loss(Y_test, pred_prob_X_test)))
    return best_iter, X, Y


 # 定义用最优分类器进行预测的函数submit
def submit(train_X, train_Y, test, clf):
    best_iter = LogisticRegression(**clf.get_params())
    best_iter.fit(train_X, train_Y)
    # 过滤掉instance_id字段
    tmp_test = test.filter(regex='item.*|user.*|shop.*|context_page_id|time_intervel|property.*|cat_age.*')
    tmp_test = scale_func(list(tmp_test.columns), tmp_test)
    pred_test = best_iter.predict_proba(tmp_test)
    
    get_instance_id_df = pd.read_table('./data/round1_ijcai_18_test_a_20180301.txt', sep=' ')
    
    sub_df = pd.DataFrame({
        'instance_id':get_instance_id_df['instance_id'],
        'predicted_score': pred_test[:,1]
    })
    sub_df.to_csv('result.txt', index=False, sep=' ')



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
if __name__=='__main__':
    # 读取训练集
    train = pd.read_table("./data/round1_ijcai_18_train_20180301.txt", sep=' ')
    
    # 训练集数据处理
    print("---------------训练集数据处理开始----------------------")
    data = drop_col(train)
    data = null_duplicates_process(data)
    data = category(data)
    data = jian_thousand(data)
    data = gender_age(data)
    data = item_brand_id(data)
    data = user_occupation_id(data)
    data = context_timestamp(data)
    print("---------------训练集数据处理完成----------------------")
    
    # 单模型调优时的参数
    param_grid_1 = {
        'C':[0.001, 0.01, 0.1, 1, 10]  # 缩小范围，简化处理，不然耗时太长
        ,'penalty':['l1', 'l2']
    }
    lr_clf = LogisticRegression()
    
    # 模型性能度量
    scoring='neg_log_loss'
    
    # 单模型网格搜索
    print("---------------模型调参开始----------------------")
    best_iter, train_X, train_Y = LogisticGridCV(data, lr_clf, param_grid_1, scoring)
    print("---------------模型调参完成----------------------")
    
    # 读取测试集
    test = pd.read_table("./data/round1_ijcai_18_test_a_20180301.txt", sep=' ')
    
    # 测试集数据处理
    print("---------------测试集数据处理开始----------------------")
    data_tes = drop_col(test)
    data_tes = null_duplicates_process(data_tes)
    data_tes = category(data_tes)
    data_tes = jian_thousand(data_tes)
    data_tes = gender_age(data_tes)
    data_tes = item_brand_id(data_tes)
    data_tes = user_occupation_id(data_tes)
    data_tes = context_timestamp(data_tes)
    print("---------------测试集数据处理完成----------------------")
    
    # 预测测试集,生成result.txt文件
    submit(train_X, train_Y, data_tes, best_iter)