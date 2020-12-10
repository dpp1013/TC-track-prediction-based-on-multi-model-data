'''
监督化文件
'''
import pandas as pd
import numpy as np

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)


def series_to_supervised(data, n_in, n_out, Label):
    #     label : list 标签
    col_name = data.columns
    cols, names = list(), list()
    # 输入序列
    for i in range(n_in, 0, -1):
        cols.append(data.shift(i))
        names += [(x + '(t-%d)') % i for x in col_name]
    # 预测序列 (t,t+1,...,t+n)
    for i in range(0, n_out):
        for j in Label:
            cols.append(data[j].shift(-i))
            names += ['label_' + j + '_(t+%d)' % i]
        #     print(names)
    data_super = pd.concat(cols, axis=1)
    data_super.columns = names

    return data_super, cols


def supervised(data):
    #     k:总步长
    start = 0
    data_new = pd.DataFrame()
    count = 0
    while start < len(data) - 2:
        target = start + 1
        # start 定死了，target 往后移
        while data['SID'][start] == data['SID'][target] and target < len(data) - 1:
            target += 1
        count += 1
        #         print('start:%d,end:%d'%(start,target))
        data_local, cols = series_to_supervised(data[start:target], 8, 8, ['LAT', 'LON'])
        data_new = data_new.append(data_local, ignore_index=True)
        start = target
    return data_new, count


if __name__ == '__main__':
    path = 'dataset/hidden_vector_data.csv'
    hidden_vector_data = pd.read_csv(path, usecols=['SID', 'TIME', 'h1', 'h2', 'h3', 'h4', 'LAT', 'LON'])
    data_new, count = supervised(hidden_vector_data)

    del_features_1 = [['SID(t-' + str(i) + ')', 'TIME(t-' + str(i) + ')'] for i in range(8, 0, -1)]

    for i in del_features_1:
        data_new.drop(i, axis=1, inplace=True)
    data_new.drop(data_new.index[[0,1,2,3,4,5,6,7]],inplace=True)
    print(data_new.head())
    data_new.to_csv('24_predicted_3_supervised_data.csv')
