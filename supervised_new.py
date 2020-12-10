'''
监督化文件
'''
import pandas as pd
import numpy as np

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

n_in = 24
n_out = 24


def series_to_supervised(data):
    all_n = n_in + n_out
    if len(data) < all_n:
        print("error: 数据量不够 跳过")
        return None, None
    #     label : list 标签
    col_name = data.columns
    cols, names = list(), list()

    for i in range(all_n):
        cols.append(data.shift(-i))
        names += [(x + '(t-%d)') % i for x in col_name]
    data_super = pd.concat(cols, axis=1)[:-all_n]
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
        print('start:%d,end:%d,start sid: %s,end sid: %s' % (start, target, data['SID'][start], data['SID'][target]))
        data_local, cols = series_to_supervised(data[start:target])
        if data_local is not None:
            count += 1
            data_new = data_new.append(data_local, ignore_index=True)
        start = target
    return data_new, count


if __name__ == '__main__':
    path = 'dataset/hidden_vector_data.csv'
    hidden_vector_data = pd.read_csv(path, usecols=['SID', 'TIME', 'h1', 'h2', 'h3', 'h4', 'LAT', 'LON'])
    # print(hidden_vector_data.head())
    data_new, count = supervised(hidden_vector_data)
    print("nan 的情况")
    del_features = []
    for i in range(n_in + n_out):
        del_features.append('SID(t-' + str(i) + ')')
        del_features.append('TIME(t-' + str(i) + ')')
    for i in range(n_in, n_in + n_out):
        del_features.append('h1(t-' + str(i) + ')')
        del_features.append('h2(t-' + str(i) + ')')
        del_features.append('h3(t-' + str(i) + ')')
        del_features.append('h4(t-' + str(i) + ')')
    for col in del_features:
        del data_new[col]
    print(pd.isna(data_new).sum())  # 看一下里面有没有nan
    # data_new.drop(data_new.index[[0, 1, 2, 3, 4, 5, 6, 7]], inplace=True)
    # print(data_new.head())
    data_new.to_csv('dataset/{}_predicted_{}_supervised_data.csv'.format(n_in * 3, n_out * 3))