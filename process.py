import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
'''
数据说明
SID：唯一标示
SEASON：台风开始的年
NUMBER：一年的第几个台风
SUBBASIN：子盆地当前风暴的位置
NAME：名字
ISO_TIME：观测的系统时间
NATURE：类型
LAT
LON
WMO_WIND：最大持续风速
WMO_PRES：最小中心压力
WMO_AGENCY：负责在当前位置对系统发出警告的WMO机构
TRACK_TYPE：轨道类型(主或支路)
DIST2LAND：从当前位置到着陆的当前距离
LANDFALL：未来3小时至陆地的最小距离
IFLAG：一种标识用于在给定时间填充值的内插类型的标志
STORM_SPEED：风暴的转化速度
STORM_DIR：风暴转化方向


'''

# 显示所有列
pd.set_option('display.max_columns', None)

# 显示所有行
pd.set_option('display.max_rows', None)

# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
file = '../../dataset/ibtracs.WP.list.v04r00.csv'

data = pd.read_csv(file)
data = data[['SID', 'BASIN', 'SUBBASIN', 'ISO_TIME',
             'NATURE', 'LAT', 'LON', 'WMO_WIND', 'WMO_PRES', 'WMO_AGENCY', 'TRACK_TYPE', 'DIST2LAND', 'LANDFALL',
              'USA_AGENCY', 'USA_ATCF_ID', 'USA_LAT', 'USA_LON','STORM_SPEED','STORM_DIR']]
data = data[1:]
print(data['ISO_TIME'].dtypes)
# data['ISO_TIME'] =
# print(data.head())

