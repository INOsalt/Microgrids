import numpy as np
import pandas as pd

# 节点负荷矩阵# %Nodedata=[Bus ID	kW  KVAR ]
# 读取基础负荷数据
base_load_df = pd.read_csv('grid/base_load.csv')
base_load_data = base_load_df.to_numpy()
# 读取负荷百分比数据
load_percent_df = pd.read_csv('grid/load_percent.csv')
load_percent_data = load_percent_df[['hour', 'Wkdy']].to_numpy()

# 创建一个字典来存储每小时的节点负荷矩阵
nodedata_dict = {}
for hour, percent in load_percent_data:
    # 对于每个小时，先将负荷数据转换为浮点数，然后计算基础负荷乘以对应小时的负荷百分比
    adjusted_load = base_load_data.astype(float)  # 将整个数组转换为浮点数
    adjusted_load[:, 1] *= percent/100  # 调整kW
    adjusted_load[:, 2] *= percent/100  # 调整KVAR
    # 将调整后的负荷矩阵存储到字典中，使用小时作为键
    nodedata_dict[int(hour)] = adjusted_load
    # 将调整后的负荷矩阵存储到字典中，使用小时作为键
    nodedata_dict[int(hour)] = adjusted_load

# PV WT发电 Bus ID	PG	QGmin 	QGmax
# 读取PV数据
pv_df = pd.read_csv('grid/PV.csv')
pv_data = pv_df[['Bus ID', 'PG', 'QGmin', 'QGmax']].to_numpy()
# 读取WT发电单元数据
wt_df = pd.read_csv('grid/WT.csv')
wt_data = wt_df[['Bus ID', 'PG', 'QGmin', 'QGmax']].to_numpy()
# 读取每小时可用的可再生能源百分比数据
available_re_df = pd.read_csv('grid/available_re.csv')
available_re_data = available_re_df[['hour', 'PV', 'WT']].to_numpy()
# 创建两个字典分别存储每小时的PV和WT发电能力向量
pv_capacity_dict = {}
wt_capacity_dict = {}
for hour, pv_percent, wt_percent in available_re_data:
    # 计算每小时的PV发电能力
    pv_capacity = pv_data.copy()
    pv_capacity[:, 1] = pv_capacity[:, 1] * pv_percent / 100
    pv_capacity_dict[int(hour)] = pv_capacity

    # 计算每小时的WT发电能力
    wt_capacity = wt_data.copy()
    wt_capacity[:, 1] = wt_capacity[:, 1] * wt_percent / 100
    wt_capacity_dict[int(hour)] = wt_capacity

