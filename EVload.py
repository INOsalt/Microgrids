from docplex.mp.model import Model
import pandas as pd
import os
import numpy as np

# 初始化
EV_penetration = 1000 #辆/每节点
power_0_to_5 = 60  # kW 快充
power_6_to_39 = 7  # kW 慢充

# 读取矩阵
# 设置包含CSV文件的文件夹路径
folder_path = 'TM'  # 你需要替换为实际的路径

# 初始化一个空字典来存储矩阵
transition_matrices = {}
# 生成步长为0.05的序列从0到24
keys = np.arange(0, 24.05, 0.05)

# 遍历文件夹中的所有文件
i = 0
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, filename)
        matrix_df = pd.read_csv(file_path)
        # 将DataFrame转换为NumPy数组
        matrix = matrix_df.to_numpy()
        #print(f"Matrix {filename} size: {matrix.shape}")#检查矩阵形状
        matrix_name = keys[i]
        i = i + 1
        # 将矩阵存储到字典中
        transition_matrices[matrix_name] = matrix

# 路网结构
num_nodes = 40
LJ = np.zeros((num_nodes, num_nodes))  # 初始化邻接矩阵
# 定义节点列表
nodes = [101, 102, 103, 104, 105, 106, 201, 202, 203, 204, 205, 206, 207, 208, 209, 301, 302, 303, 304, 305, 306,
         307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 401, 402, 403, 404, 405, 406, 407]

# 创建一个从节点编号到索引的映射
node_mapping = {node: index for index, node in enumerate(nodes)}
# 初始化所有车辆都停泊在起点，没有车辆在移动状态
start_points = [202, 203, 204, 205, 206, 208, 209, 303, 304, 305, 306, 307, 308, 309, 313, 314, 315, 316, 317,
                318, 401, 402, 403, 404, 405, 406, 407]
end_points = [101, 102, 103, 104, 105, 106]

# 初始车辆分布
initial_EV = np.zeros(2 * num_nodes)
starts = np.array([node_mapping[point] for point in start_points])
initial_EV[starts] = EV_penetration

#充电比例
charging_ratio = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06190476, 0.05571429, 0.06190476,
    0.04333333, 0.08047619, 0.0, 0.08047619, 0.09904762, 0.0, 0.0, 0.08047619,
    0.08047619, 0.09904762, 0.11761905, 0.13619048, 0.1547619, 0.08047619,
    0.0, 0.0, 0.0, 0.09904762, 0.06190476, 0.11761905, 0.13619048, 0.13619048,
    0.1547619, 0.08047619, 0.09904762, 0.11761905, 0.13619048, 0.09904762,
    0.11761905, 0.13619048, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
])

# 计算充电需求分布
charging_requirement = initial_EV * charging_ratio * 70 * 0.6  # kWh

#可转移节点对:
direct_connection = {(12, 2), (5, 4), (3, 5), (3, 2), (15, 3), (6, 1), (1, 4), (2, 1),
                     (26, 5), (5, 0), (2, 5), (25, 3), (10, 3), (2, 4), (4, 0)}
spaced_connection = {(2, 1), (2, 5), (10, 3), (26, 25), (12, 2), (7, 6), (12, 6), (10, 12),
                     (25, 26), (5, 0), (16, 15), (16, 25), (12, 10), (24, 10), (3, 2), (5, 4),
                     (15, 3), (24, 15), (1, 4), (9, 10), (6, 12), (25, 3), (3, 5), (8, 12), (6, 1),
                     (26, 5), (28, 25), (2, 4)}

def EV_load(price_1, price_2, price_3, price_4):
    # 初始化模型
    mdl = Model('EV_Charging')

    # 定义每个时刻每个节点的充电汽车数变量
    charging_cars = {(i, t): mdl.integer_var(name=f'charging_cars_{i}_{t}')
                     for i in range(40) for t in np.arange(0, 24.05, 0.05)}

    # 对于每个时间点和节点，充电车数量不超过最大车辆数
    X = initial_EV.copy()  # 复制初始状态以避免修改原始数据
    for time in keys:
        updated_X = np.dot(X, transition_matrices[time])  # 使用矩阵乘法更新X
        for i in range(40):
            current_val = updated_X[i]  # 获取更新后的单个节点状态
            mdl.add_constraint(charging_cars[i, time] <= current_val, f'max_cars_constraint_{i}_{time}')
        X = updated_X  # 更新X以用于下一个时间步的计算

    # 充电功率约束
    Y = charging_requirement.copy()
    for time in keys:
        updated_Y = np.dot(Y, transition_matrices[time])
        for i in range(40):
            power = power_0_to_5 if i in range(6) else power_6_to_39
            current_val = updated_Y[i]
            mdl.add_constraint(charging_cars[i, time] * power * 0.05<= current_val, f'max_power_constraint_{i}_{time}')
        Y = updated_Y  # 更新X以用于下一个时间步的计算

    total_charging_demand = np.sum(charging_requirement)

    # 所有充电负荷的和等于充电需求的和允许一定的偏差
    allowed_deviation = 0.01
    mdl.add_constraint(
        mdl.sum(
            charging_cars[i, t] * (power_0_to_5 if i < 6 else power_6_to_39) * 0.05
            for i in range(40) for t in keys
        ) <= total_charging_demand * (1 + allowed_deviation),
        "upper_bound_total_charging_demand_constraint"
    )
    mdl.add_constraint(
        mdl.sum(
            charging_cars[i, t] * (power_0_to_5 if i < 6 else power_6_to_39) * 0.05
            for i in range(40) for t in keys
        ) >= total_charging_demand * (1 - allowed_deviation),
        "lower_bound_total_charging_demand_constraint"
    )

    # 生成每个i的价格
    price_mapping = {}
    for node, i in node_mapping.items():
        if 101 <= node <= 106:
            price_mapping[i] = price_1
        elif 201 <= node <= 209:
            price_mapping[i] = price_2
        elif 301 <= node <= 318:
            price_mapping[i] = price_3
        elif 401 <= node <= 407:
            price_mapping[i] = price_4

    # 定义目标函数：最小化总充电成本
    total_cost = mdl.sum(charging_cars[i, t] * price_mapping[i] for i in range(40) for t in keys)
    mdl.minimize(total_cost)

    # 求解模型
    # mdl.parameters.read.datacheck = 1  # 开启数据检查
    # mdl.parameters.mip.display = 2  # 设置显示级别为详细
    # solution = mdl.solve(log_output=True)
    solution = mdl.solve()

    # 检查解决方案是否存在
    if solution:
        print("Solution found")
        # 初始化充电负荷矩阵
        charging_load = np.zeros((num_nodes, len(keys)))

        for i in range(num_nodes):
            for j, t in enumerate(keys):
                # 计算每个节点在每个时刻的充电负荷
                power = power_0_to_5 if i < 6 else power_6_to_39
                charging_load[i, j] = charging_cars[i, t].solution_value * power * 0.05

        # 计算总成本
        total_cost = 0
        for i in range(num_nodes):
            for t in keys:
                node_price = price_1 if i < 6 else (price_2 if i < 12 else (price_3 if i < 24 else price_4))
                total_cost += charging_cars[i, t].solution_value * (
                    power_0_to_5 if i < 6 else power_6_to_39) * node_price

        # 使用pandas将charging_load转换为DataFrame
        df_charging_load = pd.DataFrame(charging_load, columns=keys)

        # 将DataFrame保存为CSV文件
        csv_file_path = 'charging_load.csv'  # 定义CSV文件的名称和路径
        df_charging_load.to_csv(csv_file_path, index_label='节点编号')

        print(f'充电负荷数据已保存到 {csv_file_path}')

        return charging_load, total_cost
    else:
        print("No solution found")
        # 如果没有找到解决方案，则返回空数组和None
        return np.array([]), None



price_1, price_2, price_3, price_4 = 6, 0.21, 0.21, 0.21

charging_load, total_cost = EV_load(price_1, price_2, price_3, price_4)
print(charging_load, total_cost)
