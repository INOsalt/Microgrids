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

def EV_load1(price_1, price_2, price_3, price_4):
    # 初始化模型
    mdl = Model('EV_Charging')

    # 定义每个时刻每个节点的充电汽车数变量
    charging_cars = {(i, t): mdl.integer_var(name=f'charging_cars_{i}_{t}')
                     for i in range(40) for t in np.arange(0, 24.05, 0.05)}
    # 定义与spaced_connection数量相同的变量，变量范围是0到1
    transition_vars = {pair: mdl.continuous_var(lb=0, ub=1, name=f'transition_var_{pair[0]}_{pair[1]}')
                       for pair in spaced_connection}


    for time, matrix in transition_matrices.items():
        for (i, j), var in transition_vars.items():
            sum_parking = matrix[j + 40, 0:40].sum()
            matrix[i + 40, j] += var * sum_parking * matrix[i + 40, j + 40]  # 更新[i+40, j]位置
            matrix[i + 40, j + 40] *= (1 - var * sum_parking)  # 更新[i+40, j+40]位置
            # 对 [j+40, 41] 到 [j+40, 80] 的元素执行除法操作 # 注意：索引应该是 40:80，因为Python的范围是左闭右开
            decrease_sum = 0  # 用于记录减少的总量
            for col in range(40, 80):  # 从第41列到第80列
                original_value = matrix[j + 40, col]
                new_value = original_value / (var * sum_parking)
                decrease_sum += original_value - new_value  # 累加减少的量
                matrix[j + 40, col] = new_value  # 更新元素值
            average_decrease = decrease_sum / 40  # 平均分配给前40列
            for col in range(0, 40):  # 更新前40列
                matrix[j + 40, col] += average_decrease

    # 找出所有满足条件的(i, j, l)组合 三个点都相连
    valid_combinations = set()
    for (i, j) in direct_connection:
        for l in [l for (x, l) in direct_connection if x == i]:
            if (j, l) in direct_connection or (l, j) in direct_connection:
                valid_combinations.add((i, j, l))
    # 为找到的组合创建变量
    combination_vars = {comb: mdl.continuous_var(lb=0, ub=1, name=f"var_{comb[0]}_{comb[1]}_{comb[2]}")
                        for comb in valid_combinations}
    # 更新矩阵
    for time, matrix in transition_matrices.items():
        for (i, j, l), var in combination_vars.items():
            var = var.solution_value  # 获取决策变量的解值
            matrix[i + 40, j] += var * matrix[i + 40, j + 40]  # 更新[i+40,j]
            matrix[i + 40, j] *= (1 - var)  # 更新[i+40,j]
            matrix[i + 40, l] += matrix[i + 40, j] * var  # 更新[i+40,l]

    # 首先，基于direct_connection创建一个映射，记录每个j节点所有的目标节点k
    j_to_k_map = {}
    for (j, k) in direct_connection:
        if j not in j_to_k_map:
            j_to_k_map[j] = [k]
        else:
            j_to_k_map[j].append(k)

    aux_vars = {}  # 存储辅助变量的字典
    aux1_vars = {}
    for time in keys:
        # 确保每个时间点的键在aux1_vars中已经初始化
        if time not in aux1_vars:
            aux1_vars[time] = {}  # 初始化一个空字典用于存储该时间点的变量

    # 创建辅助变量
    for time in keys:
        aux_vars[time] = {}
        for (i, j) in spaced_connection:
            # 定义动态元素的辅助变量
            if (i + 40, j) not in aux_vars[time]:
                aux_vars[time][(i + 40, j)] = mdl.continuous_var(name=f"aux_{time}_{i + 40}_{j}")
            if (i + 40, j + 40) not in aux_vars[time]:
                aux_vars[time][(i + 40, j + 40)] = mdl.continuous_var(name=f"aux_{time}_{i + 40}_{j + 40}")

        # 根据j_to_k_map为direct_connection中的每个(j, k)创建辅助变量
        for j, ks in j_to_k_map.items():
            for k in ks:
                aux1_vars[time][(j + 40, k)] = mdl.continuous_var(name=f"aux1_{time}_{j + 40}_{k}")
                aux1_vars[time][(j + 40, k + 40)] = mdl.continuous_var(name=f"aux1_{time}_{j + 40}_{k + 40}")

    for time, matrix in transition_matrices.items():# time不用额外定义， 这里是表示车可以提前停下
        for (i, j) in spaced_connection:
            # 确定辅助变量与静态矩阵元素之间的关系
            # 1. aux_vars[time][(i+40, j+40)] <= matrix[i+40, j+40]
            mdl.add_constraint(aux_vars[time][(i + 40, j + 40)] <= matrix[i + 40, j + 40])
            # 2. aux_vars[time][(i+40, j+40)] >= 0
            mdl.add_constraint(aux_vars[time][(i + 40, j + 40)] >= 0)
            # 3. aux_vars[time][(i+40, j+40)] + aux_vars[time][(i+40, j)] == matrix[i+40, j+40] + matrix[i+40, j]
            mdl.add_constraint(
                aux_vars[time][(i + 40, j + 40)] + aux_vars[time][(i + 40, j)] == matrix[i + 40, j + 40] + matrix[
                    i + 40, j])
            # 计算 matrix[j+40, 1] 到 matrix[j+40, 40] 的和
            sum_elements = sum(matrix[j + 40, k] for k in range(40))
            # 添加辅助变量的约束 # aux_vars[time][(i+40, j)] >= matrix[i+40, j]
            mdl.add_constraint(aux_vars[time][(i + 40, j)] >= matrix[i + 40, j])
            # aux_vars[time][(i+40, j)] <= sum_elements
            mdl.add_constraint(aux_vars[time][(i + 40, j)] <= sum_elements)

        for j, ks in j_to_k_map.items():
            # 对于每个j及其所有的k，我们需要基于aux_vars[time][(i + 40, j)]计算adjustment
            # 注意：这里我们需要一个逻辑来确定每个j对应的i值，这可能依赖于您的具体业务逻辑
            if (i + 40, j) in aux_vars[time]:  # 确保(i + 40, j)存在于aux_vars中
                num_ks = len(ks)  # j直接连接到的节点总数
                adjustment = (aux_vars[time][(i + 40, j)] - matrix[i + 40, j]) / num_ks
                for k in ks:
                    # 更新辅助变量的约束
                    aux_vars[time][(j + 40, k)] = mdl.continuous_var(name=f"aux_{time}_{j + 40}_{k}")
                    mdl.add_constraint(aux_vars[time][(j + 40, k)] == matrix[j + 40, k] - adjustment)
                    mdl.add_constraint(aux_vars[time][(j + 40, k + 40)] == matrix[j + 40, k + 40] + adjustment)


    # 初始化每个时间点的辅助矩阵
    aux_matrices = {time: [[None for _ in range(80)] for _ in range(80)] for time in keys}

    for time, matrix in transition_matrices.items():
        # 复制静态矩阵的值到辅助矩阵
        for i in range(80):
            for j in range(80):
                if (i, j) not in aux_vars[time]:
                    aux_matrices[time][i][j] = matrix[i, j]

        # 为spaced_connection中的(i, j)放置辅助变量
        for (i, j) in spaced_connection:
            aux_matrices[time][i + 40][j] = aux_vars[time][(i + 40, j)]
            aux_matrices[time][i + 40][j + 40] = aux_vars[time][(i + 40, j + 40)]

        # 为j_to_k_map中的(j, k)放置辅助变量
        for j, ks in j_to_k_map.items():
            for k in ks:
                aux_matrices[time][j + 40][k] = aux1_vars[time][(j + 40, k)]
                if (j + 40, k + 40) in aux1_vars[time]:  # 检查是否存在此辅助变量
                    aux_matrices[time][j + 40][k + 40] = aux1_vars[time][(j + 40, k + 40)]

    # 对于每个时间点和节点，充电车数量不超过最大车辆数
    X = initial_EV.copy()  # 复制初始状态以避免修改原始数据
    for time in keys:
        updated_X = np.dot(X, aux_matrices[time])  # 使用矩阵乘法更新X
        for i in range(40):
            current_val = updated_X[i]  # 获取更新后的单个节点状态
            mdl.add_constraint(charging_cars[i, time] <= current_val, f'max_cars_constraint_{i}_{time}')
        X = updated_X  # 更新X以用于下一个时间步的计算

    # 充电功率约束
    Y = charging_requirement.copy()
    for time in keys:
        updated_Y = np.dot(Y, aux_matrices[time])
        for i in range(40):
            power = power_0_to_5 if i in range(6) else power_6_to_39
            current_val = updated_Y[i]
            mdl.add_constraint(charging_cars[i, time] * power <= current_val, f'max_power_constraint_{i}_{time}')
        Y = updated_Y  # 更新X以用于下一个时间步的计算

    total_charging_demand = np.sum(charging_requirement)

    # 所有充电负荷的和小于充电需求的和
    mdl.add_constraint(
        mdl.sum(
            charging_cars[i, t] * (power_0_to_5 if i < 6 else power_6_to_39)
            for i in range(40) for t in keys  # 确保这里的t与时间点索引一致
        ) == total_charging_demand,
        "total_charging_demand_constraint"
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
    solution = mdl.solve()

    # 检查解决方案是否存在
    if solution:
        # 打印或处理解决方案
        print("Solution found")
    else:
        print("No solution found")

price_1, price_2, price_3, price_4 = 6, 0.21, 0.21, 0.21

EV_load(price_1, price_2, price_3, price_4)
