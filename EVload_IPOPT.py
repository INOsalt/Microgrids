from pyomo.environ import *
import pandas as pd
import os
import numpy as np
#import pyomo.environ as pyo


# 初始化
EV_penetration = 1000 #辆/每节点
power_0_to_5 = 60  # kW 快充
power_6_to_39 = 7  # kW 慢充

# 读取矩阵
# 设置包含CSV文件的文件夹路径
folder_path = 'TMhalfhour'  # 你需要替换为实际的路径

# 初始化一个空字典来存储矩阵
transition_matrices = {}
# 生成步长为0.05的序列从0到24
keys = np.arange(0, 24, 0.5)

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
    # 初始化Pyomo模型
    model = ConcreteModel()
    to_points = [7.5, 8, 8.5, 9, 9.5, 10, 10.5]  # 表示时间点的集合 7以前和初始保持一致
    # 定义每个时刻每个节点的充电汽车数变量
    # 定义变量
    model.charging_cars = Var(range(num_nodes), to_points, within=NonNegativeReals)
    # 定义辅助变量
    # 定义索引集合
    index_set = []
    for time in to_points:
        for (i, j) in spaced_connection:
            index_set.append((time, i, j))
    # 定义辅助变量，限定在0到1之间
    model.aux_vars = Var(index_set, within=NonNegativeReals, bounds=(0, 1))

    # 定义时间点和索引
    model.T = RangeSet(len(to_points))  # 时间点的索引
    model.I = RangeSet(80)  # 状态向量和矩阵的维度

    # 假设 initial_state 和 transition_matrices 已经定义
    model.currentState = Var(model.I, model.T, within=NonNegativeReals)

    def update_rule(model, i, t):
        if t == 1:  # 假设t=1对应于to_points中的第一个时间点
            # 初始状态更新规则
            return model.currentState[i, t] == sum(
                initial_EV[j] * transition_matrices[to_points[0]][j, i] for j in model.I)
        else:
            # 后续状态更新规则，基于前一个时间点的状态
            return model.currentState[i, t] == sum(
                model.currentState[j, t - 1] * transition_matrices[to_points[t - 1]][j, i] for j in model.I)

    model.UpdateConstraint = Constraint(model.I, model.T, rule=update_rule)

    # 计算每个时间点的最大充电车辆数
    def compute_max_EV_at_time(model, time):
        current_EV = initial_EV
        for t in to_points:
            if t > time:
                break
            # 获取当前时间点的转移矩阵，并根据aux_vars更新
            matrix = transition_matrices[t]
            for i, j in spaced_connection:
                matrix[i + 40, j + 40] = matrix[i + 40, j + 40] * model.aux_vars[t, i, j]
                matrix[i + 40, j] = matrix[i + 40, j + 40] * (1 - model.aux_vars[t, i, j])
            # 更新current_EV
            current_EV = np.dot(current_EV, matrix)[:num_nodes]
        return current_EV

    # 加入充电车数量约束
    def charging_cars_constraint(model, node, time):
        max_EV = compute_max_EV_at_time(model, time)
        return sum(model.charging_cars[node, t] for t in to_points if t <= time) <= max_EV[node]

    # 应用约束
    model.charging_cars_constraint = Constraint([(node, time) for node in range(num_nodes) for time in to_points],
                                                rule=charging_cars_constraint)

    # 计算每个时间点的最大充电车辆数
    def compute_max_EV_at_time(model, time):
        current_EV = initial_EV
        for t in to_points:
            if t > time:
                break
            # 获取当前时间点的转移矩阵，并根据aux_vars更新
            matrix = transition_matrices[t]
            for i, j in spaced_connection:
                matrix[i + 40, j + 40] = matrix[i + 40, j + 40] * model.aux_vars[t, i, j]
                matrix[i + 40, j] = matrix[i + 40, j + 40] * (1 - model.aux_vars[t, i, j])
            # 更新current_EV
            current_EV = np.dot(current_EV, matrix)[:num_nodes]
        return current_EV

    # 加入充电车数量约束
    def charging_cars_constraint(model, node, time):
        max_EV = compute_max_EV_at_time(model, time)
        return sum(model.charging_cars[node, t] for t in to_points if t <= time) <= max_EV[node]

    # 应用约束
    model.charging_cars_constraint = Constraint([(node, time) for node in range(num_nodes) for time in to_points],
                                                rule=charging_cars_constraint)

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

    total_charging_demand = np.sum(charging_requirement)

    # 添加约束：所有充电负荷的和小于等于充电需求的总和
    def total_demand_constraint(model):
        return summation(
            [model.charging_cars[i, t] * (power_0_to_5 if i < 6 else power_6_to_39) for i in model.I for t in
             model.T]) <= total_charging_demand

    model.total_demand_constr = Constraint(rule=total_demand_constraint)

    def objective_rule(model):
        return sum(model.charging_cars[i, t] * price_mapping[i] for i in model.I for t in model.T)

    model.objective = Objective(rule=objective_rule, sense=minimize)

    # 使用求解器求解模型
    solver = SolverFactory('ipopt')  # 这里使用Ipopt作为求解器，确保已安装
    solution = solver.solve(model)

    # 检查解决方案是否存在
    if solution.solver.status == SolverStatus.ok and solution.solver.termination_condition == TerminationCondition.optimal:
        # 打印或处理解决方案
        print("Solution found")
        # 打印变量的解
        for i in model.I:
            for t in model.T:
                print(f"Charging cars at node {i} at time {t}: {value(model.charging_cars[i, t])}")
    else:
        print("No solution found or the solution is not optimal")

price_1, price_2, price_3, price_4 = 6, 0.21, 0.21, 0.21

EV_load(price_1, price_2, price_3, price_4)
