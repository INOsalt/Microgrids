from docplex.mp.model import Model
import numpy as np
from charging_choice import ChargingManager
from gridinfo import (end_points, nodes, node_mapping, transition_matrices, nodedata_dict,
                      pv_capacity_dict, wt_capacity_dict)

class EVChargingOptimizer:
    def __init__(self):
        self.CAP_BAT_EV = 42  # 固定的每辆车充电需求（70kWh*0.6,70来自平均统计数据）
        self.DELTA_T = 0.5  # 每个时间段长度，30分钟
        self.N_SLOTS = 48  # 一天中的时间段数量
        self.P_slow = 7 # kW
        self.P_quick = 42 # kW
        self.efficiency = 0.9

    def optimizeCommunityChargingPattern(self, community_vehicles_distribution, community_arriving_vehicles,
                                         community_leaving_vehicles, community_P_BASIC):

        model = Model("EVChargingOptimization")

        # 决策变量：每个时隙每辆车的充电状态（-1=放电, 0=不操作, 1=充电）
        charge = model.binary_var_matrix(self.N_SLOTS, max(community_vehicles_distribution), name="charge")
        discharge = model.binary_var_matrix(self.N_SLOTS, max(community_vehicles_distribution), name="discharge")

        # 约束条件
        constraints = []

        # 约束1：每个时隙充电或放电的车辆数不超过该时段车辆数
        for t in range(self.N_SLOTS):
            constraints.append(model.sum(charge[t, i] + discharge[t, i]
                                         for i in range(community_vehicles_distribution[t]))
                               <= community_vehicles_distribution[t])
            for i in range(community_vehicles_distribution[t], max(community_vehicles_distribution)):
                # 对于超出该时间段实际车辆数的索引，此时车不存在，强制充电和放电变量为0
                constraints.append(charge[t, i] == 0)
                constraints.append(discharge[t, i] == 0)

        # 约束2：紧急需求车辆的充电要求
        for t in range(self.N_SLOTS):
            emergency_charging_needed = community_leaving_vehicles[t] * self.CAP_BAT_EV
            net_charging_provided = model.sum(
                (charge[t_prime, i] - discharge[t_prime, i]) * self.P_slow * self.DELTA_T
                for t_prime in range(t + 1)  # 包括当前时段
                for i in range(community_vehicles_distribution[t_prime]))
            constraints.append(net_charging_provided >= emergency_charging_needed)

        # 约束3：总充电量需满足所有车辆的累计需求
        total_charging_demand = sum(community_arriving_vehicles) * self.CAP_BAT_EV
        total_charging_provided = model.sum(charge[t, i] * self.P_slow * self.DELTA_T
                                            for t in range(self.N_SLOTS)
                                            for i in range(community_vehicles_distribution[t]))
        total_discharging_reduced = model.sum(discharge[t, i] * self.P_slow * self.DELTA_T
                                              for t in range(self.N_SLOTS)
                                              for i in range(community_vehicles_distribution[t]))
        # 确保总充电量（考虑放电减少的量）满足总需求
        constraints.append(total_charging_provided - total_discharging_reduced == total_charging_demand)

        # 约束4：累计放电量不超过之前累计的净充电量
        for t in range(0, self.N_SLOTS):
            for i in range(max(community_vehicles_distribution)):
                cumulative_charge_until_t = model.sum(charge[t_prime, i] * self.DELTA_T for t_prime in range(t+1))
                cumulative_discharge_until_t = model.sum(discharge[t_prime, i] * self.DELTA_T for t_prime in range(t+1))
                constraints.append(cumulative_discharge_until_t <= cumulative_charge_until_t)

        model.add_constraints(constraints)

        # 计算每个时段的电网总负载，包括基本负载、充电增加和放电减少
        P_total = [community_P_BASIC[t] +
                   model.sum(charge[t, i] * self.P_slow for i in
                             range(community_vehicles_distribution[t])) -
                   model.sum(discharge[t, i] * self.P_slow for i in
                             range(community_vehicles_distribution[t]))
                   for t in range(self.N_SLOTS)]

        # 目标函数：最小化电网负载的高峰与低谷之间的差异
        model.minimize(model.max(P_total) - model.min(P_total))

        solution = model.solve()

        if solution:
            # 初始化字典来存储每半小时的净功率
            net_power_per_half_hour = []

            for t in range(self.N_SLOTS):
                # 计算每个时段的充电功率总和
                charging_power = sum(solution.get_value(charge[t, i]) * self.P_slow / self.efficiency
                                     for i in range(community_vehicles_distribution[t]))
                # 计算每个时段的放电功率总和
                discharging_power = sum(solution.get_value(discharge[t, i]) * self.P_slow * self.efficiency
                                        for i in range(community_vehicles_distribution[t]))
                # 计算EV净功率：充电功率 - 放电功率
                net_power = charging_power - discharging_power

                # 将时间段和净功率添加到字典中
                net_power_per_half_hour[t] = net_power

            return net_power_per_half_hour
        else:
            print("No solution found.")
            return {}

    def optimizeOfficeChargingPattern(self, slow_vehicles_distribution, slow_arriving_vehicles, slow_leaving_vehicles,
                                      fast_vehicles_distribution, office_P_BASIC):
        model = Model("OfficeEVChargingOptimization")

        # 决策变量
        # 慢充车辆充电状态
        slow_charge = model.binary_var_matrix(self.N_SLOTS, max(slow_vehicles_distribution), name="slow_charge")
        # 慢充车辆放电状态
        slow_discharge = model.binary_var_matrix(self.N_SLOTS, max(slow_vehicles_distribution), name="slow_discharge")
        # 快充车辆没有变量因为只要在就一定充电

        # 约束条件
        constraints = []

        # 约束1：每个时隙充电或放电的车辆数不超过该时段车辆数
        # 慢充车辆约束
        for t in range(self.N_SLOTS):
            # 充电或放电的慢充车辆数不超过该时段车辆数
            constraints.append(model.sum(slow_charge[t, i] + slow_discharge[t, i]
                                         for i in range(slow_vehicles_distribution[t]))
                               <= slow_vehicles_distribution[t])
            for i in range(slow_vehicles_distribution[t], max(slow_vehicles_distribution)):
                # 对于超出该时间段实际车辆数的索引，强制充电和放电变量为0
                constraints.append(slow_charge[t, i] == 0)
                constraints.append(slow_discharge[t, i] == 0)

        # 约束2：紧急需求车辆的充电要求
        # 慢充车辆约束
        for t in range(self.N_SLOTS):
            emergency_charging_needed = slow_leaving_vehicles[t] * self.CAP_BAT_EV
            net_charging_provided = model.sum(
                (slow_charge[t_prime, i] - slow_discharge[t_prime, i]) * self.P_slow * self.DELTA_T
                for t_prime in range(t + 1)  # 包括当前时段
                for i in range(slow_vehicles_distribution[t_prime]))
            constraints.append(net_charging_provided >= emergency_charging_needed)

        # 约束3：总充电量需满足所有车辆的累计需求
        # 慢充车辆约束
        total_charging_demand = sum(slow_arriving_vehicles) * self.CAP_BAT_EV
        total_charging_provided = model.sum(slow_charge[t, i] * self.P_slow * self.DELTA_T
                                            for t in range(self.N_SLOTS)
                                            for i in range(slow_vehicles_distribution[t]))
        total_discharging_reduced = model.sum(slow_discharge[t, i] * self.P_slow * self.DELTA_T
                                              for t in range(self.N_SLOTS)
                                              for i in range(slow_vehicles_distribution[t]))
        # 确保总充电量（考虑放电减少的量）满足总需求
        constraints.append(total_charging_provided - total_discharging_reduced == total_charging_demand)

        # 约束4：累计放电量不超过之前累计的净充电量
        # 约束4：累计放电量不超过之前累计的净充电量
        for t in range(self.N_SLOTS):
            for i in range(max(slow_vehicles_distribution)):  # 根据慢充车辆分布的最大值遍历
                cumulative_charge_until_t = model.sum(
                    slow_charge[t_prime, i] * self.DELTA_T for t_prime in range(t + 1))  # 包括当前时段
                cumulative_discharge_until_t = model.sum(
                    slow_discharge[t_prime, i] * self.DELTA_T for t_prime in range(t + 1))  # 包括当前时段
                constraints.append(cumulative_discharge_until_t <= cumulative_charge_until_t)

        model.add_constraints(constraints)

        # 目标函数
        # 计算每个时段的电网总负载，考虑慢充充电、慢充放电和快充充电
        P_total = [office_P_BASIC[t] +
                   model.sum(slow_charge[t, i] * self.P_slow - slow_discharge[t, i] * self.P_slow for i in
                             range(slow_vehicles_distribution[t])) +
                   fast_vehicles_distribution[t] * self.P_quick  # 快充车辆在场即充电
                   for t in range(self.N_SLOTS)]

        model.minimize(model.max(P_total) - model.min(P_total))

        solution = model.solve()

        if solution:
            # 初始化字典来存储每半小时的净功率
            net_power_per_half_hour = []

            for t in range(self.N_SLOTS):
                # 计算每个时段的充电功率总和
                charging_power = sum(solution.get_value(slow_charge[t, i]) * self.P_slow / self.efficiency
                                     for i in range(slow_vehicles_distribution[t]))
                # 计算每个时段的放电功率总和
                discharging_power = sum(solution.get_value(slow_discharge[t, i]) * self.P_slow * self.efficiency
                                        for i in range(slow_vehicles_distribution[t]))
                # 计算EV净功率：充电功率 - 放电功率
                net_power = (charging_power - discharging_power +
                             fast_vehicles_distribution[t] * self.P_quick / self.efficiency) #快充
                # 将时间段和净功率添加到字典中
                net_power_per_half_hour[t] = net_power

            return net_power_per_half_hour
        else:
            print("No solution found.")


def extract_charging_distribution(charging_home_matrix, charging_work_slow_matrix, charging_work_quick_matrix):
    # 初始化存储结果的字典
    charging_distribution_work_slow = {}
    charging_distribution_work_quick = {}
    charging_distribution_home = {}

    # 获取除了end_points之外的所有节点的索引
    non_end_point_indices = [idx for node, idx in node_mapping.items() if node not in end_points]

    # 反向映射：从矩阵索引到节点编号
    reverse_node_mapping = {idx: node for node, idx in node_mapping.items()}

    # 处理charging_work_slow_matrix，为end_points对应的列提取数据
    for end_point in end_points:
        idx = node_mapping[end_point]
        charging_distribution_work_slow[end_point] = charging_work_slow_matrix[idx, :]

    # 处理charging_work_quick_matrix，为end_points对应的列提取数据
    for end_point in end_points:
        idx = node_mapping[end_point]
        charging_distribution_work_quick[end_point] = charging_work_quick_matrix[idx, :]

    # 处理charging_home_matrix，为除了end_points之外的节点提取数据
    for idx in non_end_point_indices:
        node = reverse_node_mapping[idx]
        charging_distribution_home[node] = charging_home_matrix[idx, :]

    return charging_distribution_work_slow, charging_distribution_work_quick, charging_distribution_home


def calculate_leaving_vehicles(charging_distribution):
    leaving_vehicles = {}

    for node, charging_vector in charging_distribution.items():
        node_idx = node_mapping[node]  # 获取节点在转移矩阵中的索引
        leaving_vector = np.zeros_like(charging_vector)
        for i, vehicles_at_time in enumerate(charging_vector):
            transition_matrix = transition_matrices[int(i / 2)]  # 获取对应的转移矩阵
            Pjj = transition_matrix[node_idx, node_idx]  # 获取节点自身的转移概率
            leaving_vector[i] = (1 - Pjj) * vehicles_at_time  # 计算离开的车辆数量
        leaving_vehicles[node] = leaving_vector

    return leaving_vehicles


def calculate_arriving_vehicles(charging_distribution, leaving_vehicles):
    # 四舍五入后的字典
    charging_distribution_round = {node: np.round(vector).astype(int) for node, vector in charging_distribution.items()}
    leaving_vehicles_round = {node: np.round(vector).astype(int) for node, vector in leaving_vehicles.items()}

    # 计算到达的车辆
    arriving_vehicles = {}
    for node, vector in charging_distribution_round.items():
        arriving_vector = np.zeros_like(vector)
        for i in range(len(vector) - 1):
            arriving_vector[i] = vector[i + 1] - vector[i] + leaving_vehicles_round[node][i]
        # 最后一个时间点的到达车辆设置为0
        arriving_vector[-1] = 0
        arriving_vehicles[node] = np.round(arriving_vector).astype(int)

    return charging_distribution_round, leaving_vehicles_round, arriving_vehicles

def calculate_P_basic():
    P_basic_dict = {}
    for hour in range(1, 25):
        load_matrix = nodedata_dict[hour]
        pv_matrix = pv_capacity_dict.get(hour, np.zeros_like(load_matrix))
        wt_matrix = wt_capacity_dict.get(hour, np.zeros_like(load_matrix))

        for node in nodes:
            node_index = np.where(load_matrix[:, 0] == node)[0][0]
            load = load_matrix[node_index, 1]
            pv = pv_matrix[node_index, 1] if hour in pv_capacity_dict else 0
            wt = wt_matrix[node_index, 1] if hour in wt_capacity_dict else 0

            net_load = load - (pv + wt)
            if node not in P_basic_dict:
                P_basic_dict[node] = [net_load] * 2  # Each hour split into two half-hour periods
            else:
                P_basic_dict[node].extend([net_load] * 2)

    return P_basic_dict

def EVload(EV_Q1, EV_S1, EV_2, EV_3, EV_4):
    # 初始化存储每半小时所有节点EV负荷的字典
    node_EV_load = {time: np.zeros(len(nodes)) for time in range(48)}
    # 初始化存储每半小时微电网EV负荷的字典
    mic_EV_load = {time: np.zeros(4) for time in range(48)}  # 4个微电网

    # 充电选择实例
    EV_choice = ChargingManager(EV_Q1, EV_S1, EV_2, EV_3, EV_4)
    # 充电矩阵
    charging_home_matrix, charging_work_slow_matrix, charging_work_quick_matrix \
        = EV_choice.calculate_vehicle_distribution()
    # 处理为字典
    charging_distribution_work_slow, charging_distribution_work_quick, charging_distribution_home \
        = extract_charging_distribution(charging_home_matrix, charging_work_slow_matrix, charging_work_quick_matrix)
    #计算离开和到达
    # 工作慢充
    leaving_vehicles_work_slow = calculate_leaving_vehicles(charging_distribution_work_slow)
    work_slow_charging_distribution, work_slow_leaving, work_slow_arriving\
        = calculate_arriving_vehicles(charging_distribution_work_slow, leaving_vehicles_work_slow)
    # 家慢充
    leaving_vehicles_home = calculate_leaving_vehicles(charging_distribution_work_slow)
    home_charging_distribution, home_leaving, home_arriving \
        = calculate_arriving_vehicles(charging_distribution_home, leaving_vehicles_home)
    # 工作快充
    work_quick_charging_distribution = {node: np.round(vector).astype(int) for node, vector
                                        in charging_distribution_work_quick.items()}
    # Pbasic字典
    P_basic_dict = calculate_P_basic()
    #优化实例
    optimize = EVChargingOptimizer()
    for node in nodes:
        if 100 <= node <= 199:  # Office节点
            P_basic = P_basic_dict[node]
            ev_load_vector = optimize.optimizeOfficeChargingPattern(
                slow_vehicles_distribution=work_slow_charging_distribution[node],
                slow_arriving_vehicles=work_slow_arriving[node],
                slow_leaving_vehicles=work_slow_leaving[node],
                fast_vehicles_distribution=work_quick_charging_distribution.get(node, []),
                office_P_BASIC=P_basic
            )
        else:  # 社区节点
            P_basic = P_basic_dict[node]
            ev_load_vector = optimize.optimizeCommunityChargingPattern(
                community_vehicles_distribution=home_charging_distribution[node],
                community_arriving_vehicles=home_arriving[node],
                community_leaving_vehicles=home_leaving[node],
                community_P_BASIC=P_basic
            )
        # 获取当前节点的索引
        node_idx = node_mapping[node]

        # 使用该节点的EV负荷更新node_EV_load字典中的向量
        for t in range(48):
            node_EV_load[t][node_idx] = ev_load_vector[t]

    # 对每个时间段，按照节点范围汇总微电网的EV负荷
    for t in range(48):
        for node in nodes:
            # 获取当前节点的索引
            node_idx = node_mapping[node]
            # 根据节点范围，更新对应微电网的负荷
            if 100 <= node <= 199:
                mic_EV_load[t][0] += node_EV_load[t][node_idx]
            elif 200 <= node <= 299:
                mic_EV_load[t][1] += node_EV_load[t][node_idx]
            elif 300 <= node <= 399:
                mic_EV_load[t][2] += node_EV_load[t][node_idx]
            elif 400 <= node <= 499:
                mic_EV_load[t][3] += node_EV_load[t][node_idx]

    return node_EV_load, mic_EV_load


# # 假定的示例数据
# vehicles_distribution = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245]
# arriving_vehicles = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240]
# leaving_vehicles = [3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58, 63, 68, 73, 78, 83, 88, 93, 98, 103, 108, 113, 118, 123, 128, 133, 138, 143, 148, 153, 158, 163, 168, 173, 178, 183, 188, 193, 198, 203, 208, 213, 218, 223, 228, 233, 238]
# P_BASIC = [1.5] * 48  # 假设每个时隙的基本负载相同
#
# # 创建优化器实例并执行优化
# optimizer = EVChargingOptimizer(vehicles_distribution, arriving_vehicles, leaving_vehicles, P_BASIC)
# charging_pattern = optimizer.optimizeChargingPattern()
# print(charging_pattern)

# 注意：具体数据需要根据实际情况填写或修改
