import pandas as pd
from docplex.mp.model import Model
import numpy as np
import matplotlib.pyplot as plt
from gridinfo import (C_buy, C_sell, start_points, end_points, microgrid_id, EV_penetration, nodes, node_mapping,
                      transition_matrices)

class CommuterChargingChoiceCalculator:
    def __init__(self, home_prices, work_prices, work_slowprices, work_slowprices1, work_slowprices2):
        self.home_prices = home_prices
        self.work_prices = work_prices
        self.num_owners = EV_penetration * 0.1
        self.work_slowprices1 = work_slowprices1
        self.work_slowprices2 = work_slowprices2
        self.work_slowprices = work_slowprices
        #self.home_charging_capacity = 120
        #self.work_charging_capacity = 200

    def calculate_home_charging_cost(self):
        # 选择社区价格数组0-8和21-23中最高的6个数相加
        night_prices = np.concatenate((self.home_prices[0:8], self.home_prices[21:24]))
        highest_prices = np.sort(night_prices)[-6:]
        return np.sum(highest_prices)

    def calculate_private_charging_cost(self):
        # 选择C_buy数组0-8小时之间最高的6个数相加
        night_prices = C_buy[0:8]  # 不需要使用np.concatenate
        highest_prices = np.sort(night_prices)[-6:]
        return np.sum(highest_prices)

    def calculate_workquick_charging_cost(self):
        # 从公司价格数组9-19点中找出最小的一个价格
        daytime_prices = self.work_prices[9:19]
        min_sum = np.min(daytime_prices)
        return min_sum
    def calculate_workslow_charging_cost(self):
        # 选择社区价格数组9-19中最高的6个数相加
        night_prices = np.concatenate(self.work_slowprices[9:19])
        highest_prices = np.sort(night_prices)[-6:]
        return np.sum(highest_prices)
    def calculate_workslow_charging_cost1(self):
        # 选择社区价格数组9-19中最高的6个数相加
        night_prices = np.concatenate(self.work_slowprices1[9:19])
        highest_prices = np.sort(night_prices)[-6:]
        return np.sum(highest_prices)

    def calculate_workslow_charging_cost2(self):
        # 选择社区价格数组9-19中最高的6个数相加
        night_prices = np.concatenate(self.work_slowprices2[9:19])
        highest_prices = np.sort(night_prices)[-6:]
        return np.sum(highest_prices)

    def calculate_choices(self):
        # 计算各种充电方式的成本
        home_charging_cost = self.calculate_home_charging_cost()
        work_quick_charging_cost = self.calculate_workquick_charging_cost()
        work_slow_charging_cost = self.calculate_workslow_charging_cost()
        work_slow_charging_cost1 = self.calculate_workslow_charging_cost1()
        work_slow_charging_cost2 = self.calculate_workslow_charging_cost2()
        private_charging_cost = self.calculate_private_charging_cost()

        # 初始化各充电方式的车主数量
        choices = {
            'charging_at_home_private': self.num_owners // 2,  # 一半车主使用私人充电桩
            'charging_at_home_public': 0,
            'charging_at_work_quick': 0,
            'charging_at_work_slow': 0,
            'charging_at_work_slow1': 0,
            'charging_at_work_slow2': 0,
        }

        # 另一半车主决定充电地点
        decision_making_owners = self.num_owners - choices['charging_at_home_private']

        # 根据不同群体计算充电选择
        for owner_group, fraction in [('group1', 1 / 3), ('group2', 1 / 6), ('group3', 1 / 6), ('group4', 1 / 3)]:
            num_owners_in_group = int(decision_making_owners * fraction)

            if owner_group == 'group1':
                # 四种选择
                options = [home_charging_cost, work_quick_charging_cost, work_slow_charging_cost,
                           work_slow_charging_cost1]
            elif owner_group == 'group2':
                # 四种选择
                options = [home_charging_cost, work_quick_charging_cost, work_slow_charging_cost,
                           work_slow_charging_cost2]
            elif owner_group == 'group3':
                # 五种选择
                options = [home_charging_cost, work_quick_charging_cost, work_slow_charging_cost,
                           work_slow_charging_cost1, work_slow_charging_cost2]
            else:  # group4
                # 三种选择
                options = [home_charging_cost, work_quick_charging_cost, work_slow_charging_cost]

            # 选择成本最低的充电方式
            min_cost_option = options.index(min(options))

            # 根据最低成本选项分配车主
            if min_cost_option == 0:
                choices['charging_at_home_public'] += num_owners_in_group
            elif min_cost_option == 1:
                choices['charging_at_work_quick'] += num_owners_in_group
            elif min_cost_option == 2:
                choices['charging_at_work_slow'] += num_owners_in_group
            elif min_cost_option == 3 and 'work_slow_charging_cost1' in locals():
                choices['charging_at_work_slow1'] += num_owners_in_group
            elif min_cost_option == 4 and 'work_slow_charging_cost2' in locals():
                choices['charging_at_work_slow2'] += num_owners_in_group

        return choices


class CommunityChargingManager:
    def __init__(self, MG1, MGS1, MG2, MG3, MG4):
        self.MG1 = MG1
        self.MGS1 = MGS1
        self.MG2 = MG2
        self.MG3 = MG3
        self.MG4 = MG4
        # 创建节点到索引的映射
        self.node_mapping = node_mapping
        # 初始化向量
        self.charging_home = np.zeros(len(nodes)*2)
        self.charging_work_quick = np.zeros(len(nodes)*2)
        self.charging_work_slow = np.zeros(len(nodes)*2)
        # 初始化特殊点充电车辆数记录字典
        self.special_slow_charging_counts = {201: 0, 207: 0, 205: 0, 301: 0, 311: 0, 312: 0}

    def calculate_charging_distribution(self):
        # 遍历起点，计算充电选择，并更新向量
        for start_point in start_points:
            microgrid = microgrid_id(start_point)
            home_price = self.MG2 if microgrid == 1 else self.MG3 if microgrid == 2 else self.MG4

            # 使用CommuterChargingChoiceCalculator计算选择
            calculator = CommuterChargingChoiceCalculator(home_price, self.MG1, self.MGS1, self.MG2, self.MG3)
            choices = calculator.calculate_choices()

            # 根据映射更新向量
            idx = self.node_mapping[start_point]
            self.charging_home[idx] = choices['charging_at_home_private'] + choices['charging_at_home_public']
            self.charging_work_quick[idx] += choices['charging_at_work_quick']
            # 更新慢充向量
            self.charging_work_slow[idx] += (choices['charging_at_work_slow'] +
                                             choices['charging_at_work_slow1'] +
                                             choices['charging_at_work_slow2'])
            # 特殊处理慢充1和慢充2
            special_slow1 = choices['charging_at_work_slow1'] / 3
            special_slow2 = choices['charging_at_work_slow2'] / 3
            for point in [201, 207, 205]:
                self.special_slow_charging_counts[point] += special_slow1
                self.charging_work_slow[self.node_mapping[point]] += special_slow1
            for point in [301, 311, 312]:
                self.special_slow_charging_counts[point] += special_slow2
                self.charging_work_slow[self.node_mapping[point]] += special_slow2

    def calculate_vehicle_distribution(self):
        # 初始化结果矩阵
        self.charging_home_matrix = np.zeros((80, 48))
        self.charging_work_quick_matrix = np.zeros((80, 48))
        self.charging_work_slow_matrix = np.zeros((80, 48))

        # 用初始状态向量乘以每个转移矩阵
        for t, matrix in transition_matrices.items():
            idx = int(t * 2)  # 将时间转换为索引
            self.charging_home_matrix[:, idx] = np.dot(matrix, self.charging_home)
            self.charging_work_quick_matrix[:, idx] = np.dot(matrix, self.charging_work_quick)
            self.charging_work_slow_matrix[:, idx] = np.dot(matrix, self.charging_work_slow)

        # 截取前40列
        self.charging_home_matrix = self.charging_home_matrix[:, :40]
        self.charging_work_quick_matrix = self.charging_work_quick_matrix[:, :40]
        self.charging_work_slow_matrix = self.charging_work_slow_matrix[:, :40]

        # 根据special_slow_charging_counts修正self.charging_work_slow_matrix和self.charging_home_matrix
        for node, count in self.special_slow_charging_counts.items():
            special_slow_charging_ratio = count / (EV_penetration[node] * 28 / 6)
            node_idx = self.node_mapping[node]

            # 修正self.charging_work_slow_matrix
            self.charging_work_slow_matrix[node_idx, :] *= (1 - special_slow_charging_ratio)

            # 修正self.charging_home_matrix
            self.charging_home_matrix[node_idx, :] += self.charging_work_slow_matrix[node_idx,
                                                      :] * special_slow_charging_ratio

        # 输出三个矩阵
        return self.charging_home_matrix, self.charging_work_quick_matrix, self.charging_work_slow_matrix




# # EVDataVisualizer类: 可视化EV数据和充电模式
# class EVDataVisualizer:
#     def __init__(self):
#         pass
#
#     def printEVData(self, EV, title):
#         # EV数据可视化
#         plt.figure(figsize=(15, 5))
#
#         # EV到达时隙的频数直方图
#         plt.subplot(1, 3, 1)
#         plt.hist(EV['J_c'], bins=96, range=(0.5, 96.5))
#         plt.title('EV Arrival Time Slot Histogram - ' + title)
#         plt.xlabel('Arrival Time Slot')
#         plt.ylabel('Frequency')
#
#         # EV到达时间的频率直方图
#         plt.subplot(1, 3, 2)
#         plt.hist(EV['t_c'], bins=24, range=(0, 24), density=True)
#         plt.title('EV Arrival Time Frequency Histogram - ' + title)
#         plt.xlabel('Arrival Time (Hours)')
#         plt.ylabel('Frequency')
#
#         plt.tight_layout()
#         plt.show()
#
#         # EV电池状态SOC的散点图
#         plt.figure(figsize=(8, 5))
#         plt.scatter(np.arange(len(EV)), EV['SOC_con'], label='SOC Current', s=10)
#         plt.scatter(np.arange(len(EV)), EV['SOC_min'], label='SOC Minimum', s=10)
#         plt.scatter(np.arange(len(EV)), EV['SOC_max'], label='SOC Maximum', s=10)
#         plt.title('EV Battery State of Charge (SOC) Scatter Plot - ' + title)
#         plt.xlabel('EV Index')
#         plt.ylabel('State of Charge (SOC)')
#         plt.legend()
#         plt.show()
#
#     def figureResult(self, P_basic, P_SOC_crd, title):
#         # 充电模式结果可视化
#         plt.figure(figsize=(12, 6))
#         plt.plot(P_basic, label='Basic load')
#         plt.plot(P_SOC_crd, label='Optimized load', linestyle='--')
#         plt.xlabel('Time')
#         plt.ylabel('Load (KW)')
#         plt.title('Charging mode - ' + title)
#         plt.legend()
#         plt.show()
