import numpy as np
import networkx as nx
from collections import Counter
import copy
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import os
from scipy.stats import truncnorm

# 路网结构
num_nodes = 40
LJ = np.zeros((num_nodes, num_nodes))  # 初始化邻接矩阵
# 定义节点列表
nodes = [101, 102, 103, 104, 105, 106, 201, 202, 203, 204, 205, 206, 207, 208, 209, 301, 302, 303, 304, 305, 306,
         307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 401, 402, 403, 404, 405, 406, 407]

# 创建一个从节点编号到索引的映射
node_mapping = {node: index for index, node in enumerate(nodes)}

# 使用映射来创建边列表
edges = [
    (node_mapping[101], node_mapping[105]),
    (node_mapping[101], node_mapping[106]),
    (node_mapping[105], node_mapping[106]),
    (node_mapping[105], node_mapping[102]),
    (node_mapping[105], node_mapping[103]),
    (node_mapping[106], node_mapping[103]),
    (node_mapping[102], node_mapping[103]),
    (node_mapping[103], node_mapping[104]),
    (node_mapping[106], node_mapping[104]),
    (node_mapping[104], node_mapping[205]),
    (node_mapping[104], node_mapping[311]),
    (node_mapping[106], node_mapping[312]),
    (node_mapping[102], node_mapping[201]),
    (node_mapping[103], node_mapping[207]),
    (node_mapping[201], node_mapping[202]),
    (node_mapping[201], node_mapping[207]),
    (node_mapping[202], node_mapping[203]),
    (node_mapping[203], node_mapping[204]),
    (node_mapping[204], node_mapping[205]),
    (node_mapping[205], node_mapping[207]),
    (node_mapping[202], node_mapping[206]),
    (node_mapping[203], node_mapping[207]),
    (node_mapping[204], node_mapping[208]),
    (node_mapping[208], node_mapping[209]),
    (node_mapping[104], node_mapping[301]),
    (node_mapping[301], node_mapping[302]),
    (node_mapping[301], node_mapping[310]),
    (node_mapping[302], node_mapping[309]),
    (node_mapping[302], node_mapping[311]),
    (node_mapping[309], node_mapping[310]),
    (node_mapping[310], node_mapping[204]),
    (node_mapping[309], node_mapping[313]),
    (node_mapping[313], node_mapping[315]),
    (node_mapping[315], node_mapping[307]),
    (node_mapping[311], node_mapping[312]),
    (node_mapping[311], node_mapping[314]),
    (node_mapping[205], node_mapping[310]),
    (node_mapping[302], node_mapping[303]),
    (node_mapping[303], node_mapping[314]),
    (node_mapping[303], node_mapping[305]),
    (node_mapping[304], node_mapping[305]),
    (node_mapping[305], node_mapping[306]),
    (node_mapping[306], node_mapping[307]),
    (node_mapping[307], node_mapping[308]),
    (node_mapping[303], node_mapping[313]),
    (node_mapping[304], node_mapping[314]),
    (node_mapping[305], node_mapping[315]),
    (node_mapping[315], node_mapping[316]),
    (node_mapping[306], node_mapping[317]),
    (node_mapping[307], node_mapping[318]),
    (node_mapping[318], node_mapping[404]),
    (node_mapping[316], node_mapping[403]),
    (node_mapping[316], node_mapping[318]),
    (node_mapping[310], node_mapping[401]),
    (node_mapping[313], node_mapping[401]),
    (node_mapping[315], node_mapping[402]),
    (node_mapping[208], node_mapping[401]),
    (node_mapping[401], node_mapping[402]),
    (node_mapping[402], node_mapping[403]),
    (node_mapping[402], node_mapping[405]),
    (node_mapping[403], node_mapping[406]),
    (node_mapping[403], node_mapping[404]),
    (node_mapping[404], node_mapping[407]),
    (node_mapping[406], node_mapping[407]),
    (node_mapping[405], node_mapping[406]),
    (node_mapping[401], node_mapping[405]),
]

# 填充邻接矩阵
for edge in edges:
    LJ[edge[0], edge[1]] = 1
    LJ[edge[1], edge[0]] = 1

# # 设置固定的随机种子
# fixed_seed = 31
# np.random.seed(fixed_seed)

# 创建无向图并添加边
G = nx.Graph()
G.add_edges_from(edges)

# 为每个节点生成一个标签字典
labels = {node_mapping[node]: str(node) for node in nodes}


# # 使用spring_layout算法，这通常会产生一个网状的布局
# pos = nx.spring_layout(G, iterations=500)  # 增加迭代次数以优化布局
#
# # 绘制图形，设置背景透明
# plt.figure(figsize=(12, 12), facecolor='none')  # facecolor='none' 可能在某些情况下不起作用
# nx.draw(G, pos, with_labels=False, node_size=700, node_color='lightblue', font_size=10, font_weight='bold')
# nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
#
# # 添加标题
# plt.title('Network Layout')
#
# # 显示图形
# plt.show()

class Markov:
    def __init__(self, G, labels, start_points, end_points):
        # 无向图
        self.G = G
        self.G_now = copy.deepcopy(G)
        self.G_peak = copy.deepcopy(G)  # 创建G的深拷贝用于高峰期
        self.G_offpeak = copy.deepcopy(G)  # 创建G的深拷贝用于非高峰期
        self.labels = labels
        # self.start_points = start_points
        # self.end_points = end_points
        self.departure_distributions = {}
        self.back_distributions = {}
        # 创建一个空的Counter对象来存储边的出现次数
        self.edge_counts = Counter()
        # 初始化出发概率的数组
        self.departure_step = np.zeros(num_nodes)
        self.back_step = np.zeros(num_nodes)
        # 初始化停留概率的数组
        self.departure_stop = np.zeros(num_nodes)
        self.back_stop = np.zeros(num_nodes)
        # 初始化转移矩阵为零矩阵
        self.TM_departure = np.zeros((num_nodes, num_nodes))
        self.TM_back = np.zeros((num_nodes, num_nodes))
        self.TM = np.zeros((num_nodes, num_nodes))
        # 创建起点和终点的映射，这些映射将原始标识符映射到图中的索引
        self.start_mapping = [node_mapping[point] for point in start_points]
        self.end_mapping = [node_mapping[point] for point in end_points]
        self.transition_matrices = {}
        self.steady_states = {}

    def road_weight(self):# 生成权重 高峰和非高峰

        np.random.seed(0)  # 设置随机种子

        # 生成饱和度矩阵
        peak_saturations = np.random.normal(0.6, 0.2, (num_nodes, num_nodes))
        off_peak_saturations = np.random.normal(0.3, 0.1, (num_nodes, num_nodes))

        peak_saturations = np.clip(peak_saturations, 0, 1)
        off_peak_saturations = np.clip(off_peak_saturations, 0, 0.8)

        # 初始化权重矩阵
        W_peak = np.zeros_like(peak_saturations)
        W_off_peak = np.zeros_like(off_peak_saturations)

        # 计算高峰期和非高峰期的权重
        for i in range(num_nodes):
            for j in range(num_nodes):
                if LJ[i, j] == 1:  # 假设LJ是邻接矩阵，标示节点间是否直接相连
                    # 高峰期权重计算
                    W_peak[i, j] = self.calculate_weight(peak_saturations[i, j])
                    # 非高峰期权重计算
                    W_off_peak[i, j] = self.calculate_weight(off_peak_saturations[i, j])

        # 高峰
        for i in range(len(W_peak)):
            for j in range(len(W_peak[i])):
                if self.G_peak.has_edge(i, j):
                    # 假设边的权重可以通过这种方式更新
                    self.G_peak[i][j]['weight'] = W_peak[i, j]
        # 非高峰
        for i in range(len(W_off_peak)):
            for j in range(len(W_off_peak[i])):
                if self.G_offpeak.has_edge(i, j):
                    # 假设边的权重可以通过这种方式更新
                    self.G_offpeak[i][j]['weight'] = W_off_peak[i, j]

    def calculate_weight(self, saturation):
        # 根据饱和度计算单个权重
        # 初始化参数
        alpha = 1.3
        beta = 1.2
        t0 = 10
        c = 30
        lamda = 0.7
        q = 0.8

        if saturation <= 1:
            Rv = t0 * (1 + alpha * saturation ** beta)
        else:
            Rv = t0 * (1 + alpha * (2 - saturation) ** beta)

        if saturation <= 0.6:
            Cv = 0.9 * (c * (1 - lamda) ** 2 / 2 / (1 - lamda * saturation) + saturation ** 2 / 2 / q / (
                        1 - saturation))
        else:
            Cv = c * (1 - lamda) ** 2 / 2 / (2 - lamda * saturation) + 1.5 * (saturation - 0.6) * saturation / (
                        2 - saturation)

        return Rv + 2 * Cv


    def normal_distribution(self):
        # 定义时间的上下限
        lower, upper = 0, 24
        # 出发时间分布
        departure_time = {}

        for start in self.start_mapping:
            for end in self.end_mapping:
                try:
                    # 假设self.G是要用来计算最短路径的图对象
                    path = nx.shortest_path(self.G_now, source=start, target=end)
                    # 记录路径，使用映射后的标签
                    departure_time[(start, end)] = path
                except nx.NetworkXNoPath:
                    print(f"No path found from {start} to {end}.")
                except KeyError:
                    print(f"One of the nodes {start} or {end} does not exist.")

        # 计算路径长度
        path_lengths = {key: len(path) for key, path in departure_time.items()}

        for start_point in self.start_mapping:
            transfer_counts = [path_lengths[(start_point, node_mapping[end])] for end in end_points if
                               (start_point, node_mapping[end]) in path_lengths]
            if transfer_counts:
                average_transfer_count = np.mean(transfer_counts)
                mu = 9 - average_transfer_count * 0.05
                sigma = 1
                # 转换均值和标准差为截断正态分布的参数
                a, b = (lower - mu) / sigma, (upper - mu) / sigma
                # 为每个起点创建截断正态分布的出发时间
                self.departure_distributions[start_point] = truncnorm(a, b, loc=mu, scale=sigma)

        # 返程时间分布
        self.back_distributions = {}

        for start_point in self.end_mapping:
            mu = 18
            sigma = 1
            a, b = (lower - mu) / sigma, (upper - mu) / sigma
            self.back_distributions[start_point] = truncnorm(a, b, loc=mu, scale=sigma)

        # 出发车队分布
        # 初始化两个长度为40的数列
        self.departure_car = np.zeros(num_nodes, dtype=int)
        self.back_car = np.zeros(num_nodes, dtype=int)

        # 对于self.departure_car，如果第i个节点是起点，那么值为self.end_mapping的长度
        for start in self.start_mapping:
            self.departure_car[start] = len(self.end_mapping)

        # 对于self.back_car，如果第i个节点是终点，那么值为self.start_mapping的长度
        for end in self.end_mapping:
            self.back_car[end] = len(self.start_mapping)

    def time_possibility(self, time):

        for i in range(num_nodes):  # 遍历40个节点
            # 对于出发概率
            if i in self.departure_distributions:
                # 获取节点的正态分布
                norm_dist = self.departure_distributions[i]
                # 计算在时间x的PDF值并存储在向量中
                self.departure_step[i] = norm_dist.pdf(time)
                # 计算停留概率并存储在向量中，使用1 - CDF(time)来计算
                self.departure_stop[i] = 1 - norm_dist.cdf(time)
            else:
                # 如果没有为节点定义正态分布，出发概率记作0
                self.departure_step[i] = 0
                self.departure_stop[i] = 1

            # 对于返回概率
            if i in self.back_distributions:
                # 获取节点的正态分布
                norm_dist = self.back_distributions[i]
                # 计算在时间x的PDF值并存储在向量中
                self.back_step[i] = norm_dist.pdf(time)
                # 计算停留概率并存储在向量中，使用1 - CDF(time)来计算
                self.back_stop[i] = 1 - norm_dist.cdf(time)
            else:
                # 如果没有为节点定义正态分布
                self.back_step[i] = 0
                self.back_stop[i] = 1

    def calculate_arrival_distributions(self):
        self.departure_arrive = {}  # 存储从起点到终点的到达时间分布
        self.back_arrive = {}  # 存储从终点返回起点的到达时间分布

        edge_time = 0.05  # 每条边的时间

        # 处理出发到达分布
        for end in self.end_mapping:
            total_mean = 0
            total_variance = 0
            count = 0

            for start in self.start_mapping:
                # 计算路径长度
                path_length = nx.shortest_path_length(self.G_now, source=start, target=end, weight='time')
                # 计算总的转移时间
                transfer_time = path_length * edge_time

                # 获取起点的出发时间分布
                if start in self.departure_distributions:
                    departure_distribution = self.departure_distributions[start]
                    mean = departure_distribution.mean()
                    variance = departure_distribution.var()

                    # 计算平移后的均值和方差
                    shifted_mean = mean + transfer_time
                    shifted_variance = variance  # 方差在平移过程中不变

                    # 累加均值和方差
                    total_mean += shifted_mean
                    total_variance += shifted_variance
                    count += 1

            if count > 0:
                # 计算合并后的均值和方差
                combined_mean = total_mean / count
                combined_variance = total_variance / count

                # 存储合并后的分布
                self.departure_arrive[end] = norm(loc=combined_mean, scale=np.sqrt(combined_variance))

        # 处理返回起点的分布，逻辑与上述相同，只是起点和终点交换
        for start in self.start_mapping:
            total_mean = 0
            total_variance = 0
            count = 0

            for end in self.end_mapping:
                path_length = nx.shortest_path_length(self.G_now, source=end, target=start, weight='time')
                transfer_time = path_length * edge_time

                if end in self.back_distributions:
                    back_distribution = self.back_distributions[end]
                    mean = back_distribution.mean()
                    variance = back_distribution.var()

                    shifted_mean = mean + transfer_time
                    shifted_variance = variance

                    total_mean += shifted_mean
                    total_variance += shifted_variance
                    count += 1

            if count > 0:
                combined_mean = total_mean / count
                combined_variance = total_variance / count

                self.back_arrive[start] = norm(loc=combined_mean, scale=np.sqrt(combined_variance))

    def arrive_possibility(self, time):
        # 初始化存储特定时间点PDF值的字典
        self.departure_arrive_pdf = {}
        self.back_arrive_pdf = {}

        for i in range(num_nodes):  # 遍历40个节点
            # 对于出发到达概率
            if i in self.departure_arrive:
                # 获取节点的正态分布
                norm_dist = self.departure_arrive[i]
                # 计算在时间x的PDF值并存储在另一个字典中
                self.departure_arrive_pdf[i] = norm_dist.pdf(time)
            else:
                # 如果没有为节点定义正态分布，出发到达概率记作0
                self.departure_arrive_pdf[i] = 0

            # 对于返回到达概率
            if i in self.back_arrive:
                # 获取节点的正态分布
                norm_dist = self.back_arrive[i]
                # 计算在时间x的PDF值并存储在另一个字典中
                self.back_arrive_pdf[i] = norm_dist.pdf(time)
            else:
                # 如果没有为节点定义正态分布，返回到达概率记作0
                self.back_arrive_pdf[i] = 0
    def update_graph_weights(self, time):  # 更新self.G

        # 确定当前是否为高峰期
        is_peak = (7 <= time < 9) or (17 <= time < 19)

        # 选择相应的权重矩阵
        self.G = self.G_peak if is_peak else self.G_offpeak

    def departure_probability(self):
        # 存储每个起点到终点的最短路径的字典
        shortest_paths_departure = {}
        # 初始化每个节点出发的边的数量的统计为长度为40的零数组
        self.efn_departure = np.zeros(num_nodes, dtype=int)

        for start in self.start_mapping:
            for end in self.end_mapping:
                # 使用NetworkX计算最短路径
                try:
                    path = nx.shortest_path(self.G, source=start, target=end, weight='weight')
                    # 记录路径
                    shortest_paths_departure[(start, end)] = path
                except nx.NetworkXNoPath:
                    print(f"No path found from {start} to {end}.")
                except KeyError:
                    print(f"One of the nodes {start} or {end} does not exist.")

        # 根据最短路径记录每条路径所经过的边。
        path_edges = {}
        for (start, end), path in shortest_paths_departure.items():
            edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            path_edges[(start, end)] = edges
            # 更新每个节点出发的边的数量
            for edge in edges:
                source = edge[0]
                # 直接使用源节点的索引来更新数组
                self.efn_departure[source] += 1

        # 统计每个节点出发的边的次数并更新转移矩阵
        for path in path_edges.values():
            for edge in path:
                source, target = edge
                self.TM_departure[source, target] += 1

        # 将计数转换为概率
        for i in range(num_nodes):
            total_edges_from_node = np.sum(self.TM_departure[i, :])
            if total_edges_from_node > 0:
                self.TM_departure[i, :] /= total_edges_from_node

    def back_probability(self):
        # 初始化每个节点出发的边的数量的统计为长度为40的零数组
        self.efn_back = np.zeros(num_nodes, dtype=int)

        # 存储每个起点到终点的最短路径的字典
        shortest_paths_back = {}

        for start in self.end_mapping:
            for end in self.start_mapping:
                # 使用NetworkX计算最短路径
                try:
                    path = nx.shortest_path(self.G, source=start, target=end, weight='weight')
                    # 记录路径
                    shortest_paths_back[(start, end)] = path
                except nx.NetworkXNoPath:
                    print(f"No path found from {start} to {end}.")
                except KeyError:
                    print(f"One of the nodes {start} or {end} does not exist.")

        # 根据最短路径记录每条路径所经过的边。
        path_edges = {}
        for (start, end), path in shortest_paths_back.items():
            edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            path_edges[(start, end)] = edges
            # 更新每个节点出发的边的数量
            for edge in edges:
                source = edge[0]
                # 直接使用源节点的索引来更新数组
                self.efn_back[source] += 1

        # 统计每个节点出发的边的次数并更新转移矩阵
        for path in path_edges.values():
            for edge in path:
                source, target = edge
                self.TM_back[source, target] += 1

        # 将计数转换为概率
        for i in range(num_nodes):
            total_edges_from_node = np.sum(self.TM_back[i, :])
            if total_edges_from_node > 0:
                self.TM_back[i, :] /= total_edges_from_node

    # def generate_TM(self): #没有扩展的
    #     # 根据更新后的状态计算转移矩阵
    #     for i in range(num_nodes):
    #         total = (self.departure_car[i] * (self.departure_stop[i] + self.departure_step[i]) +
    #                  (self.efn_departure[i] - self.departure_car[i]) * self.departure_step[i] +
    #                  self.back_car[i] * (self.back_stop[i] + self.back_step[i]) +
    #                  (self.efn_back[i] - self.back_car[i]) * self.back_step[i])
    #         if total > 0:
    #             self.TM[i, i] = (self.departure_car[i] * self.departure_stop[i] + self.back_car[i] * self.back_stop[i]) / total
    #             for j in range(num_nodes):
    #                 if i != j:
    #                     self.TM[i, j] = (self.TM_departure[i, j] * (self.departure_car[i] * self.departure_step[i] +
    #                                                                (self.efn_departure[i] - self.departure_car[i]) *
    #                                                                self.departure_step[i]) +
    #                                      self.TM_back[i, j] * (self.back_car[i] * self.back_step[i] +
    #                                                            (self.efn_back[i] - self.back_car[i]) *
    #                                                            self.back_step[i]))/ total
    #         else:
    #             # 如果总数为0，则节点i将100%地停留在自身位置
    #             self.TM[i, i] = 1
    #             for j in range(num_nodes):
    #                 if i != j:
    #                     self.TM[i, j] = 0
    #
    #     return self.TM

    def generate_TM(self):
        # 初始化扩展转移矩阵
        self.TM = np.zeros((2 * num_nodes, 2 * num_nodes))

        # 遍历所有节点以填充转移矩阵
        for i in range(num_nodes):
            # 计算转移概率
            for j in range(num_nodes):
                if i != j:
                    # 1. Mi 到 Mj
                    self.TM[i + num_nodes, j + num_nodes] = (self.TM_departure[i, j] * (
                                (self.efn_departure[i] - self.departure_car[i]) * self.departure_step[i]) * (
                                                                         1 - self.departure_arrive_pdf[j])) + (
                                                                        self.TM_back[i, j] * (
                                                                            (self.efn_back[i] - self.back_car[i]) *
                                                                            self.back_step[i]) * (
                                                                                    1 - self.back_arrive_pdf[j]))

                    # 2. Pi 到 Mj
                    self.TM[i, j + num_nodes] = self.TM_departure[i, j] * (
                                self.departure_car[i] * self.departure_step[i]) + self.TM_back[i, j] * (
                                                            self.back_car[i] * self.back_step[i])

                    # 3. Mi 到 Pj
                    self.TM[i + num_nodes, j] = (self.TM_departure[i, j] * (
                                (self.efn_departure[i] - self.departure_car[i]) * self.departure_step[i])) * \
                                                self.departure_arrive_pdf[j] + (self.TM_back[i, j] * (
                                (self.efn_back[i] - self.back_car[i]) * self.back_step[i]) * self.back_arrive_pdf[j])

            # 4. Pi 到 Pj (始终为0，因为没有直接的Pi到Pj的转移)

            # 5. Mi 到 Mi (始终为0，因为Mi状态假设车辆总是在移动)

            # 6. Pi 到 Pi
            self.TM[i, i] = self.departure_car[i] * self.departure_stop[i] + self.back_car[i] * self.back_stop[i]

        # 标准化转移矩阵的每一行，确保概率和为1
        for i in range(2 * num_nodes):
            row_sum = np.sum(self.TM[i, :])
            if row_sum > 0:
                self.TM[i, :] /= row_sum
            else:
                # 如果该行总和为0（即该状态没有出去的转移），则保持在原状态
                self.TM[i, i] = 1.0

        return self.TM

    def run_simulation(self):
        self.road_weight()  # 生成两个权重矩阵的G G_offpeak G_peak
        self.normal_distribution()  # 计算出发概率 self.departure_distributions self.back_distributions
        self.calculate_arrival_distributions()  # 到达概率分布self.departure_arrive

        # 以0.05小时（3分钟）为步长循环1到24小时
        for time in np.arange(1, 24.05, 0.05):
            self.time_possibility(time)  # 计算当天的self.departure_step self.back_step
            self.arrive_possibility(time) # self.departure_arrive[i]
            self.update_graph_weights(time)  # 更新self.G
            self.departure_probability()  # self.TM_departure
            self.back_probability()

            TM = self.generate_TM()  # 生成转移矩阵
            self.transition_matrices[time] = TM

            # # 计算并保存稳态分布
            # A = TM - np.eye(num_nodes)
            # A = np.vstack([A.T, np.ones(num_nodes)])
            # b = np.zeros(num_nodes + 1)
            # b[-1] = 1

            # 调整稳态分布的计算方法以适应扩展状态空间
            A = TM - np.eye(2 * num_nodes)
            A = np.vstack([A.T, np.ones(2 * num_nodes)])
            b = np.zeros(2 * num_nodes + 1)
            b[-1] = 1

            try:
                steady_state = np.linalg.lstsq(A, b, rcond=None)[0]
                self.steady_states[time] = steady_state
            except np.linalg.LinAlgError:
                print(f"Cannot compute steady state for time {time} due to numerical issues.")

            # 构造保存转移矩阵的文件名
            output_dir = "TM"
            hour = int(time)
            minute = int((time % 1) * 60)
            filename = f"transition_matrix_{hour:02d}_{minute:02d}.csv"
            filepath = os.path.join(output_dir, filename)  # 将文件名和目标文件夹合并为完整路径

            # 保存转移矩阵到CSV文件
            pd.DataFrame(TM).to_csv(filepath, index=False)

        # 构造保存稳态的文件名
        # 键（时间）成为DataFrame的索引，每个稳态分布数组成为一行
        steady_states_df = pd.DataFrame.from_dict(self.steady_states, orient='index')

        # 给列命名
        steady_states_df.columns = [f"State_{i}" for i in range(steady_states_df.shape[1])]

        # 将DataFrame保存到CSV文件
        steady_states_df.to_csv(path_or_buf="SS.csv", index_label="Time")



        #这里可以添加代码将结果保存到文件或进行其他处理
        print("Simulation complete.")


# def visualize_graph(G, title="Graph Visualization"):#检查权重
#     plt.figure(figsize=(12, 8))  # 设置画布大小
#     pos = nx.spring_layout(G, seed=42)  # 为图G生成布局
#
#     # 绘制网络：节点和边
#     nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', edge_color='k', linewidths=1,
#             font_size=10, font_weight='bold', alpha=0.9)
#
#     # 绘制边的权重
#     edge_labels = nx.get_edge_attributes(G, 'weight')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
#
#     plt.title(title)
#     plt.axis('off')  # 关闭坐标轴
#     plt.show()


start_points = [202, 203, 204, 205, 206, 208, 209, 303, 304, 305, 306, 307, 308, 309, 313, 314, 315, 316, 317,
                318, 401, 402, 403, 404, 405, 406, 407]
end_points = [101, 102, 103, 104, 105, 106]

markov = Markov(G, labels, start_points, end_points)
markov.run_simulation()  # 更新权重并生成高峰期和非高峰期图
# # 可视化高峰期图
# visualize_graph(markov.G_peak, "High Peak Traffic Graph Visualization")
#
# # 可视化非高峰期图
# visualize_graph(markov.G_offpeak, "Off-Peak Traffic Graph Visualization")
