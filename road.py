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
        # 创建一个空的Counter对象来存储边的出现次数
        self.edge_counts = Counter()
        # 初始化转移矩阵为零矩阵

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
        lower, upper = 0, 24  # 定义时间的上下限

        # 初始化出发时间分布字典
        self.departure_distributions = {}

        # 遍历每个起点和终点，计算它们之间的出发时间分布
        for start in self.start_mapping:
            for end in self.end_mapping:
                try:
                    # 使用NetworkX获取起点到终点的最短路径长度（即边的数量）
                    path_length = nx.shortest_path_length(self.G, source=start, target=end, weight=None)

                    # 计算转移次数对应的时间增加，这里假设每次转移增加0.05小时
                    transfer_time = path_length * 0.05

                    # 设置mu基于路径上的边数，每条边通过时间增加0.05小时
                    mu = 9 - transfer_time  # 这里的9是示例中的起始时间，根据实际情况调整

                    sigma = 0.5  # 假定标准差为1，实际应根据数据调整

                    # 转换均值和标准差为截断正态分布的参数
                    a, b = (lower - mu) / sigma, (upper - mu) / sigma
                    # 存储每个起点到终点的出发时间分布
                    self.departure_distributions[(start, end)] = truncnorm(a, b, loc=mu, scale=sigma)

                except nx.NetworkXNoPath:
                    print(f"No path found from {start} to {end}.")
                except KeyError:
                    print(f"One of the nodes {start} or {end} does not exist.")

        # 返程时间分布
        self.back_distributions = {}

        for start_point in self.end_mapping:
            mu = 18
            sigma = 0.5
            a, b = (lower - mu) / sigma, (upper - mu) / sigma
            self.back_distributions[start_point] = truncnorm(a, b, loc=mu, scale=sigma)

        # 出发车队数量
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
        # 初始化存储PDF和CDF值的字典
        self.departure_step = {}
        self.departure_stop = {}
        self.back_step = np.zeros(num_nodes)
        self.back_stop = np.zeros(num_nodes)
        # 对于出发概率
        # 遍历所有可能的起点和终点组合
        for start in self.start_mapping:
            for end in self.end_mapping:
                # 检查是否为这对起点和终点定义了正态分布
                if (start, end) in self.departure_distributions:
                    # 获取起点到终点的正态分布对象
                    norm_dist = self.departure_distributions[(start, end)]
                    # 计算在时间x的PDF值并存储
                    self.departure_step[(start, end)] = norm_dist.pdf(time)
                    # 计算在时间x的CDF值，并计算停留概率存储
                    self.departure_stop[(start, end)] = 1 - norm_dist.cdf(time)
                else:
                    # 如果没有为这对节点定义正态分布，PDF记作0，CDF记作1
                    self.departure_step[(start, end)] = 0
                    self.departure_stop[(start, end)] = 1

        for i in range(num_nodes):  # 遍历40个节点
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
        lower, upper = 0, 24  # 定义时间的上下限
        self.departure_arrive = {}  # 存储从起点到终点的到达时间分布
        self.back_arrive = {}  # 存储从终点返回起点的到达时间分布

        edge_time = 0.05  # 每条边的时间

        # 处理出发到达分布 服从正态分布
        for end_point in self.end_mapping:
            mu = 9
            sigma = 0.5
            a, b = (lower - mu) / sigma, (upper - mu) / sigma
            self.departure_arrive[end_point] = truncnorm(a, b, loc=mu, scale=sigma)

        # 处理返回起点的分布
        # 遍历所有可能的起点和终点组合
        for start in self.end_mapping:
            for end in self.start_mapping:
                try:
                    # 使用NetworkX获取起点到终点的最短路径长度（即边的数量）
                    path_length = nx.shortest_path_length(self.G, source=start, target=end, weight=None)

                    # 计算转移次数对应的时间增加，这里假设每次转移增加0.05小时
                    transfer_time = path_length * 0.05

                    # 设置mu基于路径上的边数，每条边通过时间增加0.05小时
                    mu = 18 + transfer_time
                    sigma = 0.5

                    # 转换均值和标准差为截断正态分布的参数
                    a, b = (lower - mu) / sigma, (upper - mu) / sigma
                    # 存储每个起点到终点的出发时间分布
                    self.back_arrive[(start, end)] = truncnorm(a, b, loc=mu, scale=sigma)

                except nx.NetworkXNoPath:
                    print(f"No path found from {start} to {end}.")
                except KeyError:
                    print(f"One of the nodes {start} or {end} does not exist.")

    def arrive_possibility(self, time):
        # 初始化存储特定时间点PDF值的字典
        self.back_arrive_pdf = {}
        self.departure_arrive_pdf = np.zeros(num_nodes)

        for i in range(num_nodes):  # 遍历40个节点
            # 对于上班到达概率
            if i in self.departure_arrive:
                # 获取节点的正态分布
                norm_dist = self.departure_arrive[i]
                # 计算在时间x的PDF值
                self.departure_arrive_pdf[i] = norm_dist.pdf(time)
            else:
                # 如果没有为节点定义正态分布，出发到达概率记作0
                self.departure_arrive_pdf[i] = 0

        # 对于返回起点的概率
        # 遍历所有可能的起点和终点组合
        for start in self.end_mapping:
            for end in self.start_mapping:
                # 检查是否为这对起点和终点定义了正态分布
                if (start, end) in self.back_arrive:
                    # 获取起点到终点的正态分布对象
                    norm_dist = self.back_arrive[(start, end)]
                    # 计算在时间x的PDF值并存储
                    self.back_arrive_pdf[(start, end)] = norm_dist.pdf(time)
                else:
                    # 如果没有为这对节点定义正态分布，PDF记作0，
                    self.back_arrive_pdf[(start, end)] = 0

    def update_graph_weights(self, time):  # 更新self.G

        # 确定当前是否为高峰期
        is_peak = (7 <= time < 9) or (17 <= time < 19)

        # 选择相应的权重矩阵
        self.G = self.G_peak if is_peak else self.G_offpeak

    def departure_probability(self, end):
        # 存储每个起点到终点的最短路径的字典
        shortest_paths_departure = {}
        # 初始化每个节点出发的边的数量的统计为长度为40的零数组
        self.efn_departure = np.zeros(num_nodes, dtype=int)
        self.TM_departure = np.zeros((num_nodes, num_nodes))

        for start in self.start_mapping:
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

    def back_probability(self, end):
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

        # # 将计数转换为概率
        # for i in range(num_nodes):
        #     total_edges_from_node = np.sum(self.TM_back[i, :])
        #     if total_edges_from_node > 0:
        #         self.TM_back[i, :] /= total_edges_from_node

    def car_flow(self, time):
        self.update_graph_weights(time)  # 更新self.G
        self.normal_distribution()  # 计算出发概率 self.departure_distributions self.back_distributions
        self.calculate_arrival_distributions()  # 到达概率分布self.departure_arrive
        # 计算当天的self.departure_step[(start, end)] self.departure_stop = {} self.back_step = [] self.back_stop = []
        self.time_possibility(time)
        self.arrive_possibility(time)  # self.back_arrive_pdf = {} self.departure_arrive_pdf = []

        # 初始化边的车流量矩阵
        self.edge_flow_matrix = np.zeros((num_nodes, num_nodes))
        # 边到达的车流量矩阵
        self.edge_stop_matrix = np.zeros((num_nodes, num_nodes))

        # 上班的车队
        for end in self.end_mapping:
            self.departure_probability(end)  # 更新self.TM_departure
            # 遍历所有边，更新车流量
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if self.G.has_edge(i, j):
                        # 如果存在从i到j的边，更新这条边的车流量
                        if (i, end) in self.departure_step:
                            self.edge_flow_matrix[i, j] += (self.TM_departure[i, j] * self.departure_step[(i, end)] *
                                                            (1 - self.departure_arrive_pdf[j]))
                            self.edge_stop_matrix[i, j] += (self.TM_departure[i, j] * self.departure_step[(i, end)] *
                                                            self.departure_arrive_pdf[j])

        # 下班的车队
        for end in self.start_mapping:
            self.back_probability()  # 更新self.TM_back
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if self.G.has_edge(i, j):
                        if (i, end) in self.back_arrive_pdf:
                            # 对于每条边，累加返回流量
                            self.edge_flow_matrix[i, j] += (self.TM_back[i, j] * self.back_step[i] *
                                                            (1 - self.back_arrive_pdf[(i, end)]))
                            self.edge_stop_matrix[i, j] += (self.TM_back[i, j] * self.back_step[i] *
                                                            self.back_arrive_pdf[(i, end)])
                        else:
                            self.edge_flow_matrix[i, j] += self.TM_back[i, j] * self.back_step[i]
                            self.edge_stop_matrix[i, j] += 0
        # 初始化静止车流量数组
        self.edge_stop = np.zeros(num_nodes)

        # 遍历所有可能的起点和终点组合
        for start in self.start_mapping:
            for end in self.end_mapping:
                # 检查是否为这对起点和终点定义了停留概率
                if (start, end) in self.departure_stop:
                    # 如果存在定义的停留概率，累加到对应节点的静止车流量
                    self.edge_stop[start] += self.departure_stop[(start, end)]

    def generate_TM(self, time):
        # 初始化扩展转移矩阵
        self.TM = np.zeros((2 * num_nodes, 2 * num_nodes))
        self.car_flow(time) #计算车流量

        # 计算Pi到Mj的车队数和Mi到Pj的车流量
        for i in range(num_nodes):
            for j in range(num_nodes):
                # 计算从Pi到Mj的转移概率
                if self.efn_departure[i] > 0:  # 确保分母不为0
                    self.TM[i, num_nodes + j] = self.edge_flow_matrix[i, j] * (
                                self.departure_car[i] / self.efn_departure[i])
                else:
                    self.TM[i, num_nodes + j] = 0

                # 计算从Mi到Pj的转移概率
                self.TM[num_nodes + i, j] = self.edge_stop_matrix[i, j]

        # 处理Pi到Pi的转移概率和Mi到Mi的转移概率
        for i in range(num_nodes):
            # Pi到Pi的转移概率等于节点i上停泊车辆的比例
            self.TM[i, i] = self.edge_stop[i]

            # Mi到Mi的转移概率设置为0，因为一旦车辆开始移动，就假设它不会停留在原地
            self.TM[num_nodes + i, num_nodes + i] = 0

        return self.TM

    def run_simulation(self, initial_state):
        # 初始化状态向量记录
        state_vectors = {}

        # 当前状态向量
        current_state = np.array(initial_state)

        # 以0.05小时（3分钟）为步长循环1到24小时
        for time in np.arange(1, 24.05, 0.05):
            TM = self.generate_TM(time)  # 生成转移矩阵
            self.transition_matrices[time] = TM

            # 更新当前状态向量
            current_state = np.dot(TM, current_state)
            state_vectors[time] = current_state

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

        print("Simulation complete.")
        # 在函数结束时，返回状态向量记录
        return state_vectors


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

# 初始化所有车辆都停泊在起点，没有车辆在移动状态
initial_state = np.zeros(2 * num_nodes)
starts = [node_mapping[point] for point in start_points]
for start in starts:
    initial_state[start] = 200  # 假设每个起点最初有一辆车停泊

markov = Markov(G, labels, start_points, end_points)
markov.run_simulation(initial_state)

# 更新权重并生成高峰期和非高峰期图
# # 可视化高峰期图
# visualize_graph(markov.G_peak, "High Peak Traffic Graph Visualization")
#
# # 可视化非高峰期图
# visualize_graph(markov.G_offpeak, "Off-Peak Traffic Graph Visualization")
