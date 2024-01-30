import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import norm

# 初始化参数
alpha = 1.3  # 阻抗影响因子
beta = 1.2
t0 = 10  # 零流量行程时间
c = 30  # 信号周期
lamda = 0.7  # 绿信比
q = 0.8  # 路段车辆到达率

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
    LJ[edge[0] - 1, edge[1] - 1] = 1
    LJ[edge[1] - 1, edge[0] - 1] = 1

# 设置固定的随机种子
fixed_seed = 31
np.random.seed(fixed_seed)

# 创建无向图并添加边
G = nx.Graph()
G.add_edges_from(edges)

# 为每个节点生成一个标签字典
labels = {node_mapping[node]: str(node) for node in nodes}


# 使用spring_layout算法，这通常会产生一个网状的布局
pos = nx.spring_layout(G, iterations=500)  # 增加迭代次数以优化布局

# 计算阻抗模型权重W
S = 2 * np.random.rand(num_nodes, num_nodes)  # 随机生成S权重
Rv = np.zeros_like(S) #路段阻抗模型，通过饱和度S=Q/C，（Q为路段交通流量，C为通行能力，这里是随机生成S所以不用管Q与C的问题）
Cv = np.zeros_like(S) #节点阻抗模型，通过信号周期c，绿信比lamda，路段车辆到达率q，来计算
W = np.zeros_like(S)

for i in range(num_nodes):
    for j in range(num_nodes):
        if LJ[i, j] == 1:
            if S[i, j] <= 1:
                Rv[i, j] = t0 * (1 + alpha * S[i, j] ** beta)
            else:
                Rv[i, j] = t0 * (1 + alpha * (2 - S[i, j]) ** beta)

            if S[i, j] <= 0.6:
                Cv[i, j] = 0.9 * (
                            c * (1 - lamda) ** 2 / 2 / (1 - lamda * S[i, j]) + S[i, j] ** 2 / 2 / q / (1 - S[i, j]))
            else:
                Cv[i, j] = c * (1 - lamda) ** 2 / 2 / (2 - lamda * S[i, j]) + 1.5 * (S[i, j] - 0.6) * S[i, j] / (
                            2 - S[i, j])

            W[i, j] = Rv[i, j] + Cv[i, j]
        else:
            W[i, j] = 0

W[W > 50] = 50  # 将权重限制在一定范围内

# 给图的边赋予权重
for i in range(num_nodes):
    for j in range(num_nodes):
        if G.has_edge(i, j):
            G[i][j]['weight'] = W[i, j]

# 绘制图形，设置背景透明
plt.figure(figsize=(12, 12), facecolor='none')  # facecolor='none' 可能在某些情况下不起作用
nx.draw(G, pos, with_labels=False, node_size=700, node_color='lightblue', font_size=10, font_weight='bold')
nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')

# 添加标题
plt.title('Network Layout')

# 显示图形
plt.show()

# def Markov():
#     #最短路径
#     # 可能的出发点和终点
#     start_points = [202, 203, 204, 205, 206, 208, 209, 303, 304, 305, 306, 307, 308, 309, 313, 314, 315, 316, 317, 318,
#                     401, 402, 403, 404, 405, 406, 407]
#     end_points = [102, 103, 104, 105, 106]
#     path_lengths = {}
#     for start in start_points:
#         for end in end_points:
#             start_index = node_mapping[start]
#             end_index = node_mapping[end]
#             # 检查是否存在路径
#             if nx.has_path(G, start_index, end_index):
#                 path_length = nx.shortest_path_length(G, source=start_index, target=end_index)
#                 path_lengths[(start, end)] = path_length
#             else:
#                 path_lengths[(start, end)] = "No Path"

        # normal_distributions = {}
        #
        # for start_point in start_points:
        #     # 对于每个起点，找到所有终点的平均转移次数
        #     transfer_counts = [path_lengths[(start_point, end)] for end in end_points if (start_point, end) in path_lengths]
        #     if transfer_counts:
        #         average_transfer_count = np.mean(transfer_counts)
        #         mu = 9 - average_transfer_count * 0.05
        #         sigma = 1
        #         # 创建正态分布出发时间
        #         normal_distributions[start_point] = norm(loc=mu, scale=sigma)
        # print(normal_distributions)
        #
        #

