
from mainDOC import MGO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 开始种群等基本定义
EV_penetration = 2000 #辆/节点
N = 3 # 初始种群个数
d = 144 # 空间维数
ger = 50 # 最大迭代次数

# 电网分时电价 买卖电价都小于主电网
MG_buy = 100 * np.array([
    [0.2205, 0.2205, 0.2205, 0.2205, 0.2205, 0.2205, 0.2205, 0.2205,
     0.5802, 0.5802, 0.9863, 0.9863, 0.5802, 0.5802, 0.9863, 0.9863,
     0.9863, 0.9863, 0.9863, 0.5802, 0.5802, 0.5802, 0.5802, 0.5802],
    [0.2] * 24
])
MG_sell = 100 * np.array([
    [0.453, 0.453, 0.453, 0.453, 0.453, 0.453, 0.453, 0.453, 0.453,
     0.453, 0.453, 0.453, 0.453, 0.453, 0.453, 0.453, 0.453, 0.453,
     0.453, 0.453, 0.453, 0.453, 0.453, 0.453],
    [0.2] * 24
])
# 电网分时电价 买卖电价都小于主电网
EV1 = 100 * np.array([
    [2] * 24,
    [0.2] * 24
])
EV2 = 100 * np.array([
    [2] * 24,
    [0.2] * 24
])
EV3 = 100 * np.array([
    [2] * 24,
    [0.2] * 24
])
EV4 = 100 * np.array([
    [2] * 24,
    [0.2] * 24
])

# 位置限制
plimit = np.hstack((MG_buy, MG_sell, EV1, EV2, EV3, EV4))
# 速度限制
vlimit = np.array([1 + np.zeros(144), 1 + np.zeros(144)])  # 速度限制设为 -1 到 1

w = 0.8                          # 惯性权重
c1 = 0.5                         # 自我学习因子
c2 = 0.5                         # 群体学习因子

# 计时开始
import time
start_time = time.time()

# 初始化种群的位置和速度
x = np.zeros((N, d))
for i in range(d):
    x[:, i] = plimit[0, i] + (plimit[1, i] - plimit[0, i]) * np.random.rand(N)

v = np.random.rand(N, d)
xm = np.copy(x)                       # 每个个体的历史最佳位置
ym = np.zeros(d)                      # 种群的历史最佳位置
fxm = np.zeros(N) + 125000            # 每个个体的历史最佳适应度
fym = float('inf')                    # 种群历史最佳适应度

def obj_all(x, iter):
    """
    目标函数，用于计算优化的目标值。

    :param x: 优化变量数组
    :return: 目标函数的计算结果
    """
    print("下层迭代", iter)  # 打印当前迭代次数
    # 分割x以获得电价
    price_buy = x[0:24] / 100
    price_sell = x[24:48] / 100
    EV_1 = x[48:72] / 100
    EV_2 = x[72:96] / 100
    EV_3 = x[96:120] / 100
    EV_4 = x[120:144] / 100

    #调用EVload
    EVload = EVload(EV_1, EV_2, EV_3, EV_4, EV_penetration)

    # 调用MGO函数
    F1, Pnet_mic, PV, WT= MGO(price_buy, price_sell, EVload) # Fdown是成本

    #调用潮流
    F2 = flow(EVload,Pnet_mic,PV,WT)

    return F1, F2


def dominates(a, b):
    """
    判断解a是否支配解b。

    :param a: 解a的目标函数值，格式为[F1值, F2值]。
    :param b: 解b的目标函数值，格式为[F1值, F2值]。
    :return: 如果a支配b，则返回True；否则返回False。
    """
    better_in_one = False
    for i in range(len(a)):
        if a[i] > b[i]:  # 如果a在任一目标上比b差，则a不支配b
            return False
        elif a[i] < b[i]:  # 如果a在任一目标上比b好，则记录a至少在一个目标上比b好
            better_in_one = True
    return better_in_one  # 如果a至少在一个目标上比b好且没有在任何目标上比b差，则a支配b


# 初始化每个个体的当前适应度
fx = np.zeros(N)

# 迭代更新开始
iter = 1
record = np.zeros(ger)

# 假设已经有了初始化粒子群的代码

# 初始化帕累托前沿列表
pareto_front = []

# 迭代更新开始
for iter in range(ger):
    print("Iteration:", iter)
    # 用于本次迭代中更新帕累托前沿的临时列表
    new_pareto_front = []

    for n in range(N):
        # 计算每个个体的两个目标函数值
        F1, F2 = obj_all(x[n, :], iter)

        # 检查新解是否支配帕累托前沿中的解或是否被支配
        is_dominated = False
        is_dominating = False
        for pf in pareto_front:
            if dominates(x[n, :], pf):
                is_dominating = True
                pareto_front.remove(pf)
            elif dominates(pf, x[n, :]):
                is_dominated = True
                break
        if not is_dominated:
            new_pareto_front.append(x[n, :])

    # 更新帕累托前沿
    pareto_front.extend(new_pareto_front)

    # 速度更新
    v = w * v + c1 * np.random.rand() * (xm - x) + c2 * np.random.rand() * (np.tile(ym, (N, 1)) - x)

    # 边界速度处理
    v = np.clip(v, -vlimit[0, :], vlimit[0, :])

    # 位置更新
    x = x + v

    # 边界位置处理
    for i in range(d):
        x[:, i] = np.clip(x[:, i], plimit[0, i], plimit[1, i])

    record[iter - 1] = fym
    iter += 1

# 结果展示


# 将帕累托解集输出到 CSV 文件
df = pd.DataFrame(pareto_front, columns=['Objective 1', 'Objective 2'])
csv_file_path = '/data/pareto_front.csv'
df.to_csv(csv_file_path, index=False)

# 画出帕累托解集的图
plt.figure(figsize=(8, 6))
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color='blue', label='Pareto Front')
plt.title('Pareto Front')
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.legend()
plt.grid(True)
plt.show()



