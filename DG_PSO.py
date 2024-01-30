import numpy as np
from docplex.cp.modeler import value
from mainDOC import MGO

# 开始种群等基本定义
N = 3 # 初始种群个数
d = 48 # 空间维数
ger = 50 # 最大迭代次数
socmin = 0.2
socmax = 0.8
Price_out_lim = 100 * np.array([
    # 电网分时电价
    [0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.82, 0.82, 0.82, 1.35, 1.35, 1.35, 1.35, 1.35, 0.82, 0.82, 0.82, 1.35, 1.35, 1.35, 1.35, 1.35, 0.38, 0.38],
    [0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36, 0.36]
])
Price_in_lim = 100 * np.array([
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    [0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35]
])

# pric_co2 = 100 * np.array([
#     [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
#     [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
# ])碳价格
# 位置限制
plimit = np.hstack((Price_out_lim, Price_in_lim))
# 速度限制
vlimit = np.array([1 + np.zeros(48), 1 + np.zeros(48)])  # 速度限制设为 -1 到 1

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
    price_out = x[0:24] / 100
    price_in = x[24:48] / 100

    # 调用MGO函数，这需要具体实现
    Fdown, Pgrid_out, Pgrid_in = MGO(price_out, price_in)# Fdown是成本

    # 计算成本
    Cost_E = np.sum(Pgrid_out * price_out) + np.sum(Pgrid_in * price_in) # 收入：MGO买电的量乘以电价

    # 计算总目标函数
    F = - Cost_E + Fdown #成本

    return F

# 初始化每个个体的当前适应度
fx = np.zeros(N)
# 迭代更新开始
iter = 1
record = np.zeros(ger)

while iter <= ger:
    print("上层迭代", iter)  # 打印当前迭代次数

    for n in range(N):
        fx[n] = obj_all(x[n, :], iter)  # 计算每个个体的适应度

    for i in range(N):
        if fxm[i] > fx[i]:
            fxm[i] = fx[i]
            xm[i, :] = x[i, :]

    if fym > min(fxm):
        fym, nmax = min((val, idx) for (idx, val) in enumerate(fxm))
        ym = xm[nmax, :]

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
import matplotlib.pyplot as plt

plt.figure()
plt.plot(record)
plt.title('Convergence process')
plt.show()

price_E = ym[0:24] / 100
price_C = ym[24:48]
MGO(price_E, price_C)

# 打印最终结果
print(f'min：{fym}')
print(f'Minimum point position：{ym}')

# 计时结束
end_time = time.time()
print(f'run time：{end_time - start_time} s')


