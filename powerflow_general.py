import numpy as np
import pandas as pd
from gridinfo import nodedata_dict
"""
Branch_data阻抗矩阵
Nodedata负荷矩阵, 
zhilu支路开闭 1和0 长度和num_branches一致
num_nodes母线
num_branches支路/多少边
UB = 12.66 * 1.5  # 电压基准 kV
SB = 10  # 功率基准 MVA
Pev 电动汽车负荷
"""

# 全局变量
# 支路矩阵
# Branch_data=[文件应包含以下列：'From bus', 'To bus', 'Resistance (Ω)', 'Reactance (Ω)']
branch_data_df = pd.read_csv('grid/Branch_data.csv')
branch_data = branch_data_df[['From bus', 'To bus', 'Resistance', 'Reactance']].to_numpy()

num_nodes = 40    # 节点数量
num_branches = 49 # 支路数量
Pev_example = 10          # 额外负荷
S = [33, 34, 35, 36, 37]  #pcc
zhilu = np.ones(37)
for k in range(37):
    if k + 1 in S:
        zhilu[k] = 0
UB = 11  # 电压基准 kV
SB = 10  # 功率基准 MVA

def powerflow(Pev):
    for i in 24:
        Pev_i = Pev[i]
        powerflow(i,Pev_i)



def powerflow(i, Pev_i):
    format_long = np.set_printoptions(precision=15)  # 设置高精度打印
    # 动态初始化数组和变量
    Y = np.zeros((num_nodes, num_nodes), dtype=complex)
    Bus = np.zeros((num_nodes, 3))
    # 初始化电压相角 delt 和电压幅值 u
    delt = np.zeros(num_nodes)
    u = np.ones(num_nodes)
    # 第i小时的数据
    Nodedata = nodedata_dict[i]
    Branch_data = branch_data

    # 更新节点基础负荷功率
    Nodedata[:, 1] += Pev

    # 将阻抗矩阵变成阻抗标幺化后矩阵 A
    ZB = UB ** 2 / SB  # 阻抗基准 ohm
    A = np.zeros((num_branches, 4))
    A[:, 2:4] = Branch_data[:, 2:4] / ZB  # % 阻抗标幺化，将阻抗标幺化的值 Branch_data(:,[3,4]) / ZB 赋值给A(:,[3,4])
    A[:, 0:2] = Branch_data[:, 0:2]  # 阻抗标幺化后矩阵，将支路起始、末节点编号 Branch_data(:,[1,2]) 赋给 A(:,[1,2])

    # 节点注入功率矩阵 Bus
    Bus[:, 1:3] = Nodedata[:, 1:3] / SB / 1000  # 功率标幺化
    p = -Bus[:, 1]  # 注入功率为负
    q = -Bus[:, 2]

    # 节点导纳矩阵 Y 根据《电力系统分析》第6页提供的方法形成节点导纳矩阵，共32个节点+1个大地节点=33个节点
    for k in range(num_branches):
        if zhilu[k] == 1:
            m = int(A[k, 0])
            n = int(A[k, 1])
            Y[m, m] += 1 / (A[k, 2] + 1j * A[k, 3])
            Y[n, n] += 1 / (A[k, 2] + 1j * A[k, 3])
            Y[m, n] -= 1 / (A[k, 2] + 1j * A[k, 3])
            Y[n, m] -= 1 / (A[k, 2] + 1j * A[k, 3])

    # 生成电导 G、电纳 B 矩阵
    G = np.real(Y)
    B = np.imag(Y)


    # 初始化计数器 k，计算精度 precision
    k = 0
    precision = 1
    pp = np.zeros(num_nodes - 1)
    qq = np.zeros(num_nodes - 1)

    # 构建节点导纳矩阵 Y...

    # 功率标幺化
    Bus[:, 1:3] = Nodedata[:, 1:3] / SB / 1000
    p = -Bus[:, 1]
    q = -Bus[:, 2]

    # 主程序
    while precision > 0.0001 and k < 10:
        # 生成 delta P

        for m in range(1, num_nodes):
            pt = u[m] * u * (G[m, :] * np.cos(delt[m] - delt) + B[m, :] * np.sin(delt[m] - delt))
            pp[m - 1] = p[m] - np.sum(pt)

        # 生成 delta Q

        for m in range(1, num_nodes):
            qt = u[m] * u * (G[m, :] * np.sin(delt[m] - delt) - B[m, :] * np.cos(delt[m] - delt))
            qq[m - 1] = q[m] - np.sum(qt)

        PQ = np.zeros((num_nodes - 1) * 2)
        PQ[0:(num_nodes - 1)] = pp
        PQ[(num_nodes - 1):(num_nodes * 2)] = qq

        # 生成 H, M, N, L 子矩阵
        H = np.zeros(((num_nodes - 1), (num_nodes - 1)))
        M = np.zeros(((num_nodes - 1), (num_nodes - 1)))
        N = np.zeros(((num_nodes - 1), (num_nodes - 1)))
        L = np.zeros(((num_nodes - 1), (num_nodes - 1)))

        for m in range(1, num_nodes):
            for n in range(1, num_nodes):
                if m != n:
                    H[m - 1, n - 1] = -u[m] * u[n] * (
                            G[m, n] * np.sin(delt[m] - delt[n]) - B[m, n] * np.cos(delt[m] - delt[n]))
                    M[m - 1, n - 1] = u[m] * u[n] * (
                            G[m, n] * np.cos(delt[m] - delt[n]) + B[m, n] * np.sin(delt[m] - delt[n]))
                    N[m - 1, n - 1] = -M[m - 1, n - 1]
                    L[m - 1, n - 1] = H[m - 1, n - 1]
                else:
                    H[m - 1, m - 1] = np.sum(
                        u[m] * u * (G[m, :] * np.sin(delt[m] - delt) - B[m, :] * np.cos(delt[m] - delt))) + u[m] ** 2 * \
                                      B[m, m]
                    M[m - 1, m - 1] = u[m] ** 2 * G[m, m] - np.sum(
                        u[m] * u * (G[m, :] * np.cos(delt[m] - delt) + B[m, :] * np.sin(delt[m] - delt)))
                    N[m - 1, m - 1] = -M[m - 1, m - 1]
                    L[m - 1, m - 1] = H[m - 1, m - 1]

        # 生成雅各比矩阵 J
        J = np.zeros(((num_nodes - 1) * 2, (num_nodes - 1) * 2))
        J[0:(num_nodes - 1), 0:(num_nodes - 1)] = H
        J[0:(num_nodes - 1), (num_nodes - 1):(num_nodes * 2)] = N
        J[(num_nodes - 1):(num_nodes * 2), (num_nodes - 1):(num_nodes * 2)] = L
        J[(num_nodes - 1):(num_nodes * 2), 0:(num_nodes - 1)] = M

        # 计算 delta theta, delta u
        # uu = -np.linalg.inv(J) @ PQ
        uu = np.linalg.solve(J, -PQ)
        precision = np.max(np.abs(uu))

        # 修改各节点 delt，u
        delt[1:num_nodes] += uu[0:(num_nodes - 1)]
        u[1:num_nodes] += uu[(num_nodes - 1):(num_nodes * 2)]

        # 修改计数器
        k += 1

    p0 = u[0] * u * (G[0, :] * np.cos(delt[0] - delt) + B[0, :] * np.sin(delt[0] - delt))
    q0 = u[0] * u * (G[0, :] * np.sin(delt[0] - delt) - B[0, :] * np.cos(delt[0] - delt))

    p[0] = np.sum(p0)
    q[0] = np.sum(q0)

    # 转换电压相角为度
    delt = delt * 360 / (2 * np.pi)

    # 输出结果
    uresult = u
    deltresult = delt
    node = np.arange(num_nodes)

    return uresult, deltresult, node













Pev = 10
#uresult, deltresult, node = powerflow(Branch_data, Nodedata, zhilu, num_nodes, num_branches, UB, SB, Pev)
print(uresult, deltresult, node)
