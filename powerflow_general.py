import numpy as np
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

def powerflow(Branch_data, Nodedata, zhilu, num_nodes, num_branches, UB, SB, Pev):
    format_long = np.set_printoptions(precision=15)  # 设置高精度打印
    # 动态初始化数组和变量
    Y = np.zeros((num_nodes, num_nodes), dtype=complex)
    Bus = np.zeros((num_nodes, 3))
    # 初始化电压相角 delt 和电压幅值 u
    delt = np.zeros(num_nodes)
    u = np.ones(num_nodes)

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












# 示例调用
# Branch_data=[Branch_data(:,1)代表支路起始节点1编号，Branch_data(:,2)代表支路末节点2编号，Branch_data(:,[3,4])代表阻抗  ]
Branch_data = np.array([
    [0, 1, 0.0922, 0.0470],
    [1, 2, 0.4930, 0.2511],
    [2, 3, 0.3660, 0.1864],
    [3, 4, 0.3811, 0.1941],
    [4, 5, 0.8190, 0.7070],
    [5, 6, 0.1872, 0.6188],
    [6, 7, 0.7114, 0.2351],
    [7, 8, 1.0300, 0.7400],
    [8, 9, 1.0440, 0.7400],
    [9, 10, 0.1966, 0.0650],
    [10, 11, 0.3744, 0.1238],
    [11, 12, 1.4680, 1.1550],
    [12, 13, 0.5416, 0.7129],
    [13, 14, 0.5910, 0.5260],
    [14, 15, 0.7463, 0.5450],
    [15, 16, 1.2890, 1.7210],
    [16, 17, 0.7320, 0.5740],
    [1, 18, 0.1640, 0.1565],
    [18, 19, 1.5042, 1.3554],
    [19, 20, 0.4095, 0.4784],
    [20, 21, 0.7089, 0.9373],
    [2, 22, 0.4512, 0.3083],
    [22, 23, 0.8980, 0.7091],
    [23, 24, 0.8960, 0.7011],
    [5, 25, 0.2030, 0.1034],
    [25, 26, 0.2842, 0.1447],
    [26, 27, 1.0590, 0.9337],
    [27, 28, 0.8042, 0.7006],
    [28, 29, 0.5075, 0.2585],
    [29, 30, 0.9744, 0.9630],
    [30, 31, 0.3105, 0.3619],
    [31, 32, 0.3410, 0.5302],
    [7, 20, 2, 2],
    [8, 14, 2, 2],
    [11, 21, 2, 2],
    [17, 32, 0.5, 0.5],
    [24, 28, 0.5, 0.5]
])

# 节点负荷矩阵，共32个节点+1个大地节点（零参考节点）=33个节点
# %Nodedata=[Nodedata(:,1)代表节点编号，Nodedata(:,[2,3])代表节点基础负荷功率 ]
Nodedata = np.array([
    [0, 0, 0],
    [1, 100.00, 60.00],
    [2, 90.00, 40.00],
    [3, 120.00, 80.00],
    [4, 60.00, 30.00],
    [5, 60.00, 20.00],
    [6, 200.00, 100.00],
    [7, 200.00, 100.00],
    [8, 60.00, 20.00],
    [9, 60.00, 20.00],
    [10, 45.00, 30.00],
    [11, 60.00, 35.00],
    [12, 60.00, 35.00],
    [13, 120.00, 80.00],
    [14, 60.00, 10.00],
    [15, 60.00, 20.00],
    [16, 60.00, 20.00],
    [17, 90.00, 40.00],
    [18, 90.00, 40.00],
    [19, 90.00, 40.00],
    [20, 90.00, 40.00],
    [21, 90.00, 40.00],
    [22, 90.00, 50.00],
    [23, 420.00, 200.00],
    [24, 420.00, 200.00],
    [25, 60.00, 25.00],
    [26, 60.00, 25.00],
    [27, 60.00, 20.00],
    [28, 120.00, 70.00],
    [29, 200.00, 600.00],
    [30, 150.00, 70.00],
    [31, 210.00, 100.00],
    [32, 60.00, 40.00]
])
num_nodes = 33    # 节点数量
num_branches = 37 # 支路数量
Pev_example = 10          # 额外负荷
S = [33, 34, 35, 36, 37]  #pcc
zhilu = np.ones(37)
for k in range(37):
    if k + 1 in S:
        zhilu[k] = 0
UB = 12.66 * 1.5  # 电压基准 kV
SB = 10  # 功率基准 MVA
Pev = 10
uresult, deltresult, node = powerflow(Branch_data, Nodedata, zhilu, num_nodes, num_branches, UB, SB, Pev)
print(uresult, deltresult, node)
