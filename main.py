from Microgrid import Microgrid, OptimizationMicrogrid
import cplex
from cplex.exceptions import CplexError

#整体优化============
class TotalOptimizationManager:
    def __init__(self, microgrids, C_buy, C_sell):
        self.microgrids = microgrids
        self.C_buy = C_buy
        self.C_sell = C_sell
        self.problem = cplex.Cplex()

    def setup(self):
        for grid in self.microgrids:
            optimization = OptimizationMicrogrid(grid, len(self.microgrids), self.problem, self.C_buy, self.C_sell)

            # 将总体问题实例传递给每个方法
            optimization.add_variable()
            optimization.add_constraints()
            optimization.add_objective()

    def solve(self):
        self.problem.solve()

#==================
#全局
C_buy = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.53, 0.53, 0.53, 0.82, 0.82,
        0.82, 0.82, 0.82, 0.53, 0.53, 0.53, 0.82, 0.82, 0.82, 0.53, 0.53, 0.53]

C_sell = [0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.42, 0.42, 0.42, 0.65, 0.65,
         0.65, 0.65, 0.65, 0.42, 0.42, 0.42, 0.65, 0.65, 0.65, 0.42, 0.42, 0.42]

#======grid1========#
id1 = 1
load_1 = [88.24, 83.01, 80.15, 79.01, 76.07, 78.39, 89.95, 128.85, 155.45, 176.35, 193.71, 182.57, 179.64, 166.31, 164.61, 164.61, 174.48, 203.93, 218.99, 238.11, 216.14, 173.87, 131.07, 94.04]

P_wt_1 = [66.9, 68.2, 71.9, 72, 78.8, 94.8, 114.3, 145.1, 155.5, 142.1, 115.9, 127.1, 141.8, 145.6, 145.3, 150, 206.9, 225.5, 236.1, 210.8, 198.6, 177.9, 147.2, 58.7]

P_pv_1 = [0, 0, 0, 0, 0.06, 6.54, 20.19, 39.61, 49.64, 88.62, 101.59, 66.78, 110.46, 67.41, 31.53, 50.76, 20.6, 22.08, 2.07, 0, 0, 0, 0, 0]



ebattery_1 = 300
soc0_1 = 0.5
socmin_1 = 0.3
socmax_1 = 0.95
pcs_1 = 40
POWER_1 = 200  # 主网功率交换限制
POWER_MIC_1 = 160
#======grid2========#
id2 = 2
P_pv_2 = [43.6351830380341, 5.40581541514531, 32.8358719758836, 19.0009117783515,
        39.1178845139782, 64.4853270641430, 30.2278361336943, 41.4382515605911,
        7.56783114759613, 72.7276154279869, 16.6104475197874, 30.5651618832424,
        52.8224212142948, 60.6698641822058, 13.8455529913677, 41.3903809205061,
        79.6270489842773, 56.6087525272567, 6.44538407391447, 3.46464771000820,
        39.2924450387408, 35.7277152135362, 38.9438852676276, 13.2712897668273]
# 将某些小时的PV能力设置为0
for hour in [0, 1, 2, 3, 4, 5, 19, 20, 21, 22, 23]:
    P_pv_2[hour] = 0

load_2 = [2 * x for x in [88.24, 83.01, 80.15, 79.01, 76.07, 78.39, 89.95, 248.85,
                        55.45, 76.35, 93.71, 82.57, 79.64, 66.31, 164.61, 164.61,
                        174.48, 203.93, 218.99, 238.11, 216.14, 173.87, 131.07, 94.04]]

ebattery_2 = 300  # 电池容量
soc0_2 = 0.5  # 初始SOC
socmin_2 = 0.3  # 最小SOC
socmax_2 = 0.95  # 最大SOC
pcs_2 = 40  # 充/放电功率限制
C_de_2 = 0.7939  # 燃气轮机成本系数
POWER_2 = 300  # 主网功率交换限制
POWER_MIC_2 = 200

# 创建 Microgrid 实例
num_microgrid = 2
grid1 = Microgrid(id1, load_1, POWER_1, POWER_MIC_1, ebattery_1, socmin_1, socmax_1, soc0_1, pcs_1, P_pv=P_pv_1, P_wt=P_wt_1, C_mt=None, C_de=None)
grid2 = Microgrid(id2, load_2, POWER_2, POWER_MIC_2, ebattery_2, socmin_2, socmax_2, soc0_2, pcs_2, P_pv=P_pv_2, P_wt=None, C_mt=None, C_de=C_de_2)

# # 创建 Optimization 实例并求解
# optimization1 = OptimizationMicrogrid(grid1, num_microgrid, C_buy, C_sell)
# optimization2 = OptimizationMicrogrid(grid1, num_microgrid, C_buy, C_sell)

# 创建 TotalOptimizationManager 实例
total_optimization_manager = TotalOptimizationManager([grid1, grid2], C_buy, C_sell)

# 设置优化问题（整合所有微电网的变量、约束和目标函数）
total_optimization_manager.setup()

# 求解整合的优化问题
total_optimization_manager.solve()

