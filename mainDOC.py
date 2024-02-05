from MicrogridDOC import Microgrid, OptimizationMicrogrid
from docplex.mp.model import Model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#整体优化============
class TotalOptimizationManager:
    def __init__(self, microgrids, num_microgrid, C_buy, C_sell):
        self.microgrids = microgrids
        self.num_microgrid = num_microgrid
        self.C_buy = C_buy
        self.C_sell = C_sell
        self.model = Model(name="Microgrid Optimization Problem")

    def setup(self):
        objective_all = 0
        for i in range(len(self.microgrids)):
            optimization = OptimizationMicrogrid(self.model, self.microgrids[i], self.num_microgrid, self.C_buy, self.C_sell)
            optimization.add_variable()
            optimization.add_constraints()
            objective_all += optimization.add_objective()  # 累加目标函数表达式

        self.model.minimize(objective_all)  # 最小化累加后的目标函数

        # 微电网群交易约束
        tolerance = 0.00001  # 设置一个小的容差值
        for a in range(self.num_microgrid):
            for k in range(24):
                for b in range(self.num_microgrid):
                    if b != a:  # 确保微电网不是自身
                        var_name_net1 = f"Pnetmic_{a}_{b}_{k}"  # a到b
                        var_name_net2 = f"Pnetmic_{b}_{a}_{k}"  # b到a
                        var_name_buy = f"Pbuymic_{a}_{b}_{k}"  # 微电网a从微电网b购买的功率变量
                        var_name_sell = f"Psellmic_{b}_{a}_{k}"  # 微电网b向微电网a出售的功率变量

                        # 添加具有容差的约束
                        net1_var = self.model.get_var_by_name(var_name_net1)
                        net2_var = self.model.get_var_by_name(var_name_net2)
                        self.model.add_constraint(net1_var + net2_var <= tolerance)
                        self.model.add_constraint(net1_var + net2_var >= -tolerance)

                        buy_var = self.model.get_var_by_name(var_name_buy)
                        sell_var = self.model.get_var_by_name(var_name_sell)
                        self.model.add_constraint(buy_var + sell_var <= tolerance)
                        self.model.add_constraint(buy_var + sell_var >= -tolerance)

    def solve(self):
        # 求解问题
        solution = self.model.solve(agent='local')




    def print_optimization_results(self, solution):

        self.model.export_as_lp('model')

        solve_details = self.model.solve_details

        # 打开一个文本文件用于写入
        with open('solution_output.txt', 'w') as file:
            # 检查模型是否有解
            if solution:
                file.write("Solution found!\n")
                # 对于模型中的每个变量，写入其解决方案值
                for v in self.model.iter_variables():
                    file.write(f"{v.name}: {v.solution_value}\n")
            else:
                file.write("No solution found.\n")

        if solve_details.status == 'optimal':
            print("Optimal solution found.")
            print("Objective value:", self.model.objective_value)

            # 遍历所有变量并打印它们的最优值
            for var in self.model.iter_variables():
                print(f"{var.name} = {var.solution_value}")
        else:
            print("Optimal solution not found. Status:", solve_details.status)
            print("Solve time:", solve_details.time, "seconds")

            # 如果未找到最优解，则检查未满足的约束
            if solve_details.status in ['infeasible', 'integer infeasible']:
                print("Checking for unsatisfied constraints...")
                unsatisfied_constraints = self.model.find_unsatisfied_constraints()
                for constraint in unsatisfied_constraints:
                    print(constraint)


    def calculate_grid_power_flows(self):
        Pgrid_out = np.zeros(24)
        Pgrid_in = np.zeros(24)

        for microgrid in self.microgrids:
            for k in range(24):
                Pgrid_out[k] += self.model.get_var_by_name(f'Pbuy_{microgrid.id}_{k}').solution_value
                Pgrid_in[k] += self.model.get_var_by_name(
                    f'Psell_{microgrid.id}_{k}').solution_value  # Psell是负值

        return Pgrid_out, Pgrid_in

    def calculate_objective(self, C_re):
        # 提取解决方案
        extracted_values = {}
        for var in self.model.iter_variables():
            extracted_values[var.name] = var.solution_value

        # 计算目标函数值
        Fdown = 0

        # 遍历每个微电网
        for grid in self.microgrids:
            grid_id = grid.id

            # 添加燃气轮机和柴油轮机成本
            for k in range(24):
                if grid.C_de is not None:
                    Fdown += extracted_values.get(f'Pde_{grid_id}_{k}', 0) * grid.C_de
                if grid.C_mt is not None:
                    Fdown += extracted_values.get(f'Pmt_{grid_id}_{k}', 0) * grid.C_mt

            # 添加储能成本
            for k in range(24):
                Fdown += extracted_values.get(f'Pbat_{grid_id}_{k}', 0) * 0.339

            # 添加外电网购售成本
            for k in range(24):
                Fdown += extracted_values.get(f'Pbuy_{grid_id}_{k}', 0) * self.C_buy[k]
                Fdown += extracted_values.get(f'Psell_{grid_id}_{k}', 0) * self.C_sell[k]

            # 添加弃电惩罚
            if grid.P_pv is not None:
                for k in range(24):
                    curtail_Ppv = max(0, grid.P_pv[k] - extracted_values.get(f'Ppv_{grid_id}_{k}', 0))
                    Fdown += curtail_Ppv * C_re
            if grid.P_wt is not None:
                for k in range(24):
                    curtail_Pwt = max(0, grid.P_wt[k] - extracted_values.get(f'Pwt_{grid_id}_{k}', 0))
                    Fdown += curtail_Pwt * C_re

            # 微电网群交易成本
            if self.num_microgrid > 1:
                for grid_id in range(self.num_microgrid):
                    for l in range(self.num_microgrid):
                        if l != grid_id:
                            for k in range(24):
                                pbuymic = extracted_values.get(f'Pbuymic_{grid_id}_{l}_{k}', 0)
                                psellmic = extracted_values.get(f'Psellmic_{grid_id}_{l}_{k}', 0)
                                Fdown += pbuymic * self.C_buy[k] + psellmic * self.C_sell[k]

        return Fdown

class Visualization:
    def __init__(self, optimization_manager, microgrids, num_microgrid, model):
        self.optimization_manager = optimization_manager
        self.num_microgrid = num_microgrid
        self.microgrids = microgrids
        self.model = model
        self.extracted_values = {}

    def extract_solution_to_dict(self):
        self.extracted_values.clear()

        # 从模型中提取每个变量的解决方案值
        for var in self.model.iter_variables():
            self.extracted_values[var.name] = var.solution_value

        # 特别处理微电网间交换功率Pnetmic
        for a in range(self.num_microgrid):
            for b in range(self.num_microgrid):
                if a != b:
                    for k in range(24):
                        var_name = f'Pbuymic_{a}_{b}_{k}'
                        if self.model.get_var_by_name(var_name):
                            self.extracted_values[var_name] = self.model.get_var_by_name(var_name).solution_value
                        var_name = f'Psellmic_{a}_{b}_{k}'
                        if self.model.get_var_by_name(var_name):
                            self.extracted_values[var_name] = self.model.get_var_by_name(var_name).solution_value


            # 创建DataFrame
            # 创建DataFrame
        df = pd.DataFrame(list(self.extracted_values.items()), columns=['Variable', 'Value'])

        # 导出到Excel
        df.to_excel('result' + '.xlsx', index=False)

    def plot_microgrid_charts(self, microgrid_id):
        # 为每个小时准备数据
        hours = np.arange(1, 25)
        Pmt_data = np.array([self.extracted_values.get(f'Pmt_{microgrid_id}_{k}', 0) for k in range(24)])
        Pde_data = np.array([self.extracted_values.get(f'Pde_{microgrid_id}_{k}', 0) for k in range(24)])
        Ppv_data = np.array([self.extracted_values.get(f'Ppv_{microgrid_id}_{k}', 0) for k in range(24)])
        Pwt_data = np.array([self.extracted_values.get(f'Pwt_{microgrid_id}_{k}', 0) for k in range(24)])
        Pcha_data = -np.array([self.extracted_values.get(f'Pcha_{microgrid_id}_{k}', 0) for k in range(24)])  # 负值
        Pdis_data = -np.array([self.extracted_values.get(f'Pdis_{microgrid_id}_{k}', 0) for k in range(24)])
        Pbuy_data = np.array([self.extracted_values.get(f'Pbuy_{microgrid_id}_{k}', 0) for k in range(24)])
        Psell_data = np.array([self.extracted_values.get(f'Psell_{microgrid_id}_{k}', 0) for k in range(24)]) # 负值

        # 微电网间的功率交换
        Pbuymic_data_dict = {}
        for j in range(self.num_microgrid):
            if j != microgrid_id:
                Pbuymic_data_dict[j] = np.array(
                    [self.extracted_values.get(f'Pbuymic_{microgrid_id}_{j}_{k}', 0) for k in range(24)])
        Psellmic_data_dict = {}
        for j in range(self.num_microgrid):
            if j != microgrid_id:
                Psellmic_data_dict[j] = np.array(
                    [self.extracted_values.get(f'Psellmic_{microgrid_id}_{j}_{k}', 0) for k in range(24)])

        # 绘制堆叠条形图
        plt.figure(figsize=(12, 8))
        plt.bar(hours, Pmt_data, label='Pmt')
        plt.bar(hours, Pde_data, bottom=Pmt_data, label='Pde')
        plt.bar(hours, Ppv_data, bottom=Pmt_data + Pde_data, label='Ppv')
        plt.bar(hours, Pwt_data, bottom=Pmt_data + Pde_data + Ppv_data, label='Pwt')
        plt.bar(hours, Pdis_data, bottom=Pmt_data + Pde_data + Ppv_data + Pwt_data, label='Pdis')
        plt.bar(hours, Pbuy_data, bottom=Pmt_data + Pde_data + Ppv_data + Pwt_data + Pdis_data, label='Pbuy')
        # 微电网间交换功率的堆叠
        current_bottom = Pmt_data + Pde_data + Ppv_data + Pwt_data + Pdis_data + Pbuy_data
        for j, Pbuymic_data in Pbuymic_data_dict.items():
            plt.bar(hours, Pbuymic_data, bottom=current_bottom, label=f'Pbuymic to Grid {j}')
            current_bottom = current_bottom.astype('float64')
            current_bottom += Pbuymic_data

        # 负数部分，需要从0往下堆叠
        plt.bar(hours, Pcha_data, label='Pcha')
        plt.bar(hours, Psell_data, bottom=Pcha_data, label='Psell')
        # 微电网间交换功率的堆叠（负数）
        for j, Psellmic_data in Psellmic_data_dict.items():
            plt.bar(hours, Psellmic_data, bottom=Pcha_data + Psell_data, label=f'Psellmic to Grid {j}')


        # 绘制负载和发电量的折线图
        plt.plot(hours, self.microgrids[microgrid_id].load, label='Load', color='red', linestyle='--')

        # 添加图例和标签
        plt.xlabel('Hour')
        plt.ylabel('Power (kW)')
        plt.title(f'Microgrid {microgrid_id}')
        plt.legend()
        plt.grid(True)
        plt.show()


def MGO(C_buy, C_sell):
    C_re = 0.1
    #======grid1========#
    id1 = 0
    load_1 = [88.24, 83.01, 80.15, 79.01, 76.07, 78.39, 89.95, 128.85, 155.45, 176.35, 193.71, 182.57, 179.64, 166.31, 164.61, 164.61, 174.48, 203.93, 218.99, 238.11, 216.14, 173.87, 131.07, 94.04]

    P_wt_1 = [66.9, 68.2, 71.9, 72, 78.8, 94.8, 114.3, 145.1, 155.5, 142.1, 115.9, 127.1, 141.8, 145.6, 145.3, 150, 206.9, 225.5, 236.1, 210.8, 198.6, 177.9, 147.2, 58.7]

    P_pv_1 = [0, 0, 0, 0, 0.06, 6.54, 20.19, 39.61, 49.64, 88.62, 101.59, 66.78, 110.46, 67.41, 31.53, 50.76, 20.6, 22.08, 2.07, 0, 0, 0, 0, 0]


    ebattery_1 = 300
    soc0_1 = 0.5
    socmin_1 = 0.3
    socmax_1 = 0.95
    pcs_1 = 40
    POWER_1 = 160  # 主网功率交换限制
    C_de_1 = 0.7939  # 柴油轮机成本系数
    POWER_MIC_1 = 100
    #======grid2========#
    id2 = 1
    P_pv_2 = [0, 0, 0, 0, 0, 0, 30.2278361336943, 41.4382515605911, 7.56783114759613, 72.7276154279869,
              16.6104475197874, 30.5651618832424, 52.8224212142948, 60.6698641822058, 13.8455529913677,
              41.3903809205061, 79.6270489842773, 56.6087525272567, 6.44538407391447, 0, 0, 0, 0, 0]

    load_2 = [176.48, 166.02, 160.3, 158.02, 152.14, 156.78, 179.9, 497.7, 110.9, 152.7, 187.42, 165.14,
              159.28, 132.62, 329.22, 329.22, 348.96, 407.86, 437.98, 476.22, 432.28, 347.74, 262.14, 188.08]
    ebattery_2 = 300  # 电池容量
    soc0_2 = 0.5  # 初始SOC
    socmin_2 = 0.3  # 最小SOC
    socmax_2 = 0.95  # 最大SOC
    pcs_2 = 40  # 充/放电功率限制
    C_mt_2 = 0.939  # 燃气轮机成本系数
    POWER_2 = 250  # 主网功率交换限制
    POWER_MIC_2 = 100
    #=======grid3=========#
    # 商业区域电网
    id3 = 2
    load_3 = [
    106.366569275484, 106.889216100582, 109.234278126231, 119.426356247170, 134.237337562312, 137.374520910876,
    155.384596992178, 163.419896012172, 176.311691418602, 178.445403906834, 187.748871931280, 189.117240142180,
    197.952879157646, 229.262602022253, 231.095578035511, 235.747030971555, 238.965724595163, 241.209217603922,
    248.626493624983, 124.237337562312, 119.426356247170, 109.234278126231, 106.889216100582, 106.366569275484
    ]

    # 热负荷
    LoadH = [x * 0.79 for x in load_3]
    ebattery_3 = 300  # 电池容量
    soc0_3 = 0.5  # 初始SOC
    socmin_3 = 0.3  # 最小SOC
    socmax_3 = 0.95  # 最大SOC
    pcs_3 = 40  # 充/放电功率限制
    C_mt_3 = 0.939  # 燃气轮机成本系数
    POWER_3 = 160  # 主网功率交换限制
    POWER_MIC_3 = 100

    # 创建 Microgrid 实例
    num_microgrid = 3
    grid1 = Microgrid(id1, load_1, POWER_1, POWER_MIC_1, ebattery_1, socmin_1, socmax_1, soc0_1, pcs_1, P_pv=P_pv_1, P_wt=P_wt_1, C_mt=None, C_de=C_de_1, C_re=C_re)
    grid2 = Microgrid(id2, load_2, POWER_2, POWER_MIC_2, ebattery_2, socmin_2, socmax_2, soc0_2, pcs_2, P_pv=P_pv_2, P_wt=None, C_mt=C_mt_2, C_de=None, C_re=C_re)
    grid3 = Microgrid(id3, load_3, POWER_3, POWER_MIC_3, ebattery_3, socmin_3, socmax_3, soc0_3, pcs_3, P_pv=None, P_wt=None, C_mt=C_mt_3, C_de=None, C_re=C_re)

    # 创建 TotalOptimizationManager 实例
    total_optimization_manager = TotalOptimizationManager([grid1, grid2, grid3], num_microgrid, C_buy, C_sell)

    # 设置优化问题
    total_optimization_manager.setup()

    # 求解优化问题
    solution = total_optimization_manager.solve()

    # 计算主电网交易量
    Pgrid_out, Pgrid_in = total_optimization_manager.calculate_grid_power_flows()

    # 获取目标函数的值
    Fdown = total_optimization_manager.calculate_objective(C_re)

    # #===============画图=================

    #打印结果
    total_optimization_manager.print_optimization_results(solution)

    # 创建可视化实例
    visualization = Visualization(total_optimization_manager, [grid1, grid2, grid3], num_microgrid, total_optimization_manager.model)
    visualization.extract_solution_to_dict()  # 提取解决方案

    # 对每个微电网绘制图表
    for microgrid_id in range(num_microgrid):
        visualization.plot_microgrid_charts(microgrid_id)
    #
    return Fdown, Pgrid_out, Pgrid_in




#测试#==================
#全局
C_buy = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.53, 0.53, 0.53, 0.82, 0.82,
        0.82, 0.82, 0.82, 0.53, 0.53, 0.53, 0.82, 0.82, 0.82, 0.53, 0.53, 0.53]

C_sell = [0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.42, 0.42, 0.42, 0.65, 0.65,
         0.65, 0.65, 0.65, 0.42, 0.42, 0.42, 0.65, 0.65, 0.65, 0.42, 0.42, 0.42]
Fdown, Pgrid_out, Pgrid_in = MGO(C_buy, C_sell)
print(Fdown, Pgrid_out, Pgrid_in)

# #全局
# C_buy = [0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.82, 0.82, 0.82, 1.35, 1.35, 1.35, 1.35, 1.35, 0.82, 0.82, 0.82, 1.35, 1.35, 1.35, 1.35, 1.35, 0.38, 0.38]
#
# C_sell = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
#
# Fdown, Pgrid_out, Pgrid_in = MGO(C_buy, C_sell)
# print(Fdown, Pgrid_out, Pgrid_in)
