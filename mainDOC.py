from MicrogridDOC import Microgrid, OptimizationMicrogrid
from docplex.mp.model import Model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gridinfo import microgrid_pv, microgrid_wt, microgrid_load_dict, C_buy, C_sell, expand_array

#整体优化============
class TotalOptimizationManager:
    def __init__(self, microgrids, num_microgrid, C_buymic, C_sellmic):
        self.microgrids = microgrids
        self.num_microgrid = num_microgrid
        self.C_buymic = C_buymic
        self.C_sellmic = C_sellmic
        self.model = Model(name="Microgrid Optimization Problem")

    def setup(self):
        objective_all = 0
        for i in range(len(self.microgrids)):
            optimization = OptimizationMicrogrid(self.model, self.microgrids[i], self.num_microgrid,
                                                 self.C_buymic,self.C_sellmic)
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

            # # 如果未找到最优解，则检查未满足的约束
            # if solve_details.status in ['infeasible', 'integer infeasible']:
            #     print("Checking for unsatisfied constraints...")
            #     unsatisfied_constraints = self.model.find_unsatisfied_constraints()
            #     for constraint in unsatisfied_constraints:
            #         print(constraint)


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
                Fdown += extracted_values.get(f'Pbuy_{grid_id}_{k}', 0) * C_buy[k]
                Fdown += extracted_values.get(f'Psell_{grid_id}_{k}', 0) * C_sell[k]

            # 微电网群交易成本
            if self.num_microgrid > 1:
                for grid_id in range(self.num_microgrid):
                    for l in range(self.num_microgrid):
                        if l != grid_id:
                            for k in range(24):
                                pbuymic = extracted_values.get(f'Pbuymic_{grid_id}_{l}_{k}', 0)
                                psellmic = extracted_values.get(f'Psellmic_{grid_id}_{l}_{k}', 0)
                                Fdown += pbuymic * self.C_buymic[k] + psellmic * self.C_sellmic[k]

        return Fdown

    def extract_pnetmic_values_by_hour(self):
        """
        提取优化后的Pnetmic_{a}_{b}_{k}值，首先按小时k索引，然后按微电网对(a, b)。
        """
        pnetmic_values_by_hour = {}
        for k in range(24):
            for a in range(self.num_microgrid):
                for b in range(self.num_microgrid):
                    if a != b:  # 仅考虑不同微电网间的交易
                        var_name = f"Pnetmic_{a}_{b}_{k}"
                        variable = self.model.get_var_by_name(var_name)
                        if variable is not None:  # 如果变量存在
                            if k not in pnetmic_values_by_hour:
                                pnetmic_values_by_hour[k] = {}
                            pnetmic_values_by_hour[k][(a, b)] = variable.solution_value
                        else:
                            pnetmic_values_by_hour[k][(a, b)] = 0
        return pnetmic_values_by_hour

    def extract_pnet_values_by_hour(self):
        """
        提取优化后的Pnet_{a}_{k}值，首先按小时k索引，然后按微电网a。
        """
        pnet_values_by_hour = {}
        for k in range(24):
            for a in range(self.num_microgrid):
                var_name = f"Pnet_{a}_{k}"
                variable = self.model.get_var_by_name(var_name)
                if variable is not None:  # 如果变量存在
                    if k not in pnet_values_by_hour:
                        pnet_values_by_hour[k] = {}
                    pnet_values_by_hour[k][a] = variable.solution_value
                else:
                    pnet_values_by_hour[k][a] = 0
        return pnet_values_by_hour

    def extract_psg_values_by_hour(self):
        """
        提取优化后的Psg_{microgrid.id}_{i}值，按小时i索引。
        每个小时的键对应一个数组，包含了所有微电网在该小时的Psg值。
        """
        psg_values_by_hour = {}
        for i in range(24):  # 对于一天中的每个小时
            psg_values_by_hour[i] = []  # 初始化当前小时的Psg列表
            for microgrid_id in range(self.num_microgrid):  # 遍历每个微电网
                var_name = f"Psg_{microgrid_id}_{i}"  # 构建变量名
                variable = self.model.get_var_by_name(var_name)  # 尝试获取变量
                if variable is not None:  # 如果变量存在
                    psg_values_by_hour[i].append(variable.solution_value)  # 添加变量的解决方案值到列表
                else:
                    psg_values_by_hour[i].append(0)  # 如果变量不存在，添加0到列表
        return psg_values_by_hour

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
        Psg_data = np.array([self.extracted_values.get(f'Psg_{microgrid_id}_{k}', 0) for k in range(24)])
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
        plt.bar(hours, Psg_data, label='Psg')
        plt.bar(hours, Ppv_data, bottom=Psg_data, label='Ppv')
        plt.bar(hours, Pwt_data, bottom=Psg_data + Ppv_data, label='Pwt')
        plt.bar(hours, Pdis_data, bottom=Psg_data + Ppv_data + Pwt_data, label='Pdis')
        plt.bar(hours, Pbuy_data, bottom=Psg_data + Ppv_data + Pwt_data + Pdis_data, label='Pbuy')
        # 微电网间交换功率的堆叠
        current_bottom = Psg_data + Ppv_data + Pwt_data + Pdis_data + Pbuy_data
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


def MGO(C_buymic, C_sellmic, EVload):
    C_re = 0.1


    #======grid1========#
    id1 = 0
    load_1 = microgrid_load_dict[id1]
    P_wt_1 = microgrid_wt[id1]
    P_pv_1 = microgrid_pv[id1]
    ev_load_1 = EVload[id1]
    load_1 = ev_load_1 + load_1

    ebattery_1 = 300
    soc0_1 = 0.5
    socmin_1 = 0.3
    socmax_1 = 0.95
    pcs_1 = 40
    POWER_1 = 20000  # 主网功率交换限制
    C_sg_1 = 0.7939  # 柴油轮机成本系数
    POWER_MIC_1 = [0, 18000, 18000, 0] #1-1234 #12 13 24 23 34 18000, 18000, 16000, 16000, 16000
    POWER_SG_1 = 15000
    #======grid2========#
    id2 = 1
    load_2 = microgrid_load_dict[id2]
    P_wt_2 = microgrid_wt[id2]
    P_pv_2 = microgrid_pv[id2]
    ev_load_2 = EVload[id2]
    load_2 = ev_load_2 + load_2

    ebattery_2 = 300  # 电池容量
    soc0_2 = 0.5  # 初始SOC
    socmin_2 = 0.3  # 最小SOC
    socmax_2 = 0.95  # 最大SOC
    pcs_2 = 40  # 充/放电功率限制
    C_sg_2 = 0.7939  # 柴油轮机成本系数
    POWER_2 = 20000  # 主网功率交换限制
    POWER_MIC_2 = [18000, 0, 16000, 16000] #2-1234 #12 13 24 23 34 18000, 18000, 16000, 16000, 16000
    POWER_SG_2 = 6000
    #=======grid3=========#
    # 商业区域电网
    id3 = 2
    load_3 = microgrid_load_dict[id3]
    P_wt_3 = microgrid_wt[id3]
    P_pv_3 = microgrid_pv[id3]
    ev_load_3 = EVload[id3]
    load_3 = ev_load_3 + load_3

    ebattery_3 = 300  # 电池容量
    soc0_3 = 0.5  # 初始SOC
    socmin_3 = 0.3  # 最小SOC
    socmax_3 = 0.95  # 最大SOC
    pcs_3 = 40  # 充/放电功率限制
    C_sg_3 = 0.7939  # 柴油轮机成本系数
    POWER_3 = 20000  # 主网功率交换限制
    POWER_MIC_3 = [18000, 16000, 0, 16000] #3-1234 #12 13 24 23 34 18000, 18000, 16000, 16000, 16000
    POWER_SG_3 = 6000

    # =======grid4=========#
    # 商业区域电网
    id4 = 4
    load_4 = microgrid_load_dict[id4]
    P_wt_4 = microgrid_wt[id4]
    P_pv_4 = microgrid_pv[id4]
    ev_load_4 = EVload[id4]
    load_4 = ev_load_4 + load_4

    ebattery_4 = 300  # 电池容量
    soc0_4 = 0.5  # 初始SOC
    socmin_4 = 0.3  # 最小SOC
    socmax_4 = 0.95  # 最大SOC
    pcs_4 = 40  # 充/放电功率限制
    C_sg_4 = 0.7939  # 柴油轮机成本系数
    POWER_4 = 0  # 主网功率交换限制
    POWER_MIC_4 = [0, 16000, 16000, 0]  # 4-1234 #12 13 24 23 34 18000, 18000, 16000, 16000, 16000
    POWER_SG_4 = 4000

    #===============优化部分===========#
    # 创建 Microgrid 实例
    num_microgrid = 4
    grid1 = Microgrid(id1, load_1, POWER_1, POWER_MIC_1, POWER_SG_1, ebattery_1, socmin_1, socmax_1, soc0_1, pcs_1,
                      P_pv=P_pv_1, P_wt=P_wt_1, C_sg=C_sg_1)
    grid2 = Microgrid(id2, load_2, POWER_2, POWER_MIC_2, POWER_SG_2, ebattery_2, socmin_2, socmax_2, soc0_2, pcs_2,
                      P_pv=P_pv_2, P_wt=P_wt_2, C_sg=C_sg_2)
    grid3 = Microgrid(id3, load_3, POWER_3, POWER_MIC_3, POWER_SG_3, ebattery_3, socmin_3, socmax_3, soc0_3, pcs_3,
                      P_pv=P_pv_3, P_wt=P_wt_3, C_sg=C_sg_3)
    grid4 = Microgrid(id4, load_4, POWER_4, POWER_MIC_4, POWER_SG_4, ebattery_4, socmin_4, socmax_4, soc0_4, pcs_4,
                      P_pv=P_pv_4, P_wt=P_wt_4, C_sg=C_sg_4)


    # 创建 TotalOptimizationManager 实例
    total_optimization_manager = TotalOptimizationManager([grid1, grid2, grid3, grid4], num_microgrid,
                                                          C_buymic, C_sellmic)

    # 设置优化问题
    total_optimization_manager.setup()

    # 求解优化问题
    solution = total_optimization_manager.solve()

    # 计算主电网交易量
    #Pgrid_out, Pgrid_in = total_optimization_manager.calculate_grid_power_flows()

    # 获取目标函数的值
    Fdown = total_optimization_manager.calculate_objective(C_re)
    pnetmic_values_by_hour = total_optimization_manager.extract_pnetmic_values_by_hour()
    pnet_values_by_hour = total_optimization_manager.extract_pnet_values_by_hour()
    psg_values_by_hour = total_optimization_manager.extract_psg_values_by_hour()

    # #===============画图=================

    # #打印结果
    # total_optimization_manager.print_optimization_results(solution)
    #
    # # 创建可视化实例
    # visualization = Visualization(total_optimization_manager, [grid1, grid2, grid3], num_microgrid, total_optimization_manager.model)
    # visualization.extract_solution_to_dict()  # 提取解决方案
    #
    # # 对每个微电网绘制图表
    # for microgrid_id in range(num_microgrid):
    #     visualization.plot_microgrid_charts(microgrid_id)
    # #
    return Fdown, pnetmic_values_by_hour, pnet_values_by_hour, psg_values_by_hour # 长度48




#测试#==================
#全局
# C_buy = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.53, 0.53, 0.53, 0.82, 0.82,
#         0.82, 0.82, 0.82, 0.53, 0.53, 0.53, 0.82, 0.82, 0.82, 0.53, 0.53, 0.53]
#
# C_sell = [0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.42, 0.42, 0.42, 0.65, 0.65,
#          0.65, 0.65, 0.65, 0.42, 0.42, 0.42, 0.65, 0.65, 0.65, 0.42, 0.42, 0.42]
# Fdown = MGO(C_buy, C_sell)
# print(Fdown)

# #全局
# C_buy = [0.38, 0.38, 0.38, 0.38, 0.38, 0.38, 0.82, 0.82, 0.82, 1.35, 1.35, 1.35, 1.35, 1.35, 0.82, 0.82, 0.82, 1.35, 1.35, 1.35, 1.35, 1.35, 0.38, 0.38]
#
# C_sell = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
#
# Fdown, Pgrid_out, Pgrid_in = MGO(C_buy, C_sell)
# print(Fdown, Pgrid_out, Pgrid_in)
