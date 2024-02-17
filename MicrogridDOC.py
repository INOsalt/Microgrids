from gridinfo import C_buy, C_sell, expand_array
import numpy as np


class Microgrid:
    def __init__(self, id, load, POWER, POWER_MIC, POWER_SG, ebattery, socmin, socmax, soc0, pcs,
                 P_pv=None, P_wt=None, C_sg=None):
        #ID标识
        self.id = id
        # 负载 (Load)
        self.load = load
        # 风电 (WT)
        self.P_wt = P_wt
        # 光伏 (PV)
        self.P_pv = P_pv
        # 主网功率交换限制
        self.POWER = POWER
        # 微电网功率交换限制
        self.POWER_MIC = POWER_MIC
        # 储能电池参数
        self.ebattery = ebattery  # 电池容量
        self.socmin = socmin  # 最小SOC
        self.socmax = socmax  # 最大SOC
        self.soc0 = soc0  # 初始SOC
        self.pcs = pcs  # 充/放电功率限制
        self.C_sg = C_sg  # 柴油机机成本系数
        self.POWER_SG = POWER_SG # 分布式发电上限
        #self.C_de = C_de #燃气轮机
        #self.C_re = C_re #弃电惩罚

class OptimizationMicrogrid:
    def __init__(self, model, microgrid, num_microgrid,C_buymic,C_sellmic):
        # 初始化微电网实体、数量、购电和售电价格等参数
        self.microgrid = microgrid
        self.num_microgrid = num_microgrid
        self.C_buy = expand_array(C_buy)
        self.C_sell =expand_array(C_sell)
        self.C_buymic = C_buymic  # 购电价格
        self.C_sellmic = C_sellmic  # 售电价格
        self.model = model #Model(name="Microgrid Optimization")  # 创建 DOcplex 模型

    def add_variable(self):
        # 创建燃气机出力变量
        self.Psg = np.array([self.model.continuous_var(lb=0, ub=self.microgrid.POWER_SG,
                                                       name=f"Psg_{self.microgrid.id}_{i}")
                             for i in range(48)]) if self.microgrid.C_sg is not None else None

        # # 创建柴油出力变量
        # self.Pde = np.array([self.model.continuous_var(lb=-0, ub=200, name=f"Pde_{self.microgrid.id}_{i}") for i in range(48)]) if self.microgrid.C_de is not None else None

        # 创建 PV 出力变量
        #self.Ppv = np.array([self.model.continuous_var(lb=0, ub=self.model.infinity, name=f"Ppv_{self.microgrid.id}_{i}") for i in range(48)]) if self.microgrid.P_pv is not None else None

        # 创建 WT 出力变量
        #self.Pwt = np.array([self.model.continuous_var(lb=0, ub=self.model.infinity, name=f"Pwt_{self.microgrid.id}_{i}") for i in range(48)]) if self.microgrid.P_wt is not None else None

        # 创建蓄电池出力变量
        self.Pbat = np.array([self.model.continuous_var(lb=-self.model.infinity, ub=self.model.infinity,
                                                        name=f"Pbat_{self.microgrid.id}_{i}") for i in range(48)])

        # 创建充放电变量
        self.Pcha = np.array([self.model.continuous_var(lb=0, ub=self.model.infinity,
                                                        name=f"Pcha_{self.microgrid.id}_{i}") for i in range(48)])
        self.Pdis = np.array([self.model.continuous_var(lb=-self.model.infinity, ub=0,
                                                        name=f"Pdis_{self.microgrid.id}_{i}") for i in range(48)])

        # 创建充放电状态标志变量
        self.Temp_cha = np.array([self.model.binary_var(name=f"Temp_cha_{self.microgrid.id}_{i}") for i in range(48)])
        self.Temp_dis = np.array([self.model.binary_var(name=f"Temp_dis_{self.microgrid.id}_{i}") for i in range(48)])
        self.Temp_static = np.array([self.model.binary_var(name=f"Temp_static_{self.microgrid.id}_{i}") for i in range(48)])

        # 创建电网交换功率变量
        self.Pnet = np.array([self.model.continuous_var(lb=-self.microgrid.POWER, ub=self.microgrid.POWER,
                                                        name=f"Pnet_{self.microgrid.id}_{i}") for i in range(48)])

        # 创建电网购售电量变量
        self.Pbuy = np.array([self.model.continuous_var(lb=0, ub=self.microgrid.POWER,
                                                        name=f"Pbuy_{self.microgrid.id}_{i}") for i in range(48)])
        self.Psell = np.array([self.model.continuous_var(lb=-self.microgrid.POWER, ub=0,
                                                         name=f"Psell_{self.microgrid.id}_{i}") for i in range(48)])

        # 创建购售电标志变量
        self.Temp_net = np.array([self.model.binary_var(name=f"Temp_net_{self.microgrid.id}_{i}") for i in range(48)])


        # 创建微电网群交易相关变量
        if self.num_microgrid > 1:
            self.Pnetmic = np.array([[self.model.continuous_var(lb=-self.model.infinity, ub=self.model.infinity,
                                                                name=f"Pnetmic_{self.microgrid.id}_{j}_{i}")
                                      for i in range(48)] for j in range(self.num_microgrid) if j != self.microgrid.id])
            self.Pbuymic = np.array([[self.model.continuous_var(lb=0, ub=self.model.infinity,
                                                                name=f"Pbuymic_{self.microgrid.id}_{j}_{i}")
                                      for i in range(48)] for j in range(self.num_microgrid) if j != self.microgrid.id])
            self.Psellmic = np.array([[self.model.continuous_var(lb=-self.model.infinity, ub=0,
                                                                 name=f"Psellmic_{self.microgrid.id}_{j}_{i}")
                                       for i in range(48)] for j in range(self.num_microgrid) if j != self.microgrid.id])
            self.Tempmic = np.array([[self.model.binary_var(name=f"Tempmic_{self.microgrid.id}_{j}_{i}")
                                      for i in range(48)] for j in range(self.num_microgrid) if j != self.microgrid.id])

        else:
            self.Pnetmic = self.Pbuymic = self.Psellmic = None




    # =========约束===============
    # 添加约束
    # 主网功率交换约束
    def add_constraints(self):
        # 添加主网功率交换约束
        for k in range(48):
            self.model.add_constraint(self.Pnet[k] <= self.microgrid.POWER)
            self.model.add_constraint(self.Pnet[k] >= -self.microgrid.POWER)

        # 添加微电网间功率交换约束
        if self.num_microgrid > 1:
            # 微电网间的功率交换约束
            for l in range(self.num_microgrid):
                if l != self.microgrid.id:
                    for k in range(48):
                        var_name = f"Pnetmic_{self.microgrid.id}_{l}_{k}"
                        pnetmic_var = self.model.get_var_by_name(var_name)

                        # 添加功率交换上下限约束
                        self.model.add_constraint(pnetmic_var <= self.microgrid.POWER_MIC[l])
                        self.model.add_constraint(pnetmic_var >= -self.microgrid.POWER_MIC[l])
        # 添加燃气机和柴油机约束
        if self.microgrid.C_sg is not None:
            for k in range(48):
                self.model.add_constraint(self.Psg[k] <= self.microgrid.POWER_SG)
                self.model.add_constraint(self.Psg[k] >= 0)

        # 添加风能和太阳能约束
        # if self.microgrid.P_wt is not None:
        #     for k in range(48):
        #         self.model.add_constraint(self.Pwt[k] <= self.microgrid.P_wt[k])
        #         self.model.add_constraint(self.Pwt[k] >= 0)
        #
        # if self.microgrid.P_pv is not None:
        #     for k in range(48):
        #         self.model.add_constraint(self.Ppv[k] <= self.microgrid.P_pv[k])
        #         self.model.add_constraint(self.Ppv[k] >= 0)

        # 设置容差值
        tolerance = 0.00001

        # 添加功率平衡约束，允许一定的容差
        for k in range(48):
            power_balance_import = self.Pnet[k] - self.Pbat[k]
            if self.microgrid.P_pv is not None:
                power_balance_import += self.microgrid.P_pv[k]
            if self.microgrid.P_wt is not None:
                power_balance_import += self.microgrid.P_wt[k]
            if self.microgrid.C_sg is not None:
                power_balance_import += self.Psg[k]
            # if self.microgrid.C_de is not None:
            #     power_balance_import += self.Pde[k]

            # 微电网间的功率交换
            for j in range(self.num_microgrid):
                if j != self.microgrid.id:
                    pnetmic_var_name = f"Pnetmic_{self.microgrid.id}_{j}_{k}"
                    power_balance_import += self.model.get_var_by_name(pnetmic_var_name)

            self.model.add_constraint(power_balance_import <= self.microgrid.load[k] + tolerance)
            # 添加功率平衡的上限和下限约束
            self.model.add_constraint(power_balance_import <= self.microgrid.load[k] + tolerance)
            self.model.add_constraint(power_balance_import >= self.microgrid.load[k] - tolerance)



        # 添加购售电约束
        for k in range(48):
            # 当 Temp_net[k] 为 1 时，Pnet[k] 必须大于等于 0，且 Pbuy = Pnet，Psell = 0
            self.model.add_indicator(self.Temp_net[k], self.Pnet[k] >= 0, 1)
            self.model.add_indicator(self.Temp_net[k], self.Pbuy[k] == self.Pnet[k], 1)
            self.model.add_indicator(self.Temp_net[k], self.Psell[k] == 0, 1)

            # 当 Temp_net[k] 为 0 时，Pnet[k] 必须小于等于 0，且 Psell = -Pnet，Pbuy = 0
            self.model.add_indicator(self.Temp_net[k], self.Pnet[k] <= 0, 0)
            self.model.add_indicator(self.Temp_net[k], self.Psell[k] == self.Pnet[k], 0)
            self.model.add_indicator(self.Temp_net[k], self.Pbuy[k] == 0, 0)

        #添加微电网间交易的指示约束
        for k in range(48):
            for j in range(self.num_microgrid):
                if j != self.microgrid.id:
                    # 获取变量
                    tempmic_var_name = f"Tempmic_{self.microgrid.id}_{j}_{k}"
                    tempmic_var = self.model.get_var_by_name(tempmic_var_name)

                    pnetmic_var_name = f"Pnetmic_{self.microgrid.id}_{j}_{k}"
                    pnetmic_var = self.model.get_var_by_name(pnetmic_var_name)

                    pbuymic_var_name = f"Pbuymic_{self.microgrid.id}_{j}_{k}"
                    pbuymic_var = self.model.get_var_by_name(pbuymic_var_name)

                    psellmic_var_name = f"Psellmic_{self.microgrid.id}_{j}_{k}"
                    psellmic_var = self.model.get_var_by_name(psellmic_var_name)

                    # 添加指示约束
                    self.model.add_indicator(tempmic_var, pnetmic_var >= 0, 1) #1 买电 流入电网
                    self.model.add_indicator(tempmic_var, pbuymic_var == pnetmic_var, 1)
                    self.model.add_indicator(tempmic_var, psellmic_var == 0, 1)

                    self.model.add_indicator(tempmic_var, pnetmic_var <= 0, 0)
                    self.model.add_indicator(tempmic_var, psellmic_var == pnetmic_var, 0)
                    self.model.add_indicator(tempmic_var, pbuymic_var == 0, 0)

        # 添加电池充放电约束
        for k in range(48):
            self.model.add_constraint(self.Pbat[k] <= self.microgrid.pcs)
            self.model.add_constraint(self.Pbat[k] >= -self.microgrid.pcs)

            # 充电情况的指示约束
            self.model.add_indicator(self.Temp_cha[k], self.Pbat[k] >= 0, 1)
            self.model.add_indicator(self.Temp_cha[k], self.Pcha[k] == self.Pbat[k], 1)

            # 放电情况的指示约束
            self.model.add_indicator(self.Temp_dis[k], self.Pbat[k] <= 0, 1)
            self.model.add_indicator(self.Temp_dis[k], self.Pdis[k] == self.Pbat[k], 1)

            # 静置情况的指示约束
            self.model.add_indicator(self.Temp_static[k], self.Pbat[k] == 0, 1)
            self.model.add_indicator(self.Temp_static[k], self.Pcha[k] == 0, 1)
            self.model.add_indicator(self.Temp_static[k], self.Pdis[k] == 0, 1)

            # 确保每时刻只能处于一种状态
            self.model.add_constraint(self.Temp_cha[k] + self.Temp_dis[k] + self.Temp_static[k] == 1)

        # 添加总充电和放电量相等的约束
        total_charging = np.sum(self.Pcha)
        total_discharging = np.sum(self.Pdis)
        self.model.add_constraint(total_charging + total_discharging == 0)


        # SOC
        # SOC的初始值和变量
        initial_soc = self.microgrid.soc0 * self.microgrid.ebattery  # 初始SOC转换为实际电量

        # 添加SOC更新和上下限约束
        for k in range(1, 48):
            # 累积到当前时间步的电池充放电总和
            cumulative_soc_change = initial_soc + sum(self.Pbat[i] for i in range(k))

            # 添加SOC上下限约束
            self.model.add_constraint(cumulative_soc_change <= self.microgrid.socmax * self.microgrid.ebattery)
            self.model.add_constraint(cumulative_soc_change >= self.microgrid.socmin * self.microgrid.ebattery)

        # 确保第24小时结束时的SOC与初始SOC相等
        final_soc = initial_soc + sum(self.Pbat[i] for i in range(48))
        self.model.add_constraint(final_soc == initial_soc)


    # =========目标===============
    # 设置目标函数 常数项没加
    def add_objective(self):
        # 创建目标函数表达式
        objective_expr = 0

        # 添加柴油轮机成本
        if self.microgrid.C_sg is not None:
            for k in range(48):
                objective_expr += self.Psg[k] * self.microgrid.C_sg

        # 添加储能成本
        for k in range(48):
            objective_expr += self.Pcha[k] * 0.339  # 假设储能成本为0.339
            objective_expr -= self.Pdis[k] * 0.339

        # 添加外电网购售成本
        for k in range(48):
            objective_expr += self.Pbuy[k] * self.C_buy[k]
            objective_expr += self.Psell[k] * self.C_sell[k]

        # 添加微电网购电成本
        # 如果有多于一个微电网
        if self.num_microgrid > 1:
            for k in range(48):
                for l in range(self.num_microgrid):
                    if l != self.microgrid.id:
                        pbuymic_var_name = f"Pbuymic_{self.microgrid.id}_{l}_{k}"
                        psellmic_var_name = f"Psellmic_{self.microgrid.id}_{l}_{k}"

                        pbuymic_var = self.model.get_var_by_name(pbuymic_var_name)
                        psellmic_var = self.model.get_var_by_name(psellmic_var_name)

                        # 将购买和销售电力的成本加入目标函数
                        objective_expr += pbuymic_var * self.C_buymic[k]
                        objective_expr += psellmic_var * self.C_sellmic[k]

        # # 弃电惩罚
        # if self.microgrid.P_pv is not None:
        #     for k in range(48):
        #         curtail_Ppv = self.model.max(0, self.microgrid.P_pv[k] - self.Ppv[k])
        #         objective_expr += curtail_Ppv * self.microgrid.C_re  # 光伏弃电惩罚
        # if self.microgrid.P_wt is not None:
        #     for k in range(48):
        #         curtail_Pwt = self.model.max(0, self.microgrid.P_wt[k] - self.Pwt[k])
        #         objective_expr += curtail_Pwt * self.microgrid.C_re  # 风电弃电惩罚

        # LL = self.microgrid.load # + Ldr_1 + Lidr_1
        # Peak-Valley Difference Indicator (峰谷差指标)
        # FGX = np.max(LL) - np.min(LL)
        # Deviation Indicator (偏差指标)
        # Fpl = np.sum(np.abs(LL - np.mean(LL)))
        # Cost of Demand Response (CostDR)
        #CostDR = np.sum(Ldr_1 * 0.025 + Lidr_1 * 0.05)
        #objective_expr += FGX * 20 + Fpl


        return objective_expr



