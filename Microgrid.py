import cplex
from cplex.exceptions import CplexError
class Microgrid:
    def __init__(self, id, load, POWER, POWER_MIC, ebattery, socmin, socmax, soc0, pcs, P_pv=None, P_wt=None, C_mt=None, C_de=None):
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
        self.C_mt = C_mt  # 柴油机机成本系数
        self.C_de = C_de #燃气轮机
        #self.C_re = C_re #弃电惩罚

class OptimizationMicrogrid:
    def __init__(self, microgrid, num_microgrid, C_buy, C_sell):
        # microgrid微电网实体, num_microgrid微电网数量,
        self.microgrid = microgrid
        self.num_microgrid = num_microgrid
        # self.microgrid_id = microgrid.id
        #
        # self.load = microgrid.load
        # self.P_wt = microgrid.P_wt
        # self.P_pv = microgrid.P_pv
        # self.POWER = microgrid.POWER
        # self.POWER_MIC = microgrid.POWER_MIC  # 微电网联络线限制
        # self.Ebattery = microgrid.ebattery  # 电网联络线限制
        # self.socmin = microgrid.socmin
        # self.socmax = microgrid.socmax
        # self.soc0 = microgrid.soc0
        # self.Pcs = microgrid.Pcs
        # self.C_mt = microgrid.C_mt  # 柴油机机成本系数
        # self.C_de = microgrid.C_de  # 燃气轮
        self.C_buy = C_buy  # 购电价格
        self.C_sell = C_sell  # 售电价格
        self.problem = cplex.Cplex()  # 或者从外部传入
        self.variables_info = []  # 存储变量信息
        self.constraints_info = []  # 存储约束信息


    def add_variable(self):
        try:
            # 柴油机出力变量
            if self.microgrid.C_mt is not None:
                for i in range(24):
                    var_name = f"Pmt_{self.microgrid.id}_{i}"
                    self.problem.variables.add(names=[var_name], lb=[-cplex.infinity], ub=[cplex.infinity])
            # 燃气出力变量
            if self.microgrid.C_de is not None:
                for i in range(24):
                    var_name = f"Pde_{self.microgrid.id}_{i}"
                    self.problem.variables.add(names=[var_name], lb=[-cplex.infinity], ub=[cplex.infinity])
            # PV出力变量
            if self.microgrid.P_pv is not None:
                for i in range(24):
                    var_name = f"Ppv_{self.microgrid.id}_{i}"
                    self.problem.variables.add(names=[var_name], lb=[0], ub=[cplex.infinity])
            # WT出力变量
            if self.microgrid.P_wt is not None:
                for i in range(24):
                    var_name = f"Pwt_{self.microgrid.id}_{i}"
                    self.problem.variables.add(names=[var_name], lb=[0], ub=[cplex.infinity])
            # 电池：
            # 蓄电池出力
            for i in range(24):
                var_name = f"Pbat_{self.microgrid.id}_{i}"
                self.problem.variables.add(names=[var_name], lb=[-cplex.infinity], ub=[cplex.infinity])
            # 充放电
            for i in range(24):
                var_name = f"Pcha_{self.microgrid.id}_{i}"
                self.problem.variables.add(names=[var_name], lb=[0], ub=[cplex.infinity])
            for i in range(24):
                var_name = f"Pdis_{self.microgrid.id}_{i}"
                self.problem.variables.add(names=[var_name], lb=[-cplex.infinity], ub=[0])
            for i in range(24):
                var_name = f"Temp_cha_{self.microgrid.id}_{i}"
                self.problem.variables.add(names=[var_name], types=["B"])
            for i in range(24):
                var_name = f"Temp_dis_{self.microgrid.id}_{i}"
                self.problem.variables.add(names=[var_name], types=["B"])
            for i in range(24):
                var_name = f"Temp_static_{self.microgrid.id}_{i}"
                self.problem.variables.add(names=[var_name], types=["B"])
            # 电网
            # 添加电网交换功率变量 (Pnet)
            for i in range(24):
                var_name = f"Pnet_{self.microgrid.id}_{i}"
                self.problem.variables.add(names=[var_name], lb=[-cplex.infinity], ub=[cplex.infinity])
            # 添加从电网购电量的连续变量 (Pbuy)
            for i in range(24):
                var_name = f"Pbuy_{self.microgrid.id}_{i}"
                self.problem.variables.add(names=[var_name], lb=[0], ub=[cplex.infinity])
            # 添加向电网售电量的连续变量 (Psell)
            for i in range(24):
                var_name = f"Psell_{self.microgrid.id}_{i}"
                self.problem.variables.add(names=[var_name], lb=[-cplex.infinity], ub=[0])
            # 添加购/售电标志的二进制变量 (Temp_net)
            for i in range(24):
                var_name = f"Temp_net_{self.microgrid.id}_{i}"
                self.problem.variables.add(names=[var_name], types=["B"])

            # 添加微电网群交易相关的连续变量
            for i in range(24):
                for j in range(int(self.num_microgrid)):
                    # 对于每对微电网，创建唯一的变量名
                    var_name_netmic = f"Pnetmic_{self.microgrid.id}_{j}_{i}"
                    var_name_buymic = f"Pbuymic_{self.microgrid.id}_{j}_{i}"
                    var_name_sellmic = f"Psellmic_{self.microgrid.id}_{j}_{i}"
                    # 判断是否应该添加这些变量
                    if j != self.microgrid.id:
                        self.problem.variables.add(names=[var_name_netmic], lb=[-cplex.infinity], ub=[cplex.infinity])
                        self.problem.variables.add(names=[var_name_buymic], lb=[0], ub=[cplex.infinity])  # 电网买进
                        self.problem.variables.add(names=[var_name_sellmic], lb=[-cplex.infinity], ub=[0])  # 电网卖出

            # 添加微电网群交易的购/售电标志二进制变量
            for i in range(24):
                for j in range(int(self.num_microgrid)):
                    var_name = f"Tempmic_{self.microgrid.id}_{j}_{i}"
                    # 判断是否应该添加这些变量
                    if j != self.microgrid.id:
                        self.problem.variables.add(names=[var_name], types=["B"])

        except CplexError as exc:
            print(exc)
            return None
    # =========约束===============
    # 添加约束
    # 主网功率交换约束
    def add_constraints(self):#cplex不支持numpy直接输入，可以预处理
        try:
            # 主网功率交换约束
            for k in range(24):
                self.problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[f"Pnet_{self.microgrid.id}_{k}"], val=[1.0])],
                    senses=["L"],#约束的类型L 小于等于。
                    rhs=[self.microgrid.POWER]#约束的右侧值，意味着 Pnet_k <= self.POWER
                )
                self.problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[f"Pnet_{self.microgrid.id}_{k}"], val=[1.0])],
                    senses=["G"],
                    rhs=[-self.microgrid.POWER]
                )

            # 微电网网功率交换约束
            for l in range(int(self.num_microgrid)):
                for k in range(24):
                    if l != self.microgrid.id:
                        self.problem.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(ind=[f"Pnetmic_{self.microgrid.id}_{l}_{k}"], val=[1.0])],
                            senses=["L"],
                            rhs=[self.microgrid.POWER_MIC]
                        )
                        self.problem.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(ind=[f"Pnetmic_{self.microgrid.id}_{l}_{k}"], val=[1.0])],
                            senses=["G"],
                            rhs=[-self.microgrid.POWER_MIC]
                        )

            #风光约束
            if self.microgrid.P_wt is not None:
                for k in range(24):
                    # 添加对 Pwt_k 的约束：0 <= Pwt_k <= self.P_wt[k]
                    self.problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=[f"Pwt_{self.microgrid.id}_{k}"], val=[1.0])],
                        senses=["L"],
                        rhs=[self.microgrid.P_wt[k]]
                    )
                    self.problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=[f"Pwt_{self.microgrid.id}_{k}"], val=[1.0])],
                        senses=["G"],
                        rhs=[0]
                    )
            if self.microgrid.P_pv is not None:
                for k in range(24):
                    # 添加对 Ppv_k 的约束：0 <= Ppv_k <= self.P_pv[k]
                    self.problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=[f"Ppv_{self.microgrid.id}_{k}"], val=[1.0])],
                        senses=["L"],
                        rhs=[self.microgrid.P_pv[k]]
                    )
                    self.problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=[f"Ppv_{self.microgrid.id}_{k}"], val=[1.0])],
                        senses=["G"],
                        rhs=[0]
                    )

            # 功率平衡约束
            for k in range(24):
                var_names = [f"Pnet_{self.microgrid.id}_{k}", f"Pbat_{self.microgrid.id}_{k}"]

                # 检查并添加PV相关变量
                if self.microgrid.P_pv is not None:
                    var_names.append(f"Ppv_{self.microgrid.id}_{k}")

                # 检查并添加WT相关变量
                if self.microgrid.P_wt is not None:
                    var_names.append(f"Pwt_{self.microgrid.id}_{k}")

                # 检查并添加MT（柴油机）相关变量
                if self.microgrid.C_mt is not None:
                    var_names.append(f"Pmt_{self.microgrid.id}_{k}")

                # 检查并添加DE（燃气轮机）相关变量
                if self.microgrid.C_de is not None:
                    var_names.append(f"Pde_{self.microgrid.id}_{k}")

                # 为每个相邻的微电网添加交换功率变量
                for j in range(int(self.num_microgrid)):
                    if j != self.microgrid.id:
                        var_names.append(f"Pnetmic_{self.microgrid.id}_{j}_{k}")

                # 计算所有变量的系数
                coefficients = [1.0] * len(var_names)
                coefficients[var_names.index(f"Pbat_{self.microgrid.id}_{k}")] = -1.0  # 电池出力的系数设为 -1

                # 创建并添加功率平衡约束
                self.problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=var_names, val=coefficients)],
                    senses=["E"],
                    rhs=[self.microgrid.load[k]]
                )

            # 购售电
            for k in range(24):
                # 当 Temp_net 等于 1 时，添加以下约束：
                # 如果 Temp_net_k 为 1，则 Pnet_k 必须大于等于 0
                self.problem.indicator_constraints.add(
                    indvar=f"Temp_net_{self.microgrid.id}_{k}",  # 指示变量 Temp_net_k
                    complemented=0,  # 指示变量不取反
                    rhs=0.0,  # 右侧值为 0
                    sense="G",  # 大于等于
                    lin_expr=cplex.SparsePair(ind=[f"Pnet_{self.microgrid.id}_{k}"], val=[1.0])  # 线性表达式：Pnet_k >= 0
                )

                # 当 Temp_net_k 为 1 时，添加 Pbuy_k = Pnet_k 的约束
                self.problem.linear_constraints.add(
                    lin_expr=[
                        cplex.SparsePair(ind=[f"Pbuy_{self.microgrid.id}_{k}", f"Pnet_{self.microgrid.id}_{k}"],
                                         val=[1.0, -1.0])],
                    senses=["E"],
                    rhs=[0]  # Pbuy_k - Pnet_k = 0
                )

                # 当 Temp_net_k 为 1 时，添加 Psell_k = 0 的约束
                self.problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[f"Psell_{self.microgrid.id}_{k}"], val=[1.0])],
                    senses=["E"],
                    rhs=[0]  # Psell_k = 0
                )

                # 当 Temp_net 等于 0 时，添加以下约束：
                # 如果 Temp_net_k 为 0，则 Pnet_k 必须小于等于 0
                self.problem.indicator_constraints.add(
                    indvar=f"Temp_net_{self.microgrid.id}_{k}",  # 指示变量 Temp_net_k
                    complemented=1,  # 指示变量取反
                    rhs=0.0,  # 右侧值为 0
                    sense="L",
                    lin_expr=cplex.SparsePair(ind=[f"Pnet_{self.microgrid.id}_{k}"], val=[1.0])  # 线性表达式：Pnet_k <= 0
                )

                # 当 Temp_net_k 为 0 时，添加 Psell_k = Pnet_k 的约束
                self.problem.linear_constraints.add(
                    lin_expr=[
                        cplex.SparsePair(ind=[f"Psell_{self.microgrid.id}_{k}", f"Pnet_{self.microgrid.id}_{k}"],
                                         val=[1.0, -1.0])],
                    senses=["E"],
                    rhs=[0]  # 右侧值为 0，表示 Psell_k - Pnet_k = 0，即 Psell_k = Pnet_k
                )

                # 当 Temp_net_k 为 0 时，添加 Pbuy_k = 0 的约束
                self.problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[f"Pbuy_{self.microgrid.id}_{k}"], val=[1.0])],
                    senses=["E"],  # 约束类型为 'E' (Equal, 即等于)
                    rhs=[0]  # 右侧值为 0，表示 Pbuy_k = 0
                )

            # 添加蓄电池充放电约束
            for k in range(24):
                # 电池充放电功率约束
                self.problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[f"Pbat_{self.microgrid.id}_{k}"], val=[1.0])],
                    senses=["L"], rhs=[self.microgrid.Pcs]
                )
                self.problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[f"Pbat_{self.microgrid.id}_{k}"], val=[1.0])],
                    senses=["G"], rhs=[-self.microgrid.Pcs]
                )
                # 为 Pcha 和 Pdis 添加类似的约束

                # 充电情况的指示约束
                self.problem.indicator_constraints.add(
                    indvar=f"Temp_cha_{self.microgrid.id}_{k}", complemented=0, rhs=0.0, sense="G",
                    lin_expr=cplex.SparsePair(ind=[f"Pbat_{self.microgrid.id}_{k}"], val=[1.0])
                )
                # 添加 Pcha = Pbat, Pdis = 0 的线性约束

                # 放电情况的指示约束
                self.problem.indicator_constraints.add(
                    indvar=f"Temp_dis_{self.microgrid.id}_{k}", complemented=0, rhs=0.0, sense="L",
                    lin_expr=cplex.SparsePair(ind=[f"Pbat_{self.microgrid.id}_{k}"], val=[1.0])
                )
                # 添加 Pdis = Pbat, Pcha = 0 的线性约束

                # 静置情况的指示约束
                self.problem.indicator_constraints.add(
                    indvar=f"Temp_static_{self.microgrid.id}_{k}", complemented=0, rhs=0.0, sense="E",
                    lin_expr=cplex.SparsePair(ind=[f"Pbat_{self.microgrid.id}_{k}"], val=[1.0])
                )
                # 添加 Pdis = 0, Pcha = 0 的线性约束

                # 确保在任何时刻只能处于充电、放电或静置状态之一
                self.problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[f"Temp_cha_{self.microgrid.id}_{k}", f"Temp_dis_{self.microgrid.id}_{k}",
                             f"Temp_static_{self.microgrid.id}_{k}"],
                        val=[1.0, 1.0, 1.0])],
                    senses=["E"], rhs=[1]
                )

            # SOC 约束
            # 注意：此处 SOC 约束的实现需要小心处理
            cumulative_soc = self.microgrid.soc0 * self.microgrid.Ebattery
            for k in range(24):
                # 注意：这里的 Pcha_{k} 和 Pdis_{k} 应该是特定于微电网实体的
                cumulative_soc_expr = cplex.SparsePair(
                    ind=[f"Pcha_{self.microgrid.id}_{k}", f"Pdis_{self.microgrid.id}_{k}"],
                    val=[self.microgrid.Ebattery, -self.microgrid.Ebattery])
                self.problem.linear_constraints.add(
                    lin_expr=cumulative_soc_expr,
                    senses=["L"],
                    rhs=[self.microgrid.Ebattery * self.microgrid.socmax - cumulative_soc]
                )
                self.problem.linear_constraints.add(
                    lin_expr=cumulative_soc_expr,
                    senses=["G"],
                    rhs=[self.microgrid.Ebattery * self.microgrid.socmin - cumulative_soc]
                )
                # 更新累积 SOC 的表达式，适用于后续时段

        except CplexError as exc:
            print(exc)
            return None
    # =========目标===============
    # 设置目标函数 常数项没加
    def add_objective(self):
        try:
            problem = self.problem
            objective = []

            # 添加燃气轮机成本
            if self.microgrid.C_de is not None:
                for k in range(24):
                    objective.append((f"Pde_{self.microgrid.id}_{k}", self.C_de))

            # 添加柴油轮机成本
            if self.microgrid.C_mt is not None:
                for k in range(24):
                    objective.append((f"Pmt_{self.microgrid.id}_{k}", self.C_de))

            # 添加储能成本（注意：绝对值问题需要特别处理）
            for k in range(24):
                objective.append((f"Pbat_{self.microgrid.id}_{k}", 0.339))  # 需要引入辅助变量来处理绝对值

            # 添加外电网购售成本
            for k in range(24):
                objective.append((f"Pbuy_{self.microgrid.id}_{k}", self.C_buy[k]))
                objective.append((f"Psell_{self.microgrid.id}_{k}", -self.C_sell[k]))  # 负号表示售电可能减少成本

            # 添加微电网购电成本
            for k in range(24):
                for l in range(int(self.num_microgrid)):
                    if l != self.microgrid.id:
                        objective.append((f"Pbuymic_{self.microgrid.id}_{l}_{k}", self.C_buy[k]))
                        objective.append((f"Psellmic_{self.microgrid.id}_{l}_{k}", -self.C_sell[k]))

            # 弃电惩罚 不支持f"Pwt_{k}"-self.wt,需要修改
            # for k in range(24):
            #     # 对于每个时段 k，添加风电和光伏的实际产出的弃电惩罚
            #     objective.append((f"Pwt_{k}", -self.microgrid.C_re))  # 实际风电产出的弃电惩罚
            #     objective.append((f"Ppv_{k}", -self.microgrid.C_re))  # 实际光伏产出的弃电惩罚


            # 需求响应成本

            # 设置目标函数
            objective = problem.objective.set_linear(objective)
            problem.objective.set_sense(problem.objective.sense.minimize)
            return objective

        except CplexError as exc:
            print(exc)
            return None





    def solve(self):
        problem = self.setup_problem()
        problem.solve()

        # 提取和处理结果

