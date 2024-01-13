import cplex
from cplex.exceptions import CplexError
from cplex.exceptions import CplexError
class Microgrid:
    def __init__(self, P_pv, P_wt, load, ebattery, soc0, socmin, socmax, pcs, POWER, POWER_MIC):
        # 光伏 (PV)
        self.P_pv = P_pv
        # 风电 (WT)
        self.P_wt = P_wt
        # 负载 (Load)
        self.load = load
        # 储能电池参数
        self.ebattery = ebattery  # 电池容量
        self.soc0 = soc0  # 初始SOC
        self.socmin = socmin  # 最小SOC
        self.socmax = socmax  # 最大SOC
        self.pcs = pcs  # 充/放电功率限制
        # 主网功率交换限制
        self.POWER = POWER
        # 微电网功率交换限制
        self.POWER_MIC = POWER_MIC

class Optimization_microgrid:
    def __init__(self, microgrid, num_microgrid, C_buy, C_sell, amt):#microgrid微电网实体, num_microgrid微电网数量,

        self.num_microgrid = num_microgrid
        self.microgrid = microgrid
        self.load = microgrid.load
        self.POWER = microgrid.POWER
        self.POWER_MIC = microgrid.POWER_MIC #微电网联络线限制
        self.Ebattery = microgrid.ebattery #电网联络线限制
        self.socmin = microgrid.socmin
        self.socmax = microgrid.socmax
        self.soc0 = microgrid.soc0
        self.Pcs = microgrid.Pcs
        self.C_buy = C_buy  # 购电价格
        self.C_sell = C_sell  # 售电价格
        self.amt = amt  # 燃气轮机成本系数

    def setup_problem(self):#cplex不支持numpy直接输入，可以预处理
        try:
            problem = cplex.Cplex()
            #=========变量===============
            # 燃气机出力
            for i in range(24):
                problem.variables.add(names=[f"Pmt_{i}"], lb=[-cplex.infinity], ub=[cplex.infinity])
            # 柴油机出力
            for i in range(24):
                problem.variables.add(names=[f"Pde_{i}"], lb=[-cplex.infinity], ub=[cplex.infinity])
            # PV出力
            for i in range(24):
                problem.variables.add(names=[f"Ppv_{i}"], lb=[0], ub=[cplex.infinity])
            # WT出力
            for i in range(24):
                problem.variables.add(names=[f"Pwt_{i}"], lb=[0], ub=[cplex.infinity])
            #电池：
            #蓄电池出力
            for i in range(24):
                problem.variables.add(names=[f"Pbat_{i}"], lb=[-cplex.infinity], ub=[cplex.infinity])
            # 充放电
            for i in range(24):
                problem.variables.add(names=[f"Pcha_{i}"], lb=[0], ub=[cplex.infinity])
            for i in range(24):
                problem.variables.add(names=[f"Pdis_{i}"], lb=[-cplex.infinity], ub=[0])
            for i in range(24):
                problem.variables.add(names=[f"Temp_cha_{i}"], types=["B"])
            for i in range(24):
                problem.variables.add(names=[f"Temp_dis_{i}"], types=["B"])
            for i in range(24):
                problem.variables.add(names=[f"Temp_static_{i}"], types=["B"])
            #电网
            # 添加电网交换功率变量 (Pnet)
            for i in range(24):
                problem.variables.add(names=[f"Pnet_{i}"], lb=[-cplex.infinity], ub=[cplex.infinity])
            # 添加从电网购电量的连续变量 (Pbuy)
            for i in range(24):
                problem.variables.add(names=[f"Pbuy_{i}"], lb=[0], ub=[cplex.infinity])
            # 添加向电网售电量的连续变量 (Psell)
            for i in range(24):
                problem.variables.add(names=[f"Psell_{i}"], lb=[-cplex.infinity], ub=[0])
            # 添加购/售电标志的二进制变量 (Temp_net)
            for i in range(24):
                problem.variables.add(names=[f"Temp_net_{i}"], types=["B"])

            # 添加微电网群交易相关的连续变量
            for i in range(24):
                for j in range(int(self.num_microgrid)):
                    problem.variables.add(names=[f"Pnetmic_{j}_{i}"], lb=[-cplex.infinity], ub=[cplex.infinity])
                    problem.variables.add(names=[f"Pbuymic_{j}_{i}"], lb=[0], ub=[cplex.infinity]) #电网买进
                    problem.variables.add(names=[f"Psellmic_{j}_{i}"], lb=[-cplex.infinity], ub=[0]) #电网卖出

            # 添加微电网群交易的购/售电标志二进制变量 (Temp_net2_13, Temp_net3_12)
            for i in range(24):
                for j in range(int(self.num_microgrid)):
                    problem.variables.add(names=[f"Tempmic_{j}_{i}"], types=["B"])
            # =========约束===============
            # 添加约束
            # 主网功率交换约束
            for k in range(24):
                problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[f"Pnet_{k}"], val=[1.0])],
                    senses=["L"],
                    rhs=[self.POWER]
                )
                problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[f"Pnet_{k}"], val=[1.0])],
                    senses=["G"],
                    rhs=[-self.POWER]
                )
            # 微电网网功率交换约束
            for l in range(int(self.num_microgrid)):
                for k in range(24):
                    problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=[f"Pnetmic_{l}_{k}"], val=[1.0])],
                        senses=["L"],#约束的类型L 小于等于。
                        rhs=[self.POWER_MIC]#约束的右侧值，意味着 Pnet_k <= self.POWER
                    )
                    problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=[f"Pnetmic_{l}_{k}"], val=[1.0])],
                        senses=["G"],
                        rhs=[-self.POWER_MIC]
                    )
            # 添加功率平衡约束 电网+风电+光伏 = 负载 + 电池充电
            for k in range(24):
                # 定义功率平衡相关的所有变量名称
                var_names = [f"Pnet_{k}", f"Pde_{k}", f"Pmt_{k}", f"Pw_{k}", f"Ppv_{k}", f"Pbat_{k}"]

                # 为每个相邻的微电网添加交换功率变量
                for j in range(int(self.num_microgrid - 1)):  # 其他微电网的数量
                    var_names.append(f"Pnet1_23_{j}_{k}")

                # 所有变量的系数都是 1，除了电池出力 Pbat_k，其系数为 -1
                coefficients = [1.0] * len(var_names)
                coefficients[5] = -1.0  # Pbat_k 的系数设置为 -1，假设它在列表中的位置固定

                # 创建并添加功率平衡约束
                problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=var_names, val=coefficients)],
                    senses=["E"],
                    rhs=[self.load[k]]  # 此处仅包含负载，因为 Pbat 已经在约束的左侧
                )
            #购售电
            for k in range(24):
                for k in range(24):
                    # 当 Temp_net 等于 1 时，添加以下约束：
                    # 如果 Temp_net_k 为 1，则 Pnet_k 必须大于等于 0
                    problem.indicator_constraints.add(
                        indvar=f"Temp_net_{k}",  # 指示变量 Temp_net_k
                        complemented=0,  # 指示变量不取反
                        rhs=0.0,  # 右侧值为 0
                        sense="G",  # 大于等于
                        lin_expr=cplex.SparsePair(ind=[f"Pnet_{k}"], val=[1.0])  # 线性表达式：Pnet_k >= 0
                    )

                    # 当 Temp_net_k 为 1 时，添加 Pbuy_k = Pnet_k 的约束
                    problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=[f"Pbuy_{k}", f"Pnet_{k}"], val=[1.0, -1.0])],
                        senses=["E"],
                        rhs=[0]  # Pbuy_k - Pnet_k = 0
                    )

                    # 当 Temp_net_k 为 1 时，添加 Psell_k = 0 的约束
                    problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=[f"Psell_{k}"], val=[1.0])],
                        senses=["E"],
                        rhs=[0]  #Psell_k = 0
                    )

                    # 当 Temp_net 等于 0 时，添加以下约束：
                    # 如果 Temp_net_k 为 0，则 Pnet_k 必须小于等于 0
                    problem.indicator_constraints.add(
                        indvar=f"Temp_net_{k}",  # 指示变量 Temp_net_k
                        complemented=1,  # 指示变量取反
                        rhs=0.0,  # 右侧值为 0
                        sense="L",
                        lin_expr=cplex.SparsePair(ind=[f"Pnet_{k}"], val=[1.0])  # 线性表达式：Pnet_k <= 0
                    )

                    # 当 Temp_net_k 为 0 时，添加 Psell_k = Pnet_k 的约束
                    problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=[f"Psell_{k}", f"Pnet_{k}"], val=[1.0, -1.0])],
                        senses=["E"],
                        rhs=[0]  # 右侧值为 0，表示 Psell_k - Pnet_k = 0，即 Psell_k = Pnet_k
                    )

                    # 当 Temp_net_k 为 0 时，添加 Pbuy_k = 0 的约束
                    problem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=[f"Pbuy_{k}"], val=[1.0])],
                    senses=["E"], # 约束类型为 'E' (Equal, 即等于)
                    rhs=[0] # 右侧值为 0，表示 Pbuy_k = 0
                    )

            # 添加蓄电池充放电约束
            for k in range(24):
                # 电池充放电功率约束
                problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[f"Pbat_{k}"], val=[1.0])],
                    senses=["L"], rhs=[self.Pcs]
                )
                problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[f"Pbat_{k}"], val=[1.0])],
                    senses=["G"], rhs=[-self.Pcs]
                )
                # 为 Pcha 和 Pdis 添加类似的约束

                # 充电情况的指示约束
                problem.indicator_constraints.add(
                    indvar=f"Temp_cha_{k}", complemented=0, rhs=0.0, sense="G",
                    lin_expr=cplex.SparsePair(ind=[f"Pbat_{k}"], val=[1.0])
                )
                # 添加 Pcha = Pbat, Pdis = 0 的线性约束

                # 放电情况的指示约束
                problem.indicator_constraints.add(
                    indvar=f"Temp_dis_{k}", complemented=0, rhs=0.0, sense="L",
                    lin_expr=cplex.SparsePair(ind=[f"Pbat_{k}"], val=[1.0])
                )
                # 添加 Pdis = Pbat, Pcha = 0 的线性约束

                # 静置情况的指示约束
                problem.indicator_constraints.add(
                    indvar=f"Temp_static_{k}", complemented=0, rhs=0.0, sense="E",
                    lin_expr=cplex.SparsePair(ind=[f"Pbat_{k}"], val=[1.0])
                )
                # 添加 Pdis = 0, Pcha = 0 的线性约束

                # 确保在任何时刻只能处于充电、放电或静置状态之一
                problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[f"Temp_cha_{k}", f"Temp_dis_{k}", f"Temp_static_{k}"],
                                               val=[1.0, 1.0, 1.0])],
                    senses=["E"], rhs=[1]
                )

            # SOC 约束
            cumulative_soc = self.soc0 * self.Ebattery
            for k in range(24):
                cumulative_soc += f"Pcha_{k}" - f"Pdis_{k}"  # 更新累积 SOC

                # 确保每个时刻的 SOC 在范围内
                problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[f"Pcha_{k}", f"Pdis_{k}"], val=[self.Ebattery, -self.Ebattery])],
                    senses=["L"],
                    rhs=[self.Ebattery * self.socmax - cumulative_soc]
                )
                problem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[f"Pcha_{k}", f"Pdis_{k}"], val=[self.Ebattery, -self.Ebattery])],
                    senses=["G"],
                    rhs=[self.Ebattery * self.socmin - cumulative_soc]
                )
            # =========目标===============
            # 设置目标函数
            objective_terms = []
            for k in range(24):
                # 燃气轮机成本
                objective_terms.append((f"Pde_{k}", self.amt))
                # 电池成本
                objective_terms.append((f"Pbat_{k}", 0.339))  # abs(Pbat) * 0.339
                # 购售电成本
                objective_terms.append((f"Pbuy_{k}", self.C_buy))
                objective_terms.append((f"Psell_{k}", -self.C_sell))  # 注意：售电可能是负成本

            problem.objective.set_linear(objective_terms)
            problem.objective.set_sense(problem.objective.sense.minimize)

            return problem

        except CplexError as exc:
            print(exc)
            return None


    def solve(self):
        problem = self.setup_problem()
        problem.solve()

        # 提取和处理结果

        def solve(self):
            problem = self.setup_problem()
            problem.solve()
            # 提取和处理结果