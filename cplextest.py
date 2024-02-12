# import cplex
# prob = cplex.Cplex()
# print(prob.get_version())
# from cplex.exceptions import CplexError
#
# try:
#     prob = cplex.Cplex()
#     prob.objective.set_sense(prob.objective.sense.maximize)
#
#     prob.variables.add(obj=[1.0], lb=[0.0])
#
#     prob.linear_constraints.add(
#         lin_expr=[[[0], [1.0]]],
#         senses=["L"],
#         rhs=[10.0]
#     )
#
#     prob.solve()
#     print("Solution value  = ", prob.solution.get_objective_value())
#     print("Solution variables = ", prob.solution.get_values())
# except CplexError as exc:
#     print(exc)
#


from pyomo.environ import *
from pyomo.opt import SolverFactory

# 创建一个模型
model = ConcreteModel()

# 定义变量
model.x = Var(within=Reals)
model.y = Var(within=Reals)

# 定义目标函数
model.obj = Objective(expr=(model.x-1)**2 + (model.y-2)**2, sense=minimize)

# 定义约束
model.constraint = Constraint(expr=model.x**2 + model.y**2 <= 1)

# 创建一个求解器实例，指定Ipopt的路径
solver = SolverFactory('ipopt', executable='D:\\Ipopt-3.14.14-win64-msvs2019-md\\bin\\ipopt.exe')

# 解决问题
solution = solver.solve(model, tee=True)

# 打印解决方案
x_value = value(model.x)
y_value = value(model.y)
print(f"Solution: x = {x_value}, y = {y_value}")

