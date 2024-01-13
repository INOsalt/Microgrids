import cplex
prob = cplex.Cplex()
print(prob.get_version())
from cplex.exceptions import CplexError

try:
    prob = cplex.Cplex()
    prob.objective.set_sense(prob.objective.sense.maximize)

    prob.variables.add(obj=[1.0], lb=[0.0])

    prob.linear_constraints.add(
        lin_expr=[[[0], [1.0]]],
        senses=["L"],
        rhs=[10.0]
    )

    prob.solve()
    print("Solution value  = ", prob.solution.get_objective_value())
    print("Solution variables = ", prob.solution.get_values())
except CplexError as exc:
    print(exc)
