from pyomo.environ import *
from pyomo.dae import *
import numpy as np

m = ConcreteModel()

m.pi = Param(initialize = np.pi)
m.t  = ContinuousSet(bounds = (0,2))
m.x  = ContinuousSet(bounds = (0,1))
m.u  = Var(m.x, m.t)

m.dudx  = DerivativeVar(m.u, wrt = m.x)
m.dudx2 = DerivativeVar(m.u, wrt = (m.x, m.x))
m.dudt  = DerivativeVar(m.u, wrt = m.t)


def _pde(m,i,j):
	if i == 0 or i  == 1 or j ==0:
		return Constraint.Skip

	return m.pi**2*m.dudt[i,j] == m.dudx2[i,j]
m.pde = Constraint(m.x, m.t, rule = _pde )

def _initcon(m,i):
	if i == 0 or i == 1:
		return Constraint.Skip
	return m.u[i,0] == sin(m.pi*i)
m.initcon = Constraint(m.x, rule = _initcon)

def _lowerbound(m,j):
	return m.u[0,j] == 0
m.lowerbound = Constraint(m.t, rule = _lowerbound)

def _upperbound(m,j):
	return m.pi*exp(-j) + m.dudx[1,j] == 0

m._upperbound = Constraint(m.t, rule = _upperbound)

m.obj = Objective(expr = 1)

discretizer  = TransformationFactory('dae.finite_difference')
discretizer2 = TransformationFactory('dae.collocation')

discretizer.apply_to(m, nfe = 25, wrt = m.x, scheme = 'BACKWARD')
discretizer2.apply_to(m, nfe = 20, ncp = 3, wrt = m.t)

solver = SolverFactory('ipopt')
results = solver.solve(m, tee=True) 

