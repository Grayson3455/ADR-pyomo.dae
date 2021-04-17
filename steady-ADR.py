# 1: this script solves the steady-advection-diffusion-reaction equations using the pyomo.dae

# 2: numerical examples are chosen from the paper: Edge stabilization for Galerkin 
# approximations of convection–diffusion–reaction problems, Burman et.al

# 3: The code structure refers to the paper: pyomo.dae: a modeling and automatic discretization
# framework for optimization with differential and algebraic equations, Nicholson et al.

from pyomo.environ import *
from pyomo.dae import *
import numpy as np
from matplotlib import pyplot as plt


# space-time variables
m      = ConcreteModel()
m.x    = ContinuousSet(bounds = (0,1))
m.y    = ContinuousSet(bounds = (0,1))
m.phi  = Var(m.x, m.y)


# define derivative variables
m.DphiDx   = DerivativeVar(m.phi, wrt = m.x)
m.DphiDy   = DerivativeVar(m.phi, wrt = m.y)
m.DphiDx_2 = DerivativeVar(m.phi, wrt = (m.x, m.x))
m.DphiDy_2 = DerivativeVar(m.phi, wrt = (m.y, m.y))

# define PDE parameters
m.c 	= Param(initialize = 1.0)  # the reaction constant
m.nu    = Param(initialize = 1e-5) # the diffusivity
m.bx    = Param(initialize = 1.0)  # the advection velocity, x-component
m.by    = Param(initialize = 0.0)  # the advection velocity, y-component
m.lmd     = Param(initialize = 0.05) # the constant to control the slope of manufactured solution, 


# create lambda function for manufactured solutions
u_e    	= lambda x : 0.5* ( 1.0 - np.tanh((x - 0.5)/value(m.lmd)) ) 
du_e    = lambda x : -1.0/(2.0*value(m.lmd)) * (1.0/np.cosh((x-0.5)/value(m.lmd)))**2
du_e2   = lambda x : 1.0/value(m.lmd)/value(m.lmd) * (1.0/np.cosh((x-0.5)/value(m.lmd)))**2 * np.tanh((x - 0.5)/value(m.lmd))
f       = lambda x,y : du_e(x) - value(m.nu) * du_e2(x) + value(m.c)*u_e(x) 

# define adr pde, i: x, j: y
def ADR_pde(m,i,j):
	if i == 0 or i  == 1: # boundary conditions to be implemented, no constraints
		return Constraint.Skip
	return m.bx * m.DphiDx[i,j] + m.by * m.DphiDy[i,j]- m.nu*m.DphiDx_2[i,j] - m.nu*m.DphiDy_2[i,j]  + m.c*m.phi[i,j] == f(i,j)
m.pde = Constraint(m.x, m.y, rule = ADR_pde )   

# left boundary condition [phi = 1 @ x = 0]
def BC1(m,j):
	return m.phi[0,j] == 1.0
m.BCx_0 = Constraint(m.y, rule = BC1)

# right boundary condition [phi = 0 @ x = 1]
def BC2(m,j):
	return m.phi[1,j] == 0.0
m.BCx_1 = Constraint(m.y, rule = BC2)

# trivial obj
m.obj = Objective(expr = 1)

# discretization and solve
discretizer = TransformationFactory('dae.collocation')
discretizer.apply_to(m,nfe=9,ncp=4,wrt=m.x)
discretizer.apply_to(m,nfe=9,ncp=4,wrt=m.y)

solver = SolverFactory('ipopt')
results = solver.solve(m, tee=True) 

# extract data
x   = np.zeros(len(m.x))
y   = np.zeros(len(m.y))
phi = np.zeros( ( len(m.x), len(m.y) ) )

for i in range(len(m.x)):
	for j in range(len(m.y)):
		x[i] = value(m.x[i+1])
		y[j] = value(m.y[j+1])
		phi[i,j] = value(m.phi[x[i],y[j]])

# save the data
np.savetxt('results/x.csv', x, delimiter=',')   
np.savetxt('results/y.csv', y, delimiter=',') 
np.savetxt('results/phi.csv', phi, delimiter=',')     
