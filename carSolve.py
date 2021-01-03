import numpy as np
from pyomo.environ import * 
from pyomo.dae import *

m = ConcreteModel()
cos = 0.985 # 10度
sin = 0.174
T = 60

m.mu = Param(initialize=0.0040) # 摩擦系数 
m.L = Param(initialize=100.0) # 最后的位置
m.C = Param(initialize=0.35) # 空气阻力系数
m.S = Param(initialize=2.26) # 迎风面积
m.g = Param(initialize=9.8)
m.m = Param(initialize=1560) # 1.5吨
m.cosTheta = Param(initialize=cos)
m.sinTheta = Param(initialize=sin)
m.delta = Param(initialize=1.2) # 加速阻力的旋转质量换算系数
m.rho = Param(initialize=1.22558) # 空气密度

m.tau = ContinuousSet(bounds=(0,1)) #Unscaled time 
m.time = Var(m.tau) #Scaled time 
m.tf = Var(bounds=(0, T))
m.x = Var(m.tau,bounds=(0,None)) 
m.v = Var(m.tau,bounds=(0,None))
m.a = Var(m.tau,bounds=(-3.0,1.0),initialize=0)
m.F = Var(m.tau)
m.dtime = DerivativeVar(m.time) 
m.dx = DerivativeVar(m.x) 
m.dv = DerivativeVar(m.v)


def _intJ(m, t):
    return m.F[t] * m.v[t]
m.intJ = Integral(m.tau, wrt=m.tau, rule=_intJ)

def _odel(m,t):
    if t == 0:
        return Constraint.Skip 
    return m.dx[t] == m.tf * m.v[t] 
m.odel = Constraint(m.tau, rule=_odel)

def _ode2(m,t):
    if t == 0 :
        return Constraint.Skip 
    return m.dv[t] == m.tf * m.a[t]
m.ode2 = Constraint(m.tau, rule=_ode2)

def _ode3(m,t):
    if t == 0 :
        return Constraint.Skip
    return m.dtime[t] == m.tf
m.ode3 = Constraint(m.tau, rule=_ode3)

def _ode4(m,t):
    if t == 0:
        return Constraint.Skip 
    return m.F[t] == m.C * m.rho * m.S * m.v[t]**2 * 0.5 + \
                    (m.cosTheta * m.mu * m.g + \
                    m.sinTheta * m.g + \
                    m.delta * m.a[t]) * m.m
m.ode4 = Constraint(m.tau, rule=_ode4)

def _init(m):
    yield m.x[0] == 0 
    yield m.x[1] == m.L 
    yield m.v[0] == 0 
    yield m.v[1] == 0 
    yield m.time[0] == 0 
m.initcon = ConstraintList(rule=_init)
m.obj = Objective(expr=m.intJ)

discretizer = TransformationFactory('dae.finite_difference')
discretizer.apply_to(m,nfe=15, scheme='BACKWARD')

solver = SolverFactory('ipopt') 
solver.solve(m, tee=False)

print("Spent energy= %6.2f" %(value(m.intJ)))
print("Spent tiem= %6.2f" %(value(m.tf)))

import matplotlib.pyplot as plt

x=[] 
v=[] 
a=[] 
F=[]
time=[]

for i in m.tau:

    time.append(value(m.time[i])) 
    x.append(value(m.x[i])) 
    v.append(value(m.v[i])) 
    a.append(value(m.a[i]))
    F.append(value(m.F[i]))

plt.plot(time,x,label='x') 
plt.title('location') 
plt.xlabel('time') 
plt.show()

plt.plot(time,v,label='v') 
plt.title('velocity') 
plt.xlabel('time') 
plt.show()

plt.plot(time,a,label='a') 
plt.title('acceleration') 
plt.xlabel('time') 
plt.show()

