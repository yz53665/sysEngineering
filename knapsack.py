from pyomo.environ import *

items = ['hammer', 'wrench', 'screwdriver', 'towel']
weight = {'hammer': 5, 'wrench': 7, 'screwdriver': 4, 'towel': 3}
value = {'hammer': 8,}

model = ConcreteModel()
