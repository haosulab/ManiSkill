from pyassimp import *
scene = load('F2.dae')

print('loading the scene')
release(scene)

