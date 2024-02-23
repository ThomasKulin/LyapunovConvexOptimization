from Veril import systems
from Veril import plants_via_RNN
import numpy as np

import sys
sys.path.append(
    "/Users/shenshen/drake-build/install/lib/python3.7/site-packages")
from scipy.linalg import solve_lyapunov, solve_discrete_lyapunov
import pydrake
import numpy as np
from numpy.linalg import eig, inv
from pydrake.all import (MathematicalProgram, Polynomial,
                         Expression, SolutionResult,
                         Variables, Solve, Jacobian, Evaluate,
                         RealContinuousLyapunovEquation, Substitute,
                         MosekSolver)

options = {
    'plant_name': 'DoubleIntegrator',
    # 'plant_name': 'Pendulum',
    # 'plant_name': 'Satellite',
    'num_units': 4,
    'timesteps': 20,
    'num_samples': 100,
    'batch_size': 1,
    'epochs': 3,
    'dt': 1e-3,
    'obs_idx': None,
    'tag': 'fortestonly',
}

NN = systems.get_NNorCL(NNorCL='NN', **options)
[CL, model_file_name] = systems.get_NNorCL(**options)

test_time_steps = 1
test_num_samples = 2

plant = plants_via_RNN.get(CL.plant_name, CL.dt, CL.obs_idx)
augedSys = systems.PolyRNNCL(CL, model_file_name)
x = augedSys.sample_init_states_w_tanh(test_num_samples)
initx=x[:,0:2]
initc=x[:,2:6]
ext_in = np.zeros((test_num_samples, test_time_steps,1))


def test_call_CLsys(NN, CL, x):
    predicted = NN.predict([initx,initc,ext_in])
    print('predicted')
    print(predicted)
    rollout=systems.batchSim(CL, test_time_steps, init=
        [initx,initc],
       num_samples = test_num_samples)
    print('sim roll out')
    print(rollout)

test_call_CLsys(NN, CL, x)

s,f = augedSys.statesAndDynamics()
_,  batch = augedSys.statesAndDynamics(numericals=x)

for i in range(test_num_samples):
    env = dict(zip(s,x[i].T))
    numerical_f = np.array([i.Evaluate(env) for i in f])
    next_x = x[i].T+numerical_f*options['dt']
    print('CT symbolic dyanmics insert')
    print(next_x)