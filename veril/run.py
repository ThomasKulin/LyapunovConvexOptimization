# import time, argparse
import numpy as np
from veril.systems import get_system
# from veril.sample_lyap import get_V
from veril.symbolic_verifier import verify_via_equality, outerApproximation, verify_via_bilinear, verify_via_bilinear2, levelset_sos, global_vdot, cvx_V
from veril.sample_variety import verify_via_variety
from veril.util.plots import *

if "MOSEKLM_LICENSE_FILE" not in os.environ:
    # If a mosek.lic file has already been uploaded, then simply use it here.
    if os.path.exists('/home/thomas/mosek/mosek.lic'):
        os.environ["MOSEKLM_LICENSE_FILE"] = "/home/thomas/mosek/mosek.lic"
    else:
        print("stop bein a loooooser and get mosek")


# sys_name = 'VanderPol'
# sys_name = 'Pendubot'
# sys_name = 'CartPole_2D'
sys_name = 'SphericalIP'
deg_ftrs = 3
rm_one = True
system = get_system(sys_name, deg_ftrs, 0, rm_one)

# Estimate stability boundaries
xaxis = [-np.pi/2, np.pi/2]
yaxis = [-2*np.pi, 2*np.pi]
slice_idx = (2,6)
# slice_idx = (0,2)
# stableSamples = system.sample_stable_inits(xaxis=xaxis, yaxis=yaxis, num_grid=40, slice_idx=slice_idx)
# plot_samples(system, xaxis=xaxis, yaxis=yaxis, slice_idx=slice_idx)
# stableSamples = system.sample_stable_inits_concurrent2(xaxis=xaxis, yaxis=yaxis, num_grid=10, slice_idx=slice_idx)
# plot_samples(system, xaxis=xaxis, yaxis=yaxis, slice_idx=slice_idx)

#Plot various trajectories
# initial = system.random_sample(20)
# print(initial)
# [final_states, final_Vs] = plot_traj(initial, system, int_horizon=30)
# print([system.is_at_fixed_pt(i) for i in final_states])


# attempt to feed in the Taylor expansion of J_star to solve?
x = system.sym_x
# V_tay_o3 = (6.8212102632969618e-13 - 1.5578987324818581e-05 * (x[0] * x[2]) - 0.0051738052507164269 * (x[0] * x[4]) + 2.1012013008897326e-05 * (x[0] * x[6]) + 4.0248154437782385e-05 * (x[1] * x[3]) - 0.0057826680614227952 * (x[1] * x[5]) - 5.7420851715652574e-06 * (x[1] * x[7]) + 0.0033496703145452864 * (x[2] * x[3]) + 3.8802691752094895e-06 * (x[2] * x[4]) + 28.454393006347523 * (x[2] * x[6]) - 8.6867293007904852e-05 * (x[3] * x[5]) + 29.576812631171496 * (x[3] * x[7]) - 1.2917540354345276e-06 * (x[4] * x[5]) + 0.27181865041636738 * (x[4] * x[6]) + 0.1566114348590264 * (x[5] * x[7]) + 3.2546682432677772e-06 * (x[6] * x[7]) + ((0.086865554029940908 * pow(x[0], 2)) / 2) + ((0.099904557816220332 * pow(x[1], 2)) / 2) + ((0.24675173678367993 * pow(x[5], 2)) / 2) + ((0.37634745407135395 * pow(x[4], 2)) / 2) + ((1.2940788668949346 * pow(x[7], 2)) / 2) + ((1.6007923075105379 * pow(x[6], 2)) / 2) + ((659.18539969510971 * pow(x[2], 2)) / 2) + ((1081.8083589271992 * pow(x[3], 2)) / 2))
# V_tay_removexy = (6.8212102632969618e-13 - 1.5578987324818581e-05 * (0 * x[0]) - 0.0051738052507164269 * (0 * x[2]) + 2.1012013008897326e-05 * (0 * x[4]) + 4.0248154437782385e-05 * (0 * x[1]) - 0.0057826680614227952 * (0 * x[3]) - 5.7420851715652574e-06 * (0 * x[5]) + 0.0033496703145452864 * (x[0] * x[1]) + 3.8802691752094895e-06 * (x[0] * x[2]) + 28.454393006347523 * (x[0] * x[4]) - 8.6867293007904852e-05 * (x[1] * x[3]) + 29.576812631171496 * (x[1] * x[5]) - 1.2917540354345276e-06 * (x[2] * x[3]) + 0.27181865041636738 * (x[2] * x[4]) + 0.1566114348590264 * (x[3] * x[5]) + 3.2546682432677772e-06 * (x[4] * x[5]) + ((0.086865554029940908 * pow(0, 2)) / 2) + ((0.099904557816220332 * pow(0, 2)) / 2) + ((0.24675173678367993 * pow(x[3], 2)) / 2) + ((0.37634745407135395 * pow(x[2], 2)) / 2) + ((1.2940788668949346 * pow(x[5], 2)) / 2) + ((1.6007923075105379 * pow(x[4], 2)) / 2) + ((659.18539969510971 * pow(x[0], 2)) / 2) + ((1081.8083589271992 * pow(x[1], 2)) / 2))
# V_tay_removexydot = (6.8212102632969618e-13 - 1.5578987324818581e-05 * (0 * x[0]) - 0.0051738052507164269 * (0 * 0) + 2.1012013008897326e-05 * (0 * x[2]) + 4.0248154437782385e-05 * (0 * x[1]) - 0.0057826680614227952 * (0 * 0) - 5.7420851715652574e-06 * (0 * x[3]) + 0.0033496703145452864 * (x[0] * x[1]) + 3.8802691752094895e-06 * (x[0] * 0) + 28.454393006347523 * (x[0] * x[2]) - 8.6867293007904852e-05 * (x[1] * 0) + 29.576812631171496 * (x[1] * x[3]) - 1.2917540354345276e-06 * (0 * 0) + 0.27181865041636738 * (0 * x[2]) + 0.1566114348590264 * (0 * x[3]) + 3.2546682432677772e-06 * (x[2] * x[3]) + ((0.086865554029940908 * pow(0, 2)) / 2) + ((0.099904557816220332 * pow(0, 2)) / 2) + ((0.24675173678367993 * pow(0, 2)) / 2) + ((0.37634745407135395 * pow(0, 2)) / 2) + ((1.2940788668949346 * pow(x[3], 2)) / 2) + ((1.6007923075105379 * pow(x[2], 2)) / 2) + ((659.18539969510971 * pow(x[0], 2)) / 2) + ((1081.8083589271992 * pow(x[1], 2)) / 2))

# system, V_bilinear = verify_via_bilinear2(system, V=None)
system, B = outerApproximation(system)
# V_cvx, Vdot, system = cvx_V(system, sys_name)
# V_equaliy = verify_via_equality(system, None)#V_tay_removexy)
success = False
while ~success:
    try:
        # rho_ring = verify_via_variety(system, None,  init_root_threads=285, num_samples=1)
        success=True
    except:
        print("giving it another go")

print('done')
