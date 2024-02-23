import numpy as np
from utils import load_polynomial
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    BasicVector,
    Linearize,
    LinearQuadraticRegulator,
    CommonSolverOption,
    DiagramBuilder,
    Expression,
    LeafSystem,
    LogVectorOutput,
    MathematicalProgram,
    MeshcatVisualizer,
    MosekSolver,
    Parser,
    Polynomial,
    Simulator,
    Solve,
    SolverOptions,
    StartMeshcat,
    SymbolicVectorSystem,
    Variable,
    Variables,
    WrapToSystem,
)
from pydrake.examples import PendulumParams
from scipy.integrate import quad
import sympy as sp

from underactuated import ConfigureParser

# Define new state limits for the updated system
d_theta_scale = 0.25
d_theta = d_theta_scale * np.pi
d_phi_scale = 0.25
d_phi = d_phi_scale * np.pi

x_max = np.array([1.5, 1.5, d_theta, d_phi, 20, 20, 20, 20])
x_min = np.array([-1.5, -1.5, -d_theta, -d_phi, -20, -20, -20, -20])
deg = 4

# Compute the z_max and z_min based on the updated system setup
if d_theta < np.pi / 2 and d_phi < np.pi / 2:
    z_max = np.array(
        [x_max[0], x_max[1], np.sin(x_max[2]), 1, np.sin(x_max[3]), 1, x_max[4], x_max[5],
         x_max[6], x_max[7]])
else:
    z_max = 0


prog = MathematicalProgram()
z = prog.NewIndeterminates(5, "z")

# J_star = load_polynomial(z, "SphericalIP/data/{}/{}/J_lower_bound_deg_6.pkl".format(z_max, deg))
J_star = load_polynomial(z, "data/[10. 10.  1.  1.  1.  1.  6.  6.  6.  6.]/J_lower_bound_deg_4_TP2e4_TPDOT2e2.pkl")

