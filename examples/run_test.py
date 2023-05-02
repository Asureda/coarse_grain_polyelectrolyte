
import numpy as np
from EnsembleDynamics import EnsembleDynamics
from Observables import Observables
from utils import ideal_alpha

modelParameters = {
    "Thermal properties": {
        "Temperature": 300,
    },
    "System properties": {
        "Number of polymers": 1,
        "Concentration acid particles": 1e-2,
        "Concentration salt particles": 2*1e-2,
        "Number of monomers": 100,
        "NH monomers": 32,
        "NH2 monomers": 2,
        "bond length": 3.85,
        "bond constant": 200.0,
        "bending constant": 0.002, # Optional
        "dihedral constant": 0.002, # Optional
    },
    "Particles properties" : {
        "NH": {"index": 0, "charge": +1, "sigma": 2*2.75, "epsilon":  0.32},
        "N": {"index": 1, "charge": 0, "sigma": 2*3.298, "epsilon": 0.305},
        "B": {"index": 2, "charge": +1, "sigma": 2*2.42, "epsilon": 0.20},
        "CH2": {"index": 3, "charge": 0, "sigma": 2*3.800, "epsilon": 0.47},
        "Na": {"index": 4, "charge": +1, "sigma": 2*2.21, "epsilon": 0.45},
        "Cl": {"index": 5, "charge": -1, "sigma": 2*3.550, "epsilon": 0.43},
        "NH2": {"index": 6, "charge": +1, "sigma": 2*2.99, "epsilon": 0.38},
        "N2": {"index": 7, "charge": 0, "sigma": 2*3.684, "epsilon": 0.31},
    },

    "pH properties": {
        "pK1": 8.18,
        "pK2": 10.02,
        "NUM_PHS": 12,
        "pHmin": 4.0,
        "pHmax": 12.0,
        },
    "Montecarlo properties": {
        "N_BLOCKS": 16,
        "DESIRED_BLOCK_SIZE": 200,
        "PROB_REACTION": 0.7,
    },
    "Simulation configuration": {
        "time step": 0.001,
        "USE_WCA": True,
        "USE_ELECTROSTATICS": True,
        "USE_FENE": False,
        "USE_BENDING": True,
        "USE_DIHEDRAL_POT": True,
        "USE_P3M": False,
    },

}

ensemble = EnsembleDynamics(modelParameters)
obs = Observables()

ensemble.equilibrate_pH(10.0)
ensemble.system.integrator.run(int(1e5))

