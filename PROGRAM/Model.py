import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pint
import espressomd

espressomd.assert_features(['WCA', 'ELECTROSTATICS'])
import espressomd.electrostatics
import espressomd.reaction_methods
import espressomd.polymer

from espressomd.interactions import HarmonicBond
from espressomd.interactions import FeneBond
from espressomd.interactions import AngleHarmonic
from espressomd.interactions import Dihedral
from espressomd import observables, accumulators, analyze

from property_test_class import Properties


class Model(Properties):
    def __init__(self, input_dict):
        super().__init__(input_dict)

    def _system_definition(self):
        self.system = espressomd.System(box_l=[self.box_length_reduced_units] * 3)
        self.system.time_step = self.time_step
        self.system.cell_system.skin = 2.0
        self.system.periodicity = [True, True, True]
        np.random.seed(seed=10)  # initialize the random number generator in numpy

    def _bonded_interactions(self):
        if self.USE_FENE:
            self.bond_pot = FeneBond(k=self.k_bond_reduced_units, d_r_max=4.0, r_0=1.5)
        else :
            bond_pot = HarmonicBond(k=self.k_bond_reduced_units, r_0=1.5)

        if self.USE_BENDING:
            self.bend=AngleHarmonic(bend=self.k_bending_reduced_units, phi0=np.pi*(2.0/3.0))
            system.bonded_inter.add(self.bend)

        if self.USE_DIHEDRAL_POT:
            self.dihedral = Dihedral(bend=10.0, mult=3, phase=np.pi*(2.0/3.0))
            system.bonded_inter.add(dihedral)

    def _non_bonded_interactions(self):
        if self.USE_WCA:
            for type_1, type_2 in ((x, y) for x in self.type_particles.values() for y in self.type_particles.values()):
                lj_sig = combination_rule_sigma("Berthelot", self.sigma_reduced_units[str(type_1)], self.sigma_reduced_units[str(type_2)])
                lj_eps = combination_rule_epsilon("Lorentz", self.epsilon_reduced_units[str(type_1)], self.epsilon_reduced_units[str(type_2)])
                system.non_bonded_inter[type_1, type_2].wca.set_params(epsilon=lj_eps, sigma=0.5*lj_sig)

    def _long_range_interactions(self):
        if self.USE_ELECTROSTATICS:
            if self.USE_P3M:
                self.coulomb = espressomd.electrostatics.P3M(prefactor = (self.bjerrum_length * self.KT / (self.ureg.elementary_charge ** 2)
                           ).to("sim_length * sim_energy / sim_charge^2").magnitude,
                                                        accuracy=1e-4)
            else:
                self.coulomb = espressomd.electrostatics.DH(prefactor = (self.bjerrum_length * self.KT / (self.ureg.elementary_charge ** 2)
                           ).to("sim_length * sim_energy / sim_charge^2").magnitude,
                                                       kappa = self.kappa_reduced_units,
                                                       r_cut = 1. / self.kappa_reduced_units)

            self.system.actors.add(self.coulomb)
        else:
            # this speeds up the simulation of dilute systems with small particle numbers
            self.system.cell_system.set_n_square()


    def _polymer_definition(self):
        self.polymers = espressomd.polymer.linear_polymer_positions(n_polymers=self.n_poly, beads_per_chain=self.n_mon,bond_length=self.bond_lenth_reduced_units,min_distance=0.9, seed=23)
        # add the polymer particles composed of ionizable acid groups, initially in the ionized state
        n_NH2=0
        n_NH=0
        for polymer in self.polymers:
            for index,position in enumerate(polymer):
                id = len(self.system.part)
                if index % 3 == 0 :
                    if index == 0 or index == (self.n_mon-1) :
                        self.system.part.add(id = id ,pos=position, type=self.type_particles["A2"], q=self.charge_reduced_units["A2"])
                        n_NH2 = n_NH2+1
                        # print(index,"A2")
                    else :
                        self.system.part.add(id = id ,pos=position, type=self.type_particles["A"], q=self.charge_reduced_units["A"])
                        n_NH = n_NH+1
                        # print(index,"A")
                else :
                    self.system.part.add(id = id ,pos=position, type=self.type_particles["N"], q=self.charge_reduced_units["N"])
                    # print(index,"N")

                #p = system.part.add(pos=position, type=TYPES["A"], q=CHARGES["A"])
                if index>0:
                    system.part.by_id(id).add_bond((bond_pot, id -1))
                    if USE_BENDING:
                        if index > 0 and index < len(polymer) -1:
                            self.system.part.by_id(id).add_bond((bend,id -1, id + 1))  # Ja es crearà la seg:uent partícula
                    if USE_DIHEDRAL_POT:
                        if index > 0 and index < len(polymer) -2:
                            self.system.part.by_id(id).add_bond((dihedral, id-1, id+1, id+2))

    def _add_reactions(self):

        exclusion_radius = combination_rule_sigma("Berthelot", lj_sigmas["1"], lj_sigmas["2"]) if USE_WCA else 0.0
        RE = espressomd.reaction_methods.ConstantpHEnsemble(
            kT=KT.to("sim_energy").magnitude,
            exclusion_range=exclusion_radius,
            seed=77,
            constant_pH=4.0
        )

        exclusion_radius2 = combination_rule_sigma("Berthelot", lj_sigmas["6"], lj_sigmas["2"]) if USE_WCA else 0.0
        RE2 = espressomd.reaction_methods.ConstantpHEnsemble(
            kT=KT.to("sim_energy").magnitude,
            exclusion_range=exclusion_radius2,
            seed=77,
            constant_pH=4.0
        )

        RE.add_reaction(
            gamma=K,
            reactant_types=[TYPES["HA"]],
            reactant_coefficients=[1],
            product_types=[TYPES["A"], TYPES["B"]],
            product_coefficients=[1, 1],
            default_charges={TYPES["HA"]: CHARGES["HA"],
                             TYPES["A"]: CHARGES["A"],
                             TYPES["B"]: CHARGES["B"]}
        )

        RE.set_non_interacting_type(type=len(TYPES)) # this parameter helps speed up the calculation in an interacting system
        RE2.add_reaction(
            gamma=K2,
            reactant_types=[TYPES["HA2"]],
            reactant_coefficients=[1],
            product_types=[TYPES["A2"], TYPES["B"]],
            product_coefficients=[1, 1],
            default_charges={TYPES["HA2"]: CHARGES["HA2"],
                             TYPES["A2"]: CHARGES["A2"],
                             TYPES["B"]: CHARGES["B"]}
        )
        RE2.set_non_interacting_type(type=len(TYPES)) # this parameter helps speed up the calculation in an interacting system
