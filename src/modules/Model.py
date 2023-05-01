import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pint
import espressomd

espressomd.assert_features(["WCA", "ELECTROSTATICS"])
import espressomd.electrostatics
import espressomd.reaction_methods
import espressomd.polymer

from espressomd.interactions import HarmonicBond
from espressomd.interactions import FeneBond
from espressomd.interactions import AngleHarmonic
from espressomd.interactions import Dihedral
import pint

from utils import combination_rule_epsilon, combination_rule_sigma, ideal_alpha
from Properties import Properties


class Model(Properties):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        self._bonded_interactions()
        self._create_polymer()
        self._add_salt_ions()
        self._non_bonded_interactions()
        self._long_range_interactions()
        self._add_reactions()

    def _bonded_interactions(self):

        if self.USE_FENE:
            self.bond_pot = FeneBond(
                k=self.k_bond_reduced_units.magnitude, d_r_max=4.0, r_0=self.bond_length_reduced_units.magnitude
            )
        else:
            self.bond_pot = HarmonicBond(k=self.k_bond_reduced_units.magnitude, r_0=self.bond_length_reduced_units.magnitude)
        self.system.bonded_inter.add(self.bond_pot)

        if self.USE_BENDING:
            self.bend = AngleHarmonic(
                bend=self.k_bending_reduced_units.magnitude, phi0=np.pi * (2.0 / 3.0)
            )
            self.system.bonded_inter.add(self.bend)

        if self.USE_DIHEDRAL_POT:
            self.dihedral = Dihedral(
                bend=self.k_dihedral_reduced_units.magitude,
                mult=3,
                phase= np.pi , #np.pi * (2.0 / 3.0),
            )
            self.system.bonded_inter.add(self.dihedral)

    def _non_bonded_interactions(self):
        if self.USE_WCA:
            for type_1, type_2 in (
                (x, y)
                for x in self.type_particles.keys()
                for y in self.type_particles.keys()
            ):
                lj_sig = combination_rule_sigma(
                    "Berthelot",
                    self.sigma_reduced_units[str(type_1)],
                    self.sigma_reduced_units[str(type_2)],
                )
                lj_eps = combination_rule_epsilon(
                    "Lorentz",
                    self.epsilon_reduced_units[str(type_1)],
                    self.epsilon_reduced_units[str(type_2)],
                )
                
                self.system.non_bonded_inter[
                    self.type_particles[type_1], self.type_particles[type_2]
                ].wca.set_params(epsilon=lj_eps, sigma=0.5 * lj_sig)
                # relax the overlaps with steepest descent
                self.system.integrator.set_steepest_descent(f_max=0, gamma=0.1, max_displacement=0.1)
                self.system.integrator.run(100)
                self.system.integrator.set_vv()  # to switch back to velocity Verlet

    def _long_range_interactions(self):

        if self.USE_ELECTROSTATICS:
            if self.USE_P3M:
                self.coulomb = espressomd.electrostatics.P3M(
                    prefactor=(
                        self.bjerrum_length
                        * self.KT
                        / (self.ureg.elementary_charge ** 2)
                    )
                    .to("sim_length * sim_energy / sim_charge^2")
                    .magnitude,
                    accuracy=1e-5,
                )
            else:
                self.coulomb = espressomd.electrostatics.DH(
                    prefactor=(
                        self.bjerrum_length
                        * self.KT
                        / (self.ureg.elementary_charge ** 2)
                    )
                    .to("sim_length * sim_energy / sim_charge^2")
                    .magnitude,
                    kappa=self.kappa_reduced_units,
                    r_cut=1.0 / self.kappa_reduced_units,
                )

            self.system.actors.add(self.coulomb)
        else:
            # this speeds up the simulation of dilute systems with small particle numbers
            self.system.cell_system.set_n_square()

    def _create_polymer(self):

        self.polymers = espressomd.polymer.linear_polymer_positions(
            n_polymers=self.n_poly,
            beads_per_chain=self.n_mon,
            bond_length=1.5,
            min_distance=0.5,
            seed=12,
        )
        # add the polymer particles composed of ionizable acid groups, initially in the ionized state
        for polymer in self.polymers:
            n_a = 0
            for index, position in enumerate(polymer):
                id = len(self.system.part)
                if index % 3 == 0:
                    if index == 0 or index == (self.n_mon - 1):
                        self.system.part.add(
                            id=id,
                            pos=position,
                            type=self.type_particles["N2"],
                            q=self.charge_reduced_units["N2"],
                        )
                        #print(id, "A2")
                        n_a += 1

                    else:
                        self.system.part.add(
                            id=id,
                            pos=position,
                            type=self.type_particles["N"],
                            q=self.charge_reduced_units["N"],
                        )
                        #print(id, "A")
                        n_a += 1

                else:
                    self.system.part.add(
                        id=id,
                        pos=position,
                        type=self.type_particles["CH2"],
                        q=self.charge_reduced_units["CH2"],
                    )
                    #print(id, "N")

                # p = system.part.add(pos=position, type=TYPES["A"], q=CHARGES["A"])
                if index > 0:
                    self.system.part.by_id(id).add_bond((self.bond_pot, id - 1))
                    if self.USE_BENDING:
                        if index > 0 and index < len(polymer) - 1:
                            self.system.part.by_id(id).add_bond(
                                (self.bend, id - 1, id + 1)
                            )  # Ja es crearà la seg:uent partícula
                    if self.USE_DIHEDRAL_POT:
                        if index > 0 and index < len(polymer) - 2:
                            self.system.part.by_id(id).add_bond(
                                (self.dihedral, id - 1, id + 1, id + 2)
                            )
            #print(n_a)

    def _add_salt_ions(self):

        self.system.part.add(
            pos=np.random.random((self.n_acid, 3)) * self.box_length_reduced_units,
            type=[self.type_particles["B"]] * self.n_acid,
            q=[self.charge_reduced_units["B"]] * self.n_acid,
        )

        self.system.part.add(
            pos=np.random.random((self.n_acid, 3)) * self.box_length_reduced_units,
            type=[self.type_particles["Cl"]] * self.n_acid,
            q=[self.charge_reduced_units["Cl"]] * self.n_acid,
        )
        #if self.USE_ELECTROSTATICS:
            #if self.USE_P3M:
        self.system.part.add(
            pos=np.random.random((self.n_salt, 3)) * self.box_length_reduced_units,
            type=[self.type_particles["Na"]] * self.n_salt,
            q=[self.charge_reduced_units["Na"]] * self.n_salt,
        )

        self.system.part.add(
            pos=np.random.random((self.n_salt, 3)) * self.box_length_reduced_units,
            type=[self.type_particles["Cl"]] * self.n_salt,
            q=[self.charge_reduced_units["Cl"]] * self.n_salt,
        )

    def _add_reactions(self):

        self.exclusion_radius_reaction1 = combination_rule_sigma(
                    "Berthelot",
                    self.sigma_reduced_units["N"],
                    self.sigma_reduced_units["B"],
                )
        # self.exclusion_radius_reaction1 = (0.5*self.bjerrum_length.to('sim_length').magnitude
        #     if self.USE_WCA
        #     else 0.0
        # )
        self.reaction1 = espressomd.reaction_methods.ConstantpHEnsemble(
            kT=self.KT.to("sim_energy").magnitude,
            exclusion_range=self.exclusion_radius_reaction1,
            seed=77,
            constant_pH=4.0,
        )

        self.exclusion_radius_reaction2 = combination_rule_sigma(
                    "Berthelot",
                    self.sigma_reduced_units["N2"],
                    self.sigma_reduced_units["B"],
                )

        # self.exclusion_radius_reaction2 = (0.5*self.bjerrum_length.to('sim_length').magnitude
        #     if self.USE_WCA
        #     else 0.0
        # )
        self.reaction2 = espressomd.reaction_methods.ConstantpHEnsemble(
            kT=self.KT.to("sim_energy").magnitude,
            exclusion_range=self.exclusion_radius_reaction2,
            seed=77,
            constant_pH=4.0,
        )

        self.reaction1.add_reaction(
            gamma=10 ** (-self.pK),
            reactant_types=[self.type_particles["NH"]],
            product_types=[self.type_particles["N"], self.type_particles["B"]],
            default_charges={
                self.type_particles["NH"]: self.charge_reduced_units["NH"],
                self.type_particles["N"]: self.charge_reduced_units["N"],
                self.type_particles["B"]: self.charge_reduced_units["B"],
            },
        )

        self.reaction2.add_reaction(
            gamma=10 ** (-self.pK2),
            reactant_types=[self.type_particles["NH2"]],
            product_types=[self.type_particles["N2"], self.type_particles["B"]],
            default_charges={
                self.type_particles["NH2"]: self.charge_reduced_units["NH2"],
                self.type_particles["N2"]: self.charge_reduced_units["N2"],
                self.type_particles["B"]: self.charge_reduced_units["B"],
            },
        )
        self.reaction1.set_non_interacting_type(
            type=len(self.type_particles)
        )  # this parameter helps speed up the calculation in an interacting system
        self.reaction2.set_non_interacting_type(
            type=len(self.type_particles)
        )  # this parameter helps speed up the calculation in an interacting system
