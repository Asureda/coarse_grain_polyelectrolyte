import numpy as np
import pint
import espressomd
import pandas as pd
import time
espressomd.assert_features(["WCA", "ELECTROSTATICS"])
import espressomd.electrostatics
import espressomd.reaction_methods
import espressomd.polymer

from espressomd.interactions import HarmonicBond
from espressomd.interactions import FeneBond
from espressomd.interactions import AngleHarmonic
from espressomd.interactions import Dihedral
from espressomd import observables, accumulators, analyze
from Model import Model


class EnsembleDynamics(Model):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        if self.USE_WCA:
            self._warm_up_system()

    def _warm_up_system(self):
        print("Warm up the system")

        warm_steps = 25
        warm_n_times = 100
        min_dist = 0.8
        self.system.time = 0.
        wca_cap = 5
        self.system.force_cap = wca_cap
        i = 0
        act_min_dist = self.system.analysis.min_dist()
        self.system.thermostat.set_langevin(kT=0.0, gamma=1.0, seed=4)
        e_kin = np.zeros(warm_steps + wca_cap)

        # warmup with zero temperature to remove overlaps
        while act_min_dist < min_dist:
            for j in range(warm_steps + wca_cap):
                self.system.integrator.run(int(0.5 * warm_steps))
                energy = self.system.analysis.energy()
            #    system.integrator.run(warm_steps + wca_cap)
            # Warmup criterion
            act_min_dist = self.system.analysis.min_dist()
            i += 1
            wca_cap = wca_cap + 1
            self.system.force_cap = wca_cap
        print("Final min distance warm up", act_min_dist)
        wca_cap = 0
        self.system.force_cap = wca_cap
        self.system.integrator.run(warm_steps)

        # ramp up to simulation temperature
        temp = 3.0
        while temp > 1.0:
            self.system.thermostat.set_langevin(kT=temp, gamma=1.0)
            self.system.integrator.run(warm_steps)
            temp -= 0.01
        print("Final temperature warm up", temp)
        self.system.thermostat.set_langevin(
            kT=self.KT.to("sim_energy").magnitude, gamma=1.0, seed=1434
        )
        self.system.integrator.run(warm_steps)
        print("Warm up finished")

    def equilibrate_pH(self):

        self.reaction1.reaction(reaction_steps= (self.n_nh) + 1)
        self.reaction2.reaction(reaction_steps= (self.n_nh2) + 1)
        if self.USE_WCA:
            self.system.integrator.run(steps=1000)

    def perform_sampling(self,npH):
        self.system.time = 0.
        start_time = time.time()
        n_iter=25
        n_print = int(self.num_samples/n_iter)+1
        self.num_As = np.zeros(n_print)
        self.num_As2 = np.zeros(n_print)
        self.num_B = np.zeros(n_print)
        self.qdist = np.zeros((n_print, self.n_mon))
        self.times = np.zeros(n_print)
        self.en_total = np.zeros(n_print)
        self.e_kin = np.zeros(n_print)
        self.e_coulomb = np.zeros(n_print)
        self.e_bonded = np.zeros(n_print)
        self.e_non_bonded = np.zeros(n_print)
        self.T_inst = np.zeros(n_print)
        self.rad_gyr = np.zeros(n_print)
        self.end_2end = np.zeros(n_print)

        energy_columns = ['en_total', 'e_kin', 'e_coulomb', 'e_bonded', 'e_non_bonded']
        atomic_columns = ['num_As', 'num_As2', 'num_B']
        columns = ['time', 'pH', 'T_inst'] + energy_columns + atomic_columns

        polymer_list_sorted = np.arange(self.n_mon)
        for i in range(self.num_samples):
            if self.USE_WCA and np.random.random() < self.probability_reaction:
                self.system.integrator.run(500)
            if self.USE_WCA and np.random.random() < self.probability_reaction*(2.0/(self.n_mon-2)):
                self.system.integrator.run(500)

            self.reaction1.reaction(reaction_steps=self.n_nh + 1)
            self.reaction2.reaction(reaction_steps=self.n_nh2 + 1)

            if i%n_iter ==0:
                j = (i // n_iter)
                self.num_As[j] = self.system.number_of_particles(type=self.type_particles["N"])
                self.num_As2[j] = self.system.number_of_particles(type=self.type_particles["N2"])
                self.num_B[j] = self.system.number_of_particles(type=self.type_particles["B"])

                qn = self.system.part.by_ids(polymer_list_sorted).q
                self.qdist[j, :] = qn
                self.times[j] = self.system.time
                self.energy = self.system.analysis.energy()
                self.en_total[j] = self.energy['total']
                self.e_kin[j] = self.energy['kinetic']
                self.e_coulomb[j] = self.energy['coulomb']
                self.e_bonded[j] = self.energy["bonded"]
                self.e_non_bonded[j] = self.energy["non_bonded"]
                self.rad_gyr[j] = self.system.analysis.calc_rg(chain_start=0,number_of_chains=1,chain_length=self.n_mon)[0]
                self.end_2end[j] = self.system.analysis.calc_re(chain_start=0,number_of_chains=1,chain_length=self.n_mon)[0]
                self.T_inst[j] = 2. / 3. * self.e_kin[j] /len(self.system.part)

                if i%1000 ==0:
                    print("time: {0:.2f}, n_N: {1:.2f}, n_N2: {2:.2f}".format(self.times[j], self.num_As[j], self.num_As2[j]))

                    print("time: {0:.2f}, energy: {1:.3f},e_kin: {2:.3f},e_coulomb: {3:.3f},e_bonded: {4:.3f},\
                    e_non_bonded: {5:.3f}, Temperature: {6:.3f}".format(self.times[j],self.en_total[j],self.e_kin[j],self.e_coulomb[j],self.e_bonded[j],self.e_non_bonded[j] \
                    ,self.T_inst[j]))

                    print("time: {0:.2f}, end-to-end: {1:.2f}, gyration radius: {2:.2f}".format(self.times[j], self.end_2end[j], self.rad_gyr[j]))

        print(" Completed. Time elapsed: {time.time()-start_time:.2f} seconds.")
