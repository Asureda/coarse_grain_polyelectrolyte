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
from Model import Model


class EnsembleDynamics(Model):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        if self.USE_WCA:
            self._warm_up_system()

    def _warm_up_system(self):
        print("\nWarm up the system:")

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

    def equilibrate_pH(self,pH):
        self.reaction1.constant_pH = pH
        self.reaction2.constant_pH = pH
        self.reaction1.reaction(reaction_steps= (self.n_nh) + 1)
        self.reaction2.reaction(reaction_steps= (self.n_nh2) + 1)
        if self.USE_WCA:
            self.system.integrator.run(steps=1000)

    def perform_sampling(self,npH):
        
        self.system.time = 0.
        start_time = time.time()
        self.num_As = np.zeros(self.num_samples)
        self.num_As2 = np.zeros(self.num_samples)
        self.num_B = np.zeros(self.num_samples)
        self.qdist = np.zeros((self.n_samp_iter, self.n_mon))
        self.times = np.zeros(self.num_samples)
        self.rad_gyr = np.zeros(self.num_samples)
        self.end_2end = np.zeros(self.num_samples)
        
        polymer_list_sorted = np.arange(self.n_mon)
        for i in range(self.num_samples):
            if self.USE_WCA and np.random.random() < self.probability_reaction:
                self.system.integrator.run(self.number_integration_steps)
            if self.USE_WCA and np.random.random() < self.probability_reaction*(2.0/(self.n_mon-2)):
                self.system.integrator.run(self.number_integration_steps)

            self.reaction1.reaction(reaction_steps=5*self.n_nh + 1)
            self.reaction2.reaction(reaction_steps=5*self.n_nh2 + 1)

            self.num_As[i] = self.system.number_of_particles(type=self.type_particles["N"])
            self.num_As2[i] = self.system.number_of_particles(type=self.type_particles["N2"])
            self.num_B[i] = self.system.number_of_particles(type=self.type_particles["B"])
            self.rad_gyr[i] = self.system.analysis.calc_rg(chain_start=0,number_of_chains=1,chain_length=self.n_mon)[0]
            self.end_2end[i] = self.system.analysis.calc_re(chain_start=0,number_of_chains=1,chain_length=self.n_mon)[0]
            self.times[i] = self.system.time
            if i%self.n_samp_iter ==0:
                 j = (i // self.n_samp_iter)
                 qn = self.system.part.by_ids(polymer_list_sorted).q
                 self.qdist[j, :] = qn
            #     self.energy = self.system.analysis.energy()
            #     self.en_total[j] = self.energy['total']
            #     self.e_kin[j] = self.energy['kinetic']
            #     self.e_coulomb[j] = self.energy['coulomb']
            #     self.e_bonded[j] = self.energy["bonded"]
            #     self.e_non_bonded[j] = self.energy["non_bonded"]
            #     self.T_inst[j] = 2. / 3. * self.e_kin[j] /len(self.system.part)

            if i%500 ==0:
                print("time: {0:.2f}, n_N: {1:.2f}, n_N2: {2:.2f}".format(self.times[i], self.num_As[i], self.num_As2[i]))
                print("time: {0:.2f}, end-to-end: {1:.2f}, gyration radius: {2:.2f}".format(self.times[i], self.end_2end[i], self.rad_gyr[i]))

        print(" Completed. Time elapsed: {time.time()-start_time:.2f} seconds.")
