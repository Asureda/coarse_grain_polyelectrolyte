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

from Properties import Properties


class EnsembleDynamics(Properties):
    def __init__(self, input_dict):
        super().__init__(input_dict)

    def _warm_up_system(self):
        print("Warm up the system")

        warm_steps = 1000
        warm_n_times = 100
        min_dist = 0.8

        wca_cap = 5
        self.system.force_cap = wca_cap
        i = 0
        act_min_dist = self.system.analysis.min_dist()
        self.system.thermostat.set_langevin(kT=0.0, gamma=1.0, seed=7)

        # warmup with zero temperature to remove overlaps
        while (act_min_dist < min_dist):
            for j in range(warm_steps + wca_cap):
                print(j)
                self.system.integrator.run(0.5*warm_steps)
        #    system.integrator.run(warm_steps + wca_cap)
            # Warmup criterion
            act_min_dist = self.system.analysis.min_dist()
            i += 1
            wca_cap = wca_cap + 1
            self.system.force_cap = wca_cap
        print("Final min distance warm up",act_min_dist)
        wca_cap = 0
        system.force_cap = wca_cap
        system.integrator.run(warm_steps)

        # ramp up to simulation temperature
        temp = 0
        while (temp < 1.0):
            self.system.thermostat.set_langevin(kT=temp, gamma=1.0)
            self.system.integrator.run(warm_steps)
            temp += 0.1
        print("Final temperature warm up", temp)
        self.system.thermostat.set_langevin(kT=KT.to("sim_energy").magnitude, gamma=1.0, seed=1234)
        self.system.integrator.run(warm_steps)
        print("Warm up finished")


    def perform_sampling(npH, num_samples, particles, num_As, num_As2, num_B, rad_gyr, end_2end, en_total, e_kin, e_coulomb, e_bonded, e_non_bonded):
        system.time = 0.

        global rdf_HA_Cl_avg, rdf_HA2_Cl_avg, rdf_HA_HA_avg, rdf_HA_N_avg, rdf_HA_Na_avg, rdf_Na_Cl_avg, r1, r2, r3, r4, r5, r6, c
        c = 0

        TYPES = {particle: idx for idx, particle in enumerate(particles)}

        number_of_particles = {particle: system.number_of_particles(type=TYPES[particle]) for particle in particles}
        radial_distributions = {}

        df_number_of_particles = pd.DataFrame(columns=["step"] + particles)
        df_radial_distributions = pd.DataFrame(columns=["step"] + [f"{key[0]}_{key[1]}" for key in radial_distributions.keys()])

        for i in range(num_samples):
            if np.random.random() < PROB_REACTION:
                RE.reaction(reaction_steps=n_NH + 1)
            else:
                system.integrator.run(n_int_steps)

            if np.random.random() < PROB_REACTION*(2.0/(N_MON-2)):
                RE2.reaction(reaction_steps=n_NH2 + 1)
            else:
                system.integrator.run(n_int_steps)

            for particle in particles:
                number_of_particles[particle] = system.number_of_particles(type=TYPES[particle])

            c += 1

            for n in range(N_MON):
                qn = system.part.by_id(n).q
                qdist[npH,n] += qn

            if i % n_iter_prop == 0 and i > begin_sample:
                if COMPUTE_RDF:
                    for type_a in particles:
                        for type_b in particles:
                            if type_a != type_b:
                                key = (type_a, type_b)
                                if key not in radial_distributions:
                                    radial_distributions[key] = []
                                r, rdf_ab = system.analysis.distribution(type_list_a=[TYPES[type_a]], type_list_b=[TYPES[type_b]], r_min=r_min, r_max=r_max, r_bins=r_bins)
                                radial_distributions[key].append(rdf_ab)

                    rdf_HA_Cl_avg[npH,:] += radial_distributions[("HA", "Cl")][-1]/n_samp_iter
                    rdf_HA2_Cl_avg[npH,:] += radial_distributions[("HA2", "Cl")][-1]/n_samp_iter
                    rdf_HA_HA_avg[npH,:] += radial_distributions[("HA", "HA")][-1]/n_samp_iter
                    rdf_HA_N_avg[npH,:] += radial_distributions[("HA", "N")][-1]/n_samp_iter
                    rdf_HA_Na_avg[npH,:] += radial_distributions[("HA", "Na")][-1]/n_samp_iter
                    rdf_Na_Cl_avg[npH,:] += radial_distributions[("Na", "Cl")][-1]/n_samp_iter

                df_number_of_particles.loc[i] = [i] + [number_of_particles[particle] for particle in particles]
                df_radial_distributions.loc[i] = [i] + [rdf[-1] for rdf in radial_distributions.values()]

        return df_number_of_particles, df_radial_distributions
