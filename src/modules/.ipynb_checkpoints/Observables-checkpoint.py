import espressomd.observables
import espressomd.accumulators
import numpy as np
class Observables:
    def __init__(self):
        self.observables = {}

    def add_observable(self, key_name, observable_class, *args, **kwargs):
        observable = observable_class(*args, **kwargs)
        self.observables[key_name] = observable

    def update_accumulators(self, system):
        for obs in self.observables:
            system.auto_update_accumulators.add(self.observables[obs])


    def clear_accumulators(self, system):
        system.auto_update_accumulators.clear()
        self.observables.clear()
        # for obs, valor in self.observables.items():
        #     if isinstance(valor, espressomd.accumulators.TimeSeries):
        #         self.observables[obs] = valor.time_series().fill(0)
        #     else:
        #         self.observables[obs] = valor.result().fill(0)

    def rdf_observable(self, ids1, ids2, r_min, r_max, n_r_bins):
        obs = espressomd.observables.RDF(ids1=ids1, ids2=ids2, min_r=r_min, max_r=r_max, n_r_bins=n_r_bins)
        return espressomd.accumulators.TimeSeries(obs=obs, delta_N=100)

    def particle_distance_observable(self, ids):
        obs = espressomd.observables.ParticleDistances(ids=ids)
        return espressomd.accumulators.TimeSeries(obs=obs, delta_N=100)

    def dihedral_observable(self, ids):
        obs = espressomd.observables.BondDihedrals(ids=ids)
        return espressomd.accumulators.TimeSeries(obs=obs, delta_N=100)

    def position_observable(self, ids):
        obs = espressomd.observables.ParticlePositions(ids=ids)
        return espressomd.accumulators.TimeSeries(obs=obs, delta_N=100)

    def com_pos_cor(self, pids_monomers, tau_max):
        com_pos = espressomd.observables.ComPosition(ids=pids_monomers)
        return espressomd.accumulators.Correlator(
            obs1=com_pos, tau_lin=16, tau_max=tau_max, delta_N=100,
            corr_operation="square_distance_componentwise", compress1="discard1")

    def com_vel_cor(self, pids_monomers, tau_max):
        com_vel = espressomd.observables.ComVelocity(ids=pids_monomers)
        return espressomd.accumulators.Correlator(
            obs1=com_vel, tau_lin=16, tau_max=tau_max, delta_N=100,
            corr_operation="scalar_product", compress1="discard1")
