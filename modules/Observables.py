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

    def rdf_observable_corrected(distances, nA, nB, box_size, bin_count=100):
        """Calcula la función de distribución radial (RDF) de un conjunto de distancias entre partículas.

        Args:
            distances (np.array): Array 1D con las distancias entre pares de partículas.
            particle_count (int): El número total de partículas en el sistema.
            box_size (float): El tamaño de la caja en unidades de longitud.
            bin_count (int, optional): El número de bins en los que se divide el rango de distancias. Por defecto es 100.

        Returns:
            Tuple: Un par de arrays numpy con las distancias radiales y la RDF correspondiente, normalizada por la cantidad de pares de partículas y el volumen de la caja.
        """
        n = len(dist)
        particle_count = nA*nB
        total_number = nA+nB
        bin_edges = np.linspace(0, box_size / 2, bin_count + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_volume = (4 / 3) * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
        counts, _ = np.histogram(distances, bins=bin_edges)
        density = n / (box_size**3)
        rdf_values = counts / (bin_volume * density )
        return bin_centers, rdf_values

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
