import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pint

from functools import wraps

def _positive_required(tp):
    def decorator(f):
        @wraps(f)
        def wrapper(self, value):
            if not isinstance(value, tp):
                raise TypeError(f"A {tp.__name__} is required.")
            if value <= 0:
                raise ValueError("A positive number is required")
            return f(self, value)
        return wrapper
    return decorator

def _boolean_required(f):
    @wraps(f)
    def wrapper(self, value):
        if not isinstance(value, bool):
            raise TypeError("A boolean is required.")
        return f(self, value)
    return wrapper

def _electrostatics_required(f):
    @wraps(f)
    def wrapper(self, value):
        if value:
            assert USE_WCA, "You can not use electrostatics without a short range repulsive potential. Otherwise oppositely charged particles could come infinitely close."
        return f(self, value)
    return wrapper

class Properties:
    def __init__(self, input_dict):
        self._has_read = False
        self._has_run = False
        self.input = input_dict
        self._read_parameters()
        self._set_units()
        self._simulation_settings()

    @property
    def temperature(self):
        print("Getting Temerature")
        return self._temperature
    @temperature.setter
    @_positive_required((float,int))
    def temperature(self, value):
        print("Setting Temperature")
        self._temperature = value

    @property
    def n_poly(self):
        return self._n_poly
    @n_poly.setter
    @_positive_required(int)
    def n_poly(self, value):
        self._n_poly = value

    @property
    def c_acid(self):
        return self._c_acid
    @c_acid.setter
    @_positive_required((float,int))
    def c_acid(self, value):
        self._c_acid = value

    @property
    def c_salt(self):
        return self._c_salt
    @c_salt.setter
    @_positive_required((float,int))
    def c_salt(self, value):
        self._c_salt = value

    @property
    def n_mon(self):
        return self._n_mon
    @n_mon.setter
    @_positive_required(int)
    def n_mon(self, value):
        self._n_mon = value

    @property
    def n_nh(self):
        return self._n_nh
    @n_nh.setter
    @_positive_required(int)
    def n_nh(self, value):
        if value > self._n_mon:
            raise ValueError("Number of NH monomers cannot be greater than the total number of monomers")
        self._n_nh = value

    @property
    def n_nh2(self):
        return self._n_nh2
    @n_nh2.setter
    @_positive_required(int)
    def n_nh2(self, value):
        if value > self._n_mon:
            raise ValueError("Number of NH2 monomers cannot be greater than the total number of monomers")
        self._n_nh2 = value

    @property
    def bond_l(self):
        return self._bond_l
    @bond_l.setter
    @_positive_required((float,int))
    def bond_l(self, value):
        self._bond_l = value

    @property
    def k_bond(self):
        return self._k_bond
    @k_bond.setter
    @_positive_required((float,int))
    def k_bond(self, value):
        self._k_bond = value

    @property
    def k_bond_a(self):
        return self._k_bond_a
    @k_bond_a.setter
    @_positive_required((float,int))
    def k_bond_a(self, value):
        self._k_bond_a = value

    @property
    def k_dihedral(self):
        return self._k_dihedral
    @k_dihedral.setter
    @_positive_required((float,int))
    def k_dihedral(self, value):
        self._k_dihedral = value

    @property
    def USE_WCA(self):
        return self._USE_WCA
    @USE_WCA.setter
    @_boolean_required
    def USE_WCA(self, value):
        self._USE_WCA = value

    @property
    def USE_ELECTROSTATICS(self):
        return self._USE_ELECTROSTATICS
    @USE_ELECTROSTATICS.setter
    @_boolean_required
    @_electrostatics_required
    def USE_ELECTROSTATICS(self, value):
        self._USE_ELECTROSTATICS = value

    @property
    def USE_FENE(self):
        return self._USE_FENE
    @USE_FENE.setter
    @_boolean_required
    def USE_FENE(self, value):
        self._USE_FENE = value

    @property
    def USE_BENDING(self):
        return self._USE_BENDING
    @USE_BENDING.setter
    @_boolean_required
    def USE_BENDING(self, value):
        self._USE_BENDING = value

    @property
    def USE_DIHEDRAL_POT(self):
        return self._USE_DIHEDRAL_POT
    @USE_DIHEDRAL_POT.setter
    @_boolean_required
    def USE_DIHEDRAL_POT(self, value):
        self._USE_DIHEDRAL_POT = value

    @property
    def USE_P3M(self):
        return self._USE_P3M
    @USE_P3M.setter
    @_boolean_required
    def USE_P3M(self, value):
        self._USE_P3M = value

    @property
    def n_blocks(self):
        return self._n_blocks
    @n_blocks.setter
    @_positive_required(int)
    def n_blocks(self, value):
        self._n_blocks = value

    @property
    def block_size(self):
        return self._block_size
    @block_size.setter
    @_positive_required(int)
    def block_size(self, value):
        self._block_size = value

    @property
    def probability_reaction(self):
        return self._probability_reaction
    @probability_reaction.setter
    @_positive_required((float,int))
    def probability_reaction(self, value):
        self._probability_reaction = value

    def _set_units(self):

        self.ureg = pint.UnitRegistry()
        temperature_kelvin_units = self.temperature*self.ureg.kelvin
        c_acid_molar_units = self.c_acid*self.ureg.molar
        c_salt_molar_units = self.c_salt*self.ureg.molar
        bond_length_angstrom_units = self.bond_l*self.ureg.angstrom
        k_bond_SI_units = self.k_bond*self.ureg.kcal/(self.ureg.mol*(self.ureg.angstrom**2))
        k_bending_SI_units = self.k_bond_a*self.ureg.kcal/self.ureg.mol

        self.KT = temperature_kelvin_units*self.ureg.boltzmann_constant
        WATER_PERMITTIVITY = 80
        self.bjerrum_length = self.ureg.elementary_charge**2 / (4 * self.ureg.pi * self.ureg.vacuum_permittivity * WATER_PERMITTIVITY * self.KT)

        self.ureg.define('sim_energy = {} * boltzmann_constant'.format(temperature_kelvin_units))
        self.ureg.define('sim_length = 0.5 * {}'.format(self.bjerrum_length))
        self.ureg.define('sim_charge = 1 * e')
        self.ureg.define('sim_time = 1e-12 * second')
        self.ureg.define('sim_mass = md_energy * md_time**2 / md_distance**2')
        self.ureg.define('sim_force = md_mass * md_distance / md_time**2')

        # Simulation box (reduced units)
        self.n_acid = self.n_nh + self.n_nh2
        self.box_volume = (self.n_poly*self.n_mon / (self.ureg.avogadro_constant * c_acid_molar_units)).to("nm^3")
        self.box_length = self.box_volume ** (1 / 3)
        self.box_volume_reduced_units = self.box_volume.to("sim_length^3")
        self.box_length_reduced_units = self.box_length.to("sim_length").magnitude
        self.n_salt = int((c_salt_molar_units * self.box_volume * self.ureg.avogadro_constant).to('dimensionless'))
        self.c_acid_unitless = c_acid_molar_units.to('mol/L').magnitude
        self.c_salt_unitless = c_salt_molar_units.to('mol/L').magnitude

        # Simulation parameters (reduced units)
        self.k_bond_reduced_units = (k_bond_SI_units/self.ureg.avogadro_constant).to('sim_energy/(sim_length^2)')
        self.bond_lenth_reduced_units = bond_length_angstrom_units.to("sim_length")
        self.k_bending_reduced_units = (k_bending_SI_units/self.ureg.avogadro_constant).to("sim_energy")
        self.kappa = np.sqrt(c_salt_molar_units.to('mol/L').magnitude)/0.304 / self.ureg.nm
        self.kappa_reduced_units = self.kappa.to('1/sim_length').magnitude

        # Lennard Jones parameters in reduced units
        self.type_particles = {}
        self.charge_reduced_units = {}
        self.sigma_reduced_units = {}
        self.epsilon_reduced_units = {}

        for particle, properties in self.input["Particles properties"].items():
            self.type_particles[particle] = properties["index"]
            self.charge_reduced_units[particle] = (properties["charge"] * self.ureg.e).to("sim_charge").magnitude
            self.sigma_reduced_units[particle] = (properties["sigma"] * self.ureg.angstrom).to("sim_length").magnitude
            self.epsilon_reduced_units[particle] = (((properties["epsilon"]*self.ureg.kelvin)*self.ureg.boltzmann_constant).to('sim_energy')).magnitude

    def _simulation_settings(self):

        self.num_samples = int(self.n_blocks * self.block_size / self.probability_reaction)
        self.sample_iteration_size_capture = 20
        self.sample_begin_capture = 200
        self.n_samp_iter = int((self.num_samples-self.sample_begin_capture)/self.sample_iteration_size_capture)
        self.number_integration_steps = 1000

    def _read_parameters(self):
        input_ = self.input
        # System properties
        self.temperature = input_['Thermal properties']['Temperature'] # Temperature (K)
        self.n_poly = input_['System properties']['Number of polymers']  # N_Polymers
        self.c_acid = input_['System properties']['Concentration acid particles']  # C_Polymer (mol/L)
        self.c_salt = input_['System properties']['Concentration salt particles']
        # Polymer properties
        self.n_mon = input_['System properties']['Number of monomers']  # N_Mon
        self.n_nh = input_['System properties']['NH monomers']  # N_NH
        self.n_nh2 = input_['System properties']['NH2 monomers']  # N_NH2
        self.bond_l = input_['System properties']['bond length']  # L (nm)
        self.k_bond = input_['System properties']['bond constant']  # K_Bond (sim_energy/sim_length^2)
        self.k_bond_a = input_['System properties']['bending constant']  # K_Bond_a (sim_energy/sim_length^2)
        self.k_dihedral = input_['System properties']['dihedral constant']  # K_Dihedral (sim_energy/sim_angle^2)

        # Interaction properties
        self.pK = self.input["pH properties"]["pK1"]  	                                        #pK1
        self.pK2 = self.input["pH properties"]["pK2"]	                                            #pK2
        self.NUM_PHS = self.input["pH properties"]["NUM_PHS"]	                                #NUM_pH's
        self.pHmin = self.input["pH properties"]["pHmin"]                                           # lowest pH value to be used
        self.pHmax = self.input["pH properties"]["pHmax"]                                           # highest pH value to be used
        self.n_blocks = self.input["Montecarlo properties"]["N_BLOCKS"]                                   # Number of samples per block
        self.block_size = self.input["Montecarlo properties"]["DESIRED_BLOCK_SIZE"]                         # desired number of samples per block
        self.probability_reaction = self.input["Montecarlo properties"]["PROB_REACTION"]                                   # probability of accepting the reaction move. This parameter changes the speed of convergence.
        self.time_step = self.input["Simulation configuration"]["time step"]                                       # time step (reduced units)

        self.USE_WCA = self.input["Simulation configuration"]["USE_WCA"]
        self.USE_ELECTROSTATICS = self.input["Simulation configuration"]["USE_ELECTROSTATICS"]
        self.USE_FENE = self.input["Simulation configuration"]["USE_FENE"]
        self.USE_BENDING = self.input["Simulation configuration"]["USE_BENDING"]
        self.USE_DIHEDRAL_POT = self.input["Simulation configuration"]["USE_DIHEDRAL_POT"]
        self.USE_P3M = self.input["Simulation configuration"]["USE_P3M"]
        self.particle_properties = self.input["Particles properties"]
