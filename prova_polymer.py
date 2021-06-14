# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 04:45:17 2021

@author: asure
"""

import matplotlib.pyplot as plt
import numpy as np
import setuptools
import pint
assert setuptools.version.pkg_resources.packaging.specifiers.SpecifierSet('>=0.10.1').contains(pint.__version__), \
  f'pint version {pint.__version__} is too old: several numpy operations can cast away the unit'

import espressomd
espressomd.assert_features(['WCA', 'ELECTROSTATICS'])
import espressomd.electrostatics
import espressomd.reaction_ensemble
import espressomd.polymer
from espressomd.io.writer import vtf  # pylint: disable=import-error
from espressomd.interactions import HarmonicBond
from espressomd.interactions import FeneBond
from espressomd.interactions import AngleHarmonic
from espressomd.interactions import Dihedral
from tipus_particules import save_vxyz
from tipus_particules import convert_vxyz
from espressomd import observables, accumulators, analyze

ureg = pint.UnitRegistry()

TEMPERATURE = 300 * ureg.kelvin
KT = TEMPERATURE * ureg.boltzmann_constant
WATER_PERMITTIVITY = 80
BJERRUM_LENGTH = ureg.elementary_charge**2 / (4 * ureg.pi * ureg.vacuum_permittivity * WATER_PERMITTIVITY * KT)
ureg.define(f'sim_energy = {TEMPERATURE} * boltzmann_constant')
ureg.define(f'sim_length = 0.5 * {BJERRUM_LENGTH}')
ureg.define(f'sim_charge = 1 * e')

n_poly = 1
C_ACID = 5e-3 * ureg.molar
C_SALT = 1.0 * C_ACID
N_NH = 48
N_NH2 = 2
N_ACID = N_NH+N_NH2
N_MON = 148
BOX_V = (N_MON / (ureg.avogadro_constant * C_ACID)).to("sim_length^3")
BOX_L = BOX_V ** (1 / 3)
BOX_L_UNITLESS = BOX_L.to("sim_length").magnitude
N_SALT = int((C_SALT * BOX_V * ureg.avogadro_constant).to('dimensionless'))
k_bond = 300*ureg.kcal/(ureg.mol*(ureg.angstrom**2))
k_bond_sim = (k_bond/ureg.avogadro_constant).to('sim_energy/(sim_length^2)')
C_ACID_UNITLESS = C_ACID.to('mol/L').magnitude
C_SALT_UNITLESS = C_SALT.to('mol/L').magnitude
print('N_salt = ',N_SALT, ', N_MON =',N_MON,', N_ACID = ',N_ACID)
print('BOX Volume = ',BOX_V.to("nm^3"))
print('BOX L = ',BOX_L.to("nm"))
print('Ionic strength =',0.5*(C_ACID+2*C_SALT))
bond_l = 1.5*ureg.angstrom
bond_l_sim = bond_l.to("sim_length")
k_angle = 0.01*ureg.kcal/ureg.mol
k_angle_sim = (k_angle/ureg.avogadro_constant).to("sim_energy")
print(k_angle_sim)
print(k_bond_sim)
print(bond_l_sim)

# acidity constant
pK = 8.18
pK2 = 10.02
K = 10**(-pK)
K2 = 10**(-pK2)
pKw = 14.0  # autoprotolysis constant of water
Kw = 10**(-pKw)
# variables for pH sampling
NUM_PHS = 2  # number of pH values

pHmin = 4.5  # lowest pH value to be used
pHmax = 9.5   # highest pH value to be used
pHs = np.linspace(pHmin, pHmax, NUM_PHS)  # list of pH values
# Simulate an interacting system with steric repulsion (Warning: it will be slower than without WCA!)
USE_WCA = True
# Simulate an interacting system with electrostatics (Warning: it will be very slow!)
USE_ELECTROSTATICS = True
USE_FENE = True
USE_BENDING = True
USE_DIHEDRAL_POT = False
COMPUTE_RDF = True

if USE_ELECTROSTATICS:
    assert USE_WCA, "You can not use electrostatics without a short range repulsive potential. Otherwise oppositely charged particles could come infinitely close."

# Parameters according to the binning method
N_BLOCKS = 16  # number of block to be used in data analysis
DESIRED_BLOCK_SIZE = 100  # desired number of samples per block

PROB_REACTION = 0.5  # probability of accepting the reaction move. This parameter changes the speed of convergence.

# number of reaction samples per each pH value
NUM_SAMPLES = int(N_BLOCKS * DESIRED_BLOCK_SIZE / PROB_REACTION)
print('Number of samples = ' , NUM_SAMPLES)
print("Number of pH's = ", pHs)
TYPES = {
    "HA": 0,
    "A": 1,
    "B": 2,
    "N": 3,
    "Na": 4,
    "Cl": 5,
    "HA2": 6,
    "A2": 7

}
# particle charges of different species
CHARGES = {
    "HA": (+1 * ureg.e).to("sim_charge").magnitude,
    "A": (0 * ureg.e).to("sim_charge").magnitude,
    "B": (+1 * ureg.e).to("sim_charge").magnitude,
    "N": (0 * ureg.e).to("sim_charge").magnitude,
    "Na": (+1 * ureg.e).to("sim_charge").magnitude,
    "Cl": (-1 * ureg.e).to("sim_charge").magnitude,
    "HA2": (+1 * ureg.e).to("sim_charge").magnitude,
    "A2": (0 * ureg.e).to("sim_charge").magnitude

}

lj_sigmas = {
    "0": (2.6 * ureg.angstrom).to("sim_length").magnitude,
    "1": (3.172 * ureg.angstrom).to("sim_length").magnitude,
    "2": (2.958 * ureg.angstrom).to("sim_length").magnitude,
    "3": (3.461 * ureg.angstrom).to("sim_length").magnitude,
    "4": (3.9624 * ureg.angstrom).to("sim_length").magnitude,
    "5": (3.915 * ureg.angstrom).to("sim_length").magnitude,
    "6": (2.6 * ureg.angstrom).to("sim_length").magnitude,
    "7": (3.172 * ureg.angstrom).to("sim_length").magnitude,

}

lj_epsilons = {
    "0": (((231*ureg.kelvin)*ureg.boltzmann_constant).to('sim_energy')).magnitude,
    "1": (((231*ureg.kelvin)*ureg.boltzmann_constant).to('sim_energy')).magnitude,
    "2": (((0.0726*ureg.kcal/ureg.mol)/ureg.avogadro_constant).to('sim_energy')).magnitude,
    "3": (((56.0*ureg.kelvin)*ureg.boltzmann_constant).to('sim_energy')).magnitude,
    "4": (((0.738*ureg.kcal/ureg.mol)/ureg.avogadro_constant).to('sim_energy')).magnitude,
    "5": (((0.305*ureg.kcal/ureg.mol)/ureg.avogadro_constant).to('sim_energy')).magnitude,
    "6": (((0.738*ureg.kcal/ureg.mol)/ureg.avogadro_constant).to('sim_energy')).magnitude

}



def combination_rule_epsilon(rule, eps1, eps2):
    if rule == "Lorentz":
        return (eps1 * eps2)**0.5
    else:
        return ValueError("No combination rule defined")


def combination_rule_sigma(rule, sig1, sig2):
    if rule == "Berthelot":
        return (sig1 + sig2) * 0.5
    else:
        return ValueError("No combination rule defined")

system = espressomd.System(box_l=[BOX_L_UNITLESS] * 3)
print('box',[BOX_L_UNITLESS] * 3)
system.time_step = 0.005
system.cell_system.skin = 0.4
system.periodicity = [True, True, True]
np.random.seed(seed=10)  # initialize the random number generator in numpy
outfile = open('polymer.vtf', 'w')


# we need to define bonds before creating polymers


# Simulate bonded-interactions with FENE potential

if USE_FENE:
    bond_pot = FeneBond(k=30, d_r_max=2.5, r_0=2*bond_l_sim.magnitude)
else :
    bond_pot = HarmonicBond(k=30, r_0=2*bond_l_sim.magnitude)

if USE_BENDING:
    bend=AngleHarmonic(bend=k_angle_sim.magnitude, phi0=np.pi*(2.0/3.0))
    system.bonded_inter.add(bend)
if USE_DIHEDRAL_POT:
    dihedral = Dihedral(bend=1.0, mult=3, phase=0)
    system.bonded_inter.add(dihedral)


system.bonded_inter.add(bond_pot)

# create the polymer positions
polymers = espressomd.polymer.positions(n_polymers=1, beads_per_chain=N_MON,bond_length=2*bond_l_sim.magnitude, seed=23)
#polymers = espressomd.polymer.positions(n_polymers=1, beads_per_chain=N_MON,bond_length=2*bond_l_sim.magnitude, seed=23)
# add the polymer particles composed of ionizable acid groups, initially in the ionized state
n1=0
n2=0
for polymer in polymers:
    prev_particle = None
    for position in polymer:
        iid = len(system.part)
        if iid % 3 == 0 :
            if iid == 0 or iid == (N_MON-1) :
                p = system.part.add(pos=position, type=TYPES["A2"], q=CHARGES["A2"])
                #print(iid,"A2")
                n1 = n1+1
            else :
                p = system.part.add(pos=position, type=TYPES["A"], q=CHARGES["A"])
                #print(iid,"A")
                n2 = n2+1
        else :
            p = system.part.add(pos=position, type=TYPES["N"], q=CHARGES["N"])
            #print(iid,"N")
        #p = system.part.add(pos=position, type=TYPES["A"], q=CHARGES["A"])
        if prev_particle:
            p.add_bond((bond_pot, prev_particle))
            if USE_BENDING:
                if iid < N_MON-1:
                    p.add_bond((bend,iid-1,iid+1))
            if USE_DIHEDRAL_POT:
                if iid < N_MON-2:
                    print(p.id,iid-1,iid+1,iid+2)
                    p.add_bond((dihedral, iid-1, iid+1, iid+2))

        prev_particle = p

#system.part[1].add_bond((angle_harmonic, 0, 2))
#system.part[1:-1].add_bond((angle_harmonic, np.arange(N_MON)[:-2], np.arange(N_MON)[2:]))
print('n1',n1)
print('n2',n2)
ADD_HIDRO_IONS = True
if ADD_HIDRO_IONS:
#add the corresponding number of B+ ions
    system.part.add(pos=np.random.random((n_poly*N_ACID, 3)) * BOX_L_UNITLESS,
                    type=[TYPES["B"]] * n_poly*N_ACID,
                    q=[CHARGES["B"]] * n_poly*N_ACID)

    system.part.add(pos=np.random.random((n_poly*N_ACID, 3)) * BOX_L_UNITLESS,
                type=[TYPES["Cl"]] * n_poly*N_ACID,
                q=[CHARGES["Cl"]] * n_poly*N_ACID)


#add the corresponding number of OH- ions
    # system.part.add(pos=np.random.random((N_ACID, 3)) * BOX_L_UNITLESS,
    #                 type=[TYPES["OH"]] * N_ACID,
    #                 q=[CHARGES["OH"]] * N_ACID)

#
# add salt ion pairs
ADD_SALT = True
if ADD_SALT:
    system.part.add(pos=np.random.random((N_SALT, 3)) * BOX_L_UNITLESS,
                type=[TYPES["Na"]] * N_SALT,
                q=[CHARGES["Na"]] * N_SALT)

    system.part.add(pos=np.random.random((N_SALT, 3)) * BOX_L_UNITLESS,
                type=[TYPES["Cl"]] * N_SALT,
                q=[CHARGES["Cl"]] * N_SALT)

# Write the structure of the system (type, bonds and coordinates of each particle)
vtf.writevsf(system, outfile)

# Check charge neutrality
assert np.abs(np.sum(system.part[:].q)) < 1E-10

# Set the parameters
if USE_WCA:
    for type_1, type_2 in ((x, y) for x in TYPES.values() for y in TYPES.values()):
        lj_sig = combination_rule_sigma("Berthelot", lj_sigmas[str(type_1)], lj_sigmas[str(type_2)])
        #lj_eps = combination_rule_epsilon("Lorentz", lj_epsilons[str(type_1)], lj_epsilons[str(type_2)])

        system.non_bonded_inter[type_1, type_2].wca.set_params(epsilon=1.0, sigma=lj_sig)
        #print('sigma',type_1, type_2, lj_sig)
        #print('epsilon',type_1, type_2, lj_eps)

    # relax the overlaps with steepest descent
    system.integrator.set_steepest_descent(f_max=0, gamma=0.1, max_displacement=0.1)
    system.integrator.run(1000)
    system.integrator.set_vv()  # to switch back to velocity Verlet

vtf.writevcf(system, outfile)

# save_vxyz(system,'polymer1.xyz',mode='a',aplicar_PBC=True)
# convert_vxyz('polyer1.xyz','polymer2.xyz')
# add thermostat and short integration to let the system relax
system.thermostat.set_langevin(kT=KT.to("sim_energy").magnitude, gamma=1.0, seed=7)
system.integrator.run(steps=1000)

vtf.writevcf(system, outfile)
outfile2 = 'polymer2.xyz'
#save_vxyz(system,outfile2,mode='w',aplicar_PBC=True)

# Prefactor in reduced units, if you want to make production run, better with a lower accuracy
if USE_ELECTROSTATICS:
    p3m = espressomd.electrostatics.P3M(
        prefactor=(BJERRUM_LENGTH * KT / (ureg.elementary_charge ** 2)
                   ).to("sim_length * sim_energy / sim_charge^2").magnitude,
        accuracy=1e-3)
    system.actors.add(p3m)
# else:
#     # this speeds up the simulation of dilute systems with small particle numbers
#     system.cell_system.set_n_square()
#p3m.tune()
#exclusion_radius = 1.0 if USE_WCA else 0.0
exclusion_radius = combination_rule_sigma("Berthelot", lj_sigmas["1"], lj_sigmas["2"]) if USE_WCA else 0.0
print(exclusion_radius)
RE = espressomd.reaction_ensemble.ConstantpHEnsemble(
    temperature=KT.to("sim_energy").magnitude,
    exclusion_radius=exclusion_radius,
    seed=77
)

exclusion_radius2 = combination_rule_sigma("Berthelot", lj_sigmas["6"], lj_sigmas["2"]) if USE_WCA else 0.0
print(exclusion_radius)
RE2 = espressomd.reaction_ensemble.ConstantpHEnsemble(
    temperature=KT.to("sim_energy").magnitude,
    exclusion_radius=exclusion_radius2,
    seed=77
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


# H2O autoprotolysis
# RE.add_reaction(gamma=(1 / Kw),
#                     reactant_types=[],
#                     reactant_coefficients=[],
#                     product_types=[TYPES["B"], TYPES["OH"]],
#                     product_coefficients=[1, 1],
#                     default_charges={TYPES["B"]: CHARGES["B"],
#                     TYPES["OH"]: CHARGES["OH"]}
# )

#print(RE.get_status())
#system.setup_type_map([TYPES["HA"],TYPES["A"],TYPES["B"],TYPES["N"],TYPES["Na"],TYPES["Cl"],TYPES["OH"],TYPES["HA2"],TYPES["A2"]])

pids_HA2 = system.part.select(type=TYPES["HA2"]).id
pids_HA = system.part.select(type=TYPES["HA"]).id
pids_A2 = system.part.select(type=TYPES["A2"]).id
pids_A = system.part.select(type=TYPES["A"]).id
pids_N = system.part.select(type=TYPES["N"]).id
pids_Na = system.part.select(type=TYPES["Na"]).id
pids_Cl = system.part.select(type=TYPES["Cl"]).id
# Calculate the averaged rdfs
r_bins = 500
r_min = 0.0
r_max = system.box_l[0] / 2.0

rdf_HA_Cl_avg = np.zeros((len(pHs), r_bins))
rdf_HA2_Cl_avg = np.zeros((len(pHs), r_bins))
rdf_HA_HA_avg = np.zeros((len(pHs), r_bins))
rdf_HA_N_avg = np.zeros((len(pHs), r_bins))
rdf_HA_Na_avg = np.zeros((len(pHs), r_bins))
rdf_Na_Cl_avg = np.zeros((len(pHs), r_bins))
qdist = np.zeros((len(pHs), N_MON))

def ideal_alpha(pH, pK):
    return 1. / (1 + 10**(pK - pH))

def equilibrate_pH():
    RE.reaction(reaction_steps=20 * N_ACID + 1)
    RE2.reaction(reaction_steps=20 * N_ACID + 1)
    if USE_WCA:
        system.integrator.run(steps=1000)


def perform_sampling(npH,num_samples, num_As: np.ndarray, num_As2: np.ndarray, num_B: np.ndarray, rad_gyr: np.ndarray, end_2end: np.ndarray,en_total: np.ndarray):
    system.time = 0.

    global rdf_HA_Cl_avg, rdf_HA2_Cl_avg,rdf_HA_HA_avg,rdf_HA_N_avg,rdf_HA_Na_avg,rdf_Na_Cl_avg, r1, r2, r3, r4, r5, r6, c
    c = 0

    for i in range(num_samples):
        if np.random.random() < PROB_REACTION:
            # should be at least one reaction attempt per particle
            RE.reaction(reaction_steps=N_NH + 1)

        if np.random.random() < PROB_REACTION*(2.0/(N_MON-2)):
            # should be at least one reaction attempt per particle
            RE2.reaction(reaction_steps=N_NH2 + 1)

        if USE_WCA:
            system.integrator.run(steps=1000)
        num_As[i] = system.number_of_particles(type=TYPES["A"])
        num_As2[i] = system.number_of_particles(type=TYPES["A2"])
        num_B[i] = system.number_of_particles(type=TYPES["B"])
        c = c + 1

        for n in range(N_MON):
            qn = system.part[n].q
            qdist[npH,n] = qdist[npH,n] + qn
            #print(qdist)

        #num_OH[i] = system.number_of_particles(type=TYPES["OH"])
        if COMPUTE_RDF:
            r1, rdf_HA_Cl = system.analysis.rdf(rdf_type="rdf", type_list_a=[TYPES["HA"]], type_list_b=[TYPES["Cl"]], r_min=r_min, r_max=r_max, r_bins=r_bins)
            r2, rdf_HA2_Cl = system.analysis.rdf(rdf_type="rdf", type_list_a=[TYPES["HA2"]], type_list_b=[TYPES["Cl"]], r_min=r_min, r_max=r_max, r_bins=r_bins)
            r3, rdf_HA_HA = system.analysis.rdf(rdf_type="rdf", type_list_a=[TYPES["HA"]], type_list_b=[TYPES["HA"]], r_min=r_min, r_max=r_max, r_bins=r_bins)
            r4, rdf_HA_N = system.analysis.rdf(rdf_type="rdf", type_list_a=[TYPES["HA"]], type_list_b=[TYPES["N"]], r_min=r_min, r_max=r_max, r_bins=r_bins)
            r5, rdf_HA_Na = system.analysis.rdf(rdf_type="rdf", type_list_a=[TYPES["HA"]], type_list_b=[TYPES["Na"]], r_min=r_min, r_max=r_max, r_bins=r_bins)
            r6, rdf_Na_Cl = system.analysis.rdf(rdf_type="rdf", type_list_a=[TYPES["Na"]], type_list_b=[TYPES["Cl"]], r_min=r_min, r_max=r_max, r_bins=r_bins)
            rdf_HA_Cl_avg[npH,:] += rdf_HA_Cl/num_samples
            rdf_HA2_Cl_avg[npH,:] += rdf_HA2_Cl/num_samples
            rdf_HA_HA_avg[npH,:] += rdf_HA_HA/num_samples
            rdf_HA_N_avg[npH,:] += rdf_HA_N/num_samples
            rdf_HA_Na_avg[npH,:] += rdf_HA_Na/num_samples
            rdf_Na_Cl_avg[npH,:] += rdf_Na_Cl/num_samples

        rad_gyr[i] = system.analysis.calc_rg(chain_start=0,number_of_chains=1,chain_length=N_MON)[0]
        end_2end[i] = system.analysis.calc_re(chain_start=0,number_of_chains=1,chain_length=N_MON)[0]
        times[i] = system.time
        energy = system.analysis.energy()
        en_total[i] = energy['total']
        e_kin[i] = energy['kinetic']
        #print(times)
        #save_vxyz(system,outfile2,mode='a',aplicar_PBC=True)


        # vtf.writevcf(system, outfile) #Write the final configuration at each pH production run

times = np.zeros(NUM_SAMPLES)
# Observables for cheeck that the simulation is consistent with the LAngevin thermostat parameters
e_kin = np.zeros_like(times)
T_inst = np.zeros_like(times)

# empty numpy array as placeholders for collecting data, array in matrix form, pH x NUM_SAMPLES
num_As_at_each_pH = -np.ones((len(pHs), NUM_SAMPLES))  # number of NH species observed at each sample
num_As2_at_each_pH = -np.ones((len(pHs), NUM_SAMPLES))  # number of NH2 species observed at each sample
num_B_at_each_pH = -np.ones((len(pHs), NUM_SAMPLES))  # number of B+ species observed at each sample
#num_OH_at_each_pH = -np.ones((len(pHs), NUM_SAMPLES))  # number of OH- species observed at each sample

rad_gyr_at_each_pH = np.zeros((len(pHs), NUM_SAMPLES))  # radius of gyration  observed at each sample
end_2e_at_each_pH = np.zeros((len(pHs), NUM_SAMPLES))  # end to end distance  observed at each sample
energy_at_each_pH = np.zeros((len(pHs), NUM_SAMPLES))  # energy observed at each sample

# run a productive simulation and collect the data
print(f"Simulated pH values: {pHs}")
for ipH, pH in enumerate(pHs):
    print(f"Run pH {pH:.2f} ...")

    RE.constant_pH = pH  # set new pH value
    RE2.constant_pH = pH  # set new pH value

    equilibrate_pH()  # pre-equilibrate to the new pH value
    perform_sampling(ipH,NUM_SAMPLES, num_As_at_each_pH[ipH, :], num_As2_at_each_pH[ipH, :], num_B_at_each_pH[ipH, :],rad_gyr_at_each_pH[ipH, :],end_2e_at_each_pH[ipH, :],energy_at_each_pH[ipH, :])  # perform sampling/ run production simulation
    #vtf.writevcf(system, outfile) #Write the final configuration at each pH production runç
    print(f"measured number of NH: {np.mean(num_As_at_each_pH[ipH]):.2f}, (ideal: {N_NH*ideal_alpha(pH, pK):.2f})")
    print(f"measured number of NH2: {np.mean(num_As2_at_each_pH[ipH]):.2f}, (ideal: {N_NH2*ideal_alpha(pH, pK2):.2f})")
    print(f"measured number of NH2+NH: {np.mean(num_As2_at_each_pH[ipH])+np.mean(num_As_at_each_pH[ipH]):.2f}, (ideal: {N_NH*ideal_alpha(pH, pK)+N_NH2*ideal_alpha(pH, pK2):.2f})")
    print(f"measured number of B+: {np.mean(num_B_at_each_pH[ipH]):.2f})")
    save_vxyz(system,outfile2,mode='a',aplicar_PBC=True)
    vtf.writevcf(system, outfile) #Write the final configuration at each pH production run

    print(f"radius of gyration : {np.mean(rad_gyr_at_each_pH[ipH]):.2f}")
    print(f"end to end distance : {np.mean(end_2e_at_each_pH[ipH]):.2f}")
    print(f"energy : {np.mean(energy_at_each_pH[ipH]):.2f}")

outfile3 = 'polymer3.xyz'
convert_vxyz(outfile2,outfile3)
outfile.close()

# To check that the Langevin thermostat works correctly we compute the instantaneous temperature at each sample
T_inst = 2. / 3. * e_kin / (N_MON + 2*N_ACID + 2*N_SALT)
# ionization degree alpha calculated from the Henderson-Hasselbalch equation for an ideal system
qdist = qdist / c



# statistical analysis of the results
def block_analyze(input_data, n_blocks=16):
    data = np.asarray(input_data)
    block = 0
    # this number of blocks is recommended by Janke as a reasonable compromise
    # between the conflicting requirements on block size and number of blocks
    block_size = int(data.shape[1] // n_blocks)
    print(f"block_size: {block_size}")
    # initialize the array of per-block averages
    block_average = np.zeros((n_blocks, data.shape[0]))
    # calculate averages per each block
    for block in range(n_blocks):
        block_average[block] = np.average(data[:, block * block_size: (block + 1) * block_size], axis=1)
    # calculate the average and average of the square
    av_data = np.average(data, axis=1)
    av2_data = np.average(data * data, axis=1)
    # calculate the variance of the block averages
    block_var = np.var(block_average, axis=0)
    # calculate standard error of the mean
    err_data = np.sqrt(block_var / (n_blocks - 1))
    # estimate autocorrelation time using the formula given by Janke
    # this assumes that the errors have been correctly estimated
    tau_data = np.zeros(av_data.shape)
    for val in range(av_data.shape[0]):
        if av_data[val] == 0:
            # unphysical value marks a failure to compute tau
            tau_data[val] = -1.0
        else:
            tau_data[val] = 0.5 * block_size * n_blocks / (n_blocks - 1) * block_var[val] \
                / (av2_data[val] - av_data[val] * av_data[val])
    return av_data, err_data, tau_data, block_size

# estimate the statistical error and the autocorrelation time
av_num_As, err_num_As, tau, block_size = block_analyze(num_As_at_each_pH, N_BLOCKS)
av_num_As2, err_num_As2, tau5, block_size = block_analyze(num_As2_at_each_pH, N_BLOCKS)
av_num_B, err_num_B, tau6, block_size = block_analyze(num_B_at_each_pH, N_BLOCKS)
#av_num_OH, err_num_OH, tau7, block_size = block_analyze(num_OH_at_each_pH, N_BLOCKS)

av_rad_gyr, err_rad_gyr, tau2, block_size = block_analyze(rad_gyr_at_each_pH, N_BLOCKS)
av_end_2e, err_end_2e, tau3, block_size = block_analyze(end_2e_at_each_pH, N_BLOCKS)
av_energy, err_energy, tau4, block_size = block_analyze(energy_at_each_pH, N_BLOCKS)

print(f"av = {av_num_As}")
print(f"err = {err_num_As}")
print(f"tau = {tau}")

print(f"av = {av_num_As2}")
print(f"err = {err_num_As2}")
print(f"tau = {tau5}")

print(f"av = {av_num_B}")
print(f"err = {err_num_B}")
print(f"tau = {tau6}")

# print(f"av = {av_num_OH}")
# print(f"err = {err_num_OH}")
# print(f"tau = {tau7}")

print(f"av = {av_rad_gyr}")
print(f"err = {err_rad_gyr}")
print(f"tau = {tau2}")

print(f"av = {av_end_2e}")
print(f"err = {err_end_2e}")
print(f"tau = {tau3}")

print(f"av = {av_energy}")
print(f"err = {err_energy}")
print(f"tau = {tau4}")


# calculate the average ionization degree
av_alpha = (av_num_As  + av_num_As2 ) / N_ACID
err_alpha = (err_num_As + err_num_As2) / N_ACID
NH_alpha = (av_num_As) / N_NH
err_NH_alpha = (err_num_As) / N_NH
NH2_alpha = (av_num_As2) / N_NH2
err_NH2_alpha = (err_num_As2) / N_NH2
# plot the simulation results compared with the ideal titration curve
plt.figure(figsize=(10, 6), dpi=80)
plt.errorbar(pHs, av_alpha, err_alpha, marker='o', linestyle='dotted',
             label=r"simulation")
plt.errorbar(pHs, NH_alpha, err_NH_alpha, marker='o', linestyle='dotted',
             label=r"NH simulation")
plt.errorbar(pHs, NH2_alpha, err_NH2_alpha, marker='o', linestyle='dotted',
             label=r"NH2 simulation")

pHs2 = np.linspace(pHmin, pHmax, num=50)
plt.plot(pHs2 , ideal_alpha(pHs2, pK), label=r"ideal1")
plt.plot(pHs2 , ideal_alpha(pHs2, pK2), label=r"ideal2")
plt.xlabel('pH', fontsize=16)
plt.ylabel(r'$\alpha$', fontsize=16)
plt.legend(fontsize=16)
plt.savefig('alpha.png')

plt.figure(figsize=(10, 6), dpi=80)
plt.errorbar(pHs , 1-av_alpha, err_alpha, marker='o', linestyle='dotted',
             label=r"simulation")
plt.errorbar(pHs, 1 -NH_alpha, err_NH_alpha, marker='o', linestyle='dotted',
             label=r"NH simulation")
plt.errorbar(pHs, 1-NH2_alpha, err_NH2_alpha, marker='o', linestyle='dotted',
             label=r"NH2 simulation")

plt.plot(pHs2, 1-ideal_alpha(pHs2, pK), label=r"ideal1")
plt.plot(pHs2, 1-ideal_alpha(pHs2, pK2), label=r"ideal2")
plt.xlabel('pH', fontsize=16)
plt.ylabel(r'1 - $\alpha$', fontsize=16)
plt.legend(fontsize=16)
plt.savefig('alpha_minus_one.png')

# Radius of gyration at each pH run
plt.figure(figsize=(10, 6), dpi=80)
plt.errorbar(pHs , av_rad_gyr, err_rad_gyr, marker='o', linestyle='dotted',
             label=r"simulation")
plt.xlabel('pH', fontsize=16)
plt.ylabel(r'$R_{g}$', fontsize=16)
plt.legend(fontsize=16)
plt.savefig('rad_gyr.png')

# End to end distance at each pH run
plt.figure(figsize=(10, 6), dpi=80)
plt.errorbar(pHs , av_end_2e, err_end_2e, marker='o', linestyle='dotted',
             label=r"simulation")
plt.xlabel('pH', fontsize=16)
plt.ylabel(r'$R_{e}$', fontsize=16)
plt.legend(fontsize=16)
plt.savefig('end_2e.png')

# Energy of the system at each pH run
plt.figure(figsize=(10, 6), dpi=80)
plt.errorbar(pHs , av_energy, err_energy, marker='o', linestyle='dotted',
             label=r"simulation")
plt.xlabel('pH', fontsize=16)
plt.ylabel(r'$E$', fontsize=16)
plt.legend(fontsize=16)
plt.savefig('energy.png')

# Temperature
plt.figure(figsize=(10, 6), dpi=80)
plt.plot(times, T_inst, label='$T_{\\mathrm{inst}}$')
plt.plot(times, [KT.to("sim_energy").magnitude]*len(times), label='$T$ set by thermostat')
plt.legend()
plt.xlabel('t')
plt.ylabel('T')
plt.savefig('Temperature.png')

# Energy of the system for different pH's
plt.figure(figsize=(10, 6), dpi=80)
for i in range(len(pHs)):
    plt.plot(times,energy_at_each_pH[i][:] , label=r'$E$ '+ str(i))
    plt.legend()
plt.xlabel('t')
plt.ylabel('E')
plt.savefig('Energy_sample_pH.png')



# RDF1
plt.figure(figsize=(10, 6), dpi=80)
for i in range(len(pHs)):
    plt.plot(r1,rdf_HA_Cl_avg[i][:] , label=r'$rdf$ '+ str(i))
    plt.legend()
plt.xlabel('r')
plt.ylabel('g(r)')
plt.savefig('RDF1.png')
# RDF2
plt.figure(figsize=(10, 6), dpi=80)
for i in range(len(pHs)):
    plt.plot(r2,rdf_HA2_Cl_avg[i][:] , label=r'$rdf$ '+ str(i))
    plt.legend()
plt.xlabel('r')
plt.ylabel('g(r)')
plt.savefig('RDF2.png')

# RDF3
plt.figure(figsize=(10, 6), dpi=80)
for i in range(len(pHs)):
    plt.plot(r3,rdf_HA_HA_avg[i][:] , label=r'$rdf$ '+ str(i))
    plt.legend()
plt.xlabel('r')
plt.ylabel('g(r)')
plt.savefig('RDF3.png')

# RDF4
plt.figure(figsize=(10, 6), dpi=80)
for i in range(len(pHs)):
    plt.plot(r4,rdf_HA_N_avg[i][:] , label=r'$rdf$ '+ str(i))
    plt.legend()
plt.xlabel('r')
plt.ylabel('g(r)')
plt.savefig('RDF4.png')

# RDF5
plt.figure(figsize=(10, 6), dpi=80)
for i in range(len(pHs)):
    plt.plot(r5,rdf_HA_Na_avg[i][:] , label=r'$rdf$ '+ str(i))
    plt.legend()
plt.xlabel('r')
plt.ylabel('g(r)')
plt.savefig('RDF5.png')

# RDF6
plt.figure(figsize=(10, 6), dpi=80)
for i in range(len(pHs)):
    plt.plot(r6,rdf_Na_Cl_avg[i][:] , label=r'$rdf$ '+ str(i))
    plt.legend()
plt.xlabel('r')
plt.ylabel('g(r)')
plt.savefig('RDF6.png')

# charge distribution
# Energy of the system for different pH's
n_mono=np.linspace(0,N_MON,num=N_MON,dtype=int)
plt.figure(figsize=(10, 6), dpi=80)
for i in range(len(pHs)):
    plt.plot(n_mono,qdist[i][:] , label=r'$Q$ '+ str(i))
    plt.legend()
plt.xlabel('t')
plt.ylabel('E')
plt.savefig('Q_charge.png')

# average concentration of B+ is the same as the concentration of A-

av_c_Bplus = av_alpha * C_ACID_UNITLESS
err_c_Bplus = err_alpha * C_ACID_UNITLESS  # error in the average concentration

full_pH_range = np.linspace(2, 12, 100)
ideal_c_Aminus = ideal_alpha(full_pH_range, pK) * C_ACID_UNITLESS
ideal_c_Aminus2 = ideal_alpha(full_pH_range, pK2) * C_ACID_UNITLESS
ideal_c_OH = np.power(10.0, -(pKw - full_pH_range))
ideal_c_H = np.power(10.0, -full_pH_range)
# ideal_c_M is calculated from electroneutrality
ideal_c_M = np.clip((ideal_c_Aminus + ideal_c_OH - ideal_c_H), 0, np.inf)

ideal_c_X = np.clip(-(ideal_c_Aminus + ideal_c_OH - ideal_c_H), 0, np.inf)

ideal_ionic_strength = 0.5 * \
    (ideal_c_X + ideal_c_M + ideal_c_H + ideal_c_OH + 2 * C_SALT_UNITLESS)
# in constant-pH simulation ideal_c_Aminus = ideal_c_Bplus
cpH_ionic_strength = 0.5 * (ideal_c_Aminus + 2 * C_SALT_UNITLESS)
cpH_ionic_strength_measured = 0.5 * (av_c_Bplus + 2 * C_SALT_UNITLESS)
cpH_error_ionic_strength_measured = 0.5 * err_c_Bplus

plt.figure(figsize=(10, 6), dpi=80)
plt.errorbar(pHs,
             cpH_ionic_strength_measured,
             cpH_error_ionic_strength_measured,
             c="tab:blue",
             linestyle='none', marker='o',
             label=r"measured", zorder=3)
plt.plot(full_pH_range,
         cpH_ionic_strength,
         c="tab:blue",
         ls=(0, (5, 5)),
         label=r"constant-pH", zorder=2)
plt.plot(full_pH_range,
         ideal_ionic_strength,
         c="tab:orange",
         linestyle='-',
         label=r"ideal", zorder=1)


plt.yscale("log")
plt.xlabel('input pH', fontsize=16)
plt.ylabel(r'Ionic Strength [$\mathrm{mol/L}$]', fontsize=16)
plt.legend(fontsize=16)
plt.show()
