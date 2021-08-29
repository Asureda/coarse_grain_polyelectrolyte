
import matplotlib.pyplot as plt
import numpy as np

import pint

import espressomd
espressomd.assert_features(['WCA', 'ELECTROSTATICS'])
import espressomd.electrostatics
import espressomd.reaction_ensemble
import espressomd.polymer

from espressomd.interactions import HarmonicBond
from espressomd.interactions import FeneBond
from espressomd.interactions import AngleHarmonic
from espressomd.interactions import Dihedral
from espressomd import observables, accumulators, analyze

import tipus_particules as tp
from sample_functions import ideal_alpha
from sample_functions import block_analyze
from sample_functions import use_interactions
import config as conf
ureg = pint.UnitRegistry()

input = np.loadtxt('input.dat')
TEMPERATURE = input[0] * ureg.kelvin        #Temperature (K)
n_poly = int(input[1])	                    #N_Polymers
C_ACID = input[2]* ureg.molar               #C_Polymer (mol/L)
C_SALT = 2*input[3]* ureg.molar	            #C_Salt (mol/L)
N_MON = int(input[4])	                    #N_MON
N_NH = int(input[5]) 	                    #N_NH
N_NH2 = int(input[6])	                    #N_NH2
bond_l = input[7]*ureg.angstrom             #bond_length (nm)
k_bond = input[8]*ureg.kcal/(ureg.mol*(ureg.angstrom**2)) #k_bond (kcal/mol*A^2)
k_angle = input[9]*ureg.kcal/ureg.mol	                  #k_angle (kcal/mol)
pK = input[10]  	                        #pK1
pK2 = input[11]	                            #pK2
NUM_PHS = int(input[12])	                #NUM_pH's
pHmin = input[13]                           # lowest pH value to be used
pHmax = input[14]                           # highest pH value to be used
N_BLOCKS = int(input[15])                   # Number of samples per block
DESIRED_BLOCK_SIZE = int(input[16])         # desired number of samples per block
PROB_REACTION = input[17]                   # probability of accepting the reaction move. This parameter changes the speed of convergence.
time_step = input[18]
USE_WCA = conf.USE_WCA
USE_ELECTROSTATICS = conf.USE_ELECTROSTATICS
USE_FENE = conf.USE_FENE
USE_BENDING = conf.USE_BENDING
USE_DIHEDRAL_POT = conf.USE_DIHEDRAL_POT
USE_P3M = conf.USE_P3M

KT = TEMPERATURE * ureg.boltzmann_constant
WATER_PERMITTIVITY = 80
BJERRUM_LENGTH = ureg.elementary_charge**2 / (4 * ureg.pi * ureg.vacuum_permittivity * WATER_PERMITTIVITY * KT)
ureg.define(f'sim_energy = {TEMPERATURE} * boltzmann_constant')
ureg.define(f'sim_length = 0.5 * {BJERRUM_LENGTH}')
ureg.define(f'sim_charge = 1 * e')

# Simulation box (reduced units)
N_ACID = N_NH + N_NH2
BOX_V = (N_MON / (ureg.avogadro_constant * C_ACID)).to("nm^3")
BOX_V_UNITLESS = BOX_V.to("sim_length^3")
BOX_L = BOX_V ** (1 / 3)
BOX_L_UNITLESS = BOX_L.to("sim_length").magnitude
N_SALT = int((C_SALT * BOX_V * ureg.avogadro_constant).to('dimensionless'))
C_ACID_UNITLESS = C_ACID.to('mol/L').magnitude
C_SALT_UNITLESS = C_SALT.to('mol/L').magnitude

# Simulation parameters (reduced units)
k_bond_sim = (k_bond/ureg.avogadro_constant).to('sim_energy/(sim_length^2)')
bond_l_sim = bond_l.to("sim_length")
k_angle = 0.01*ureg.kcal/ureg.mol
k_angle_sim = (k_angle/ureg.avogadro_constant).to("sim_energy")
print("k_bond_sim: {0:.1f}, bond_l_sim: {1:.1f}, k_angle_sim: {2:.7f}".format(
    k_bond_sim, bond_l_sim, k_angle_sim))
# Acid-Base parameters
K = 10**(-pK)
K2 = 10**(-pK2)
pKw = 14.0  # autoprotolysis constant of water
Kw = 10**(-pKw)
pHs = np.linspace(pHmin, pHmax, NUM_PHS)  # list of pH values

use_interactions(USE_WCA,USE_ELECTROSTATICS,USE_FENE,USE_BENDING,USE_DIHEDRAL_POT)
print(USE_WCA,USE_ELECTROSTATICS,USE_FENE,USE_BENDING,USE_DIHEDRAL_POT,USE_P3M)
COMPUTE_RDF = True
print_trajectory = True
if USE_ELECTROSTATICS:
    assert USE_WCA, "You can not use electrostatics without a short range repulsive potential. Otherwise oppositely charged particles could come infinitely close."

print("KT = {:4.3f}".format(KT.to('sim_energy')))
print("PARTICLE_SIZE = {:4.3f}".format(0.5 * (BJERRUM_LENGTH).to('sim_length')))
print("BJERRUM_LENGTH = {:4.3f}".format(BJERRUM_LENGTH.to('sim_length')))

print("N_salt: {0:.1f}, N_acid: {1:.1f}, N_salt/N_acid: {2:.7f}, c_salt/c_acid: {3:.7f}".format(
    N_SALT, N_ACID, 1.0*N_SALT/N_ACID, C_SALT/C_ACID))

KAPPA = np.sqrt(C_SALT.to('mol/L').magnitude)/0.304 / ureg.nm
KAPPA_REDUCED = KAPPA.to('1/sim_length').magnitude
print(f"KAPPA = {KAPPA:.3f}")
print(f"KAPPA_REDUCED = {KAPPA_REDUCED:.3f}")
print(f"Debye_length: {1. / KAPPA:.2f} = {(1. / KAPPA).to('sim_length'):.2f}")

print("Calculated values:")
print(f"BOX_L = {BOX_L:.3g} = {BOX_L.to('sim_length'):.3g}")
print(f"BOX_V  = {BOX_V:.3g} = {BOX_V.to('sim_length^3'):.3g}")
print(f"N_SALT = {N_SALT}")

# number of reaction samples per each pH value
NUM_SAMPLES = int(N_BLOCKS * DESIRED_BLOCK_SIZE / PROB_REACTION)
n_iter_prop = 20
begin_sample = 200
n_samp_iter = int((NUM_SAMPLES-begin_sample)/n_iter_prop)

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
    "0": (2*2.6 * ureg.angstrom).to("sim_length").magnitude,
    "1": (2*3.172 * ureg.angstrom).to("sim_length").magnitude,
    "2": (2*2.958 * ureg.angstrom).to("sim_length").magnitude,
    "3": (2*3.93 * ureg.angstrom).to("sim_length").magnitude,
    "4": (2*3.9624 * ureg.angstrom).to("sim_length").magnitude,
    "5": (2*3.915 * ureg.angstrom).to("sim_length").magnitude,
    "6": (2*2.6 * ureg.angstrom).to("sim_length").magnitude,
    "7": (2*3.172 * ureg.angstrom).to("sim_length").magnitude,

}

lj_epsilons = {
    "0": (((231*ureg.kelvin)*ureg.boltzmann_constant).to('sim_energy')).magnitude,
    "1": (((231*ureg.kelvin)*ureg.boltzmann_constant).to('sim_energy')).magnitude,
    "2": (((0.0726*ureg.kcal/ureg.mol)/ureg.avogadro_constant).to('sim_energy')).magnitude,
    "3": (((56.0*ureg.kelvin)*ureg.boltzmann_constant).to('sim_energy')).magnitude,
    "4": (((0.738*ureg.kcal/ureg.mol)/ureg.avogadro_constant).to('sim_energy')).magnitude,
    "5": (((0.305*ureg.kcal/ureg.mol)/ureg.avogadro_constant).to('sim_energy')).magnitude,
    "6": (((231*ureg.kelvin)*ureg.boltzmann_constant).to('sim_energy')).magnitude,
    "7": (((231*ureg.kelvin)*ureg.boltzmann_constant).to('sim_energy')).magnitude

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
system.time_step = time_step
system.cell_system.skin = 2.0
system.periodicity = [True, True, True]
np.random.seed(seed=10)  # initialize the random number generator in numpy


if USE_FENE:
    bond_pot = FeneBond(k=100, d_r_max=4.0, r_0=1.5)
else :
    bond_pot = HarmonicBond(k=100, r_0=1.5)

if USE_BENDING:
    bend=AngleHarmonic(bend=1.0, phi0=np.pi*(2.0/3.0))
    system.bonded_inter.add(bend)
if USE_DIHEDRAL_POT:
    dihedral = Dihedral(bend=1.0, mult=3, phase=np.pi*(2.0/3.0))
    system.bonded_inter.add(dihedral)


system.bonded_inter.add(bond_pot)

# create the polymer positions
polymers = espressomd.polymer.positions(n_polymers=n_poly, beads_per_chain=N_MON,bond_length=1.5, seed=23)
# add the polymer particles composed of ionizable acid groups, initially in the ionized state
n_NH2=0
n_NH=0
for polymer in polymers:
    for index,position in enumerate(polymer):
        id = len(system.part)
        if index % 3 == 0 :
            if index == 0 or index == (N_MON-1) :
                system.part.add(id = id ,pos=position, type=TYPES["A2"], q=CHARGES["A2"])
                n_NH2 = n_NH2+1
                # print(index,"A2")
            else :
                system.part.add(id = id ,pos=position, type=TYPES["A"], q=CHARGES["A"])
                n_NH = n_NH+1
                # print(index,"A")
        else :
            system.part.add(id = id ,pos=position, type=TYPES["N"], q=CHARGES["N"])
            # print(index,"N")

        #p = system.part.add(pos=position, type=TYPES["A"], q=CHARGES["A"])
        if index>0:
            system.part[id].add_bond((bond_pot, id -1))
            if USE_BENDING:
                if index > 0 and index < len(polymer) -1:
                    system.part[id].add_bond((bend,id -1, id + 1))  # Ja es crearà la seg:uent partícula
            if USE_DIHEDRAL_POT:
                if index > 0 and index < len(polymer) -2:
                    system.part[id].add_bond((dihedral, id-1, id+1, id+2))
print("number oh NH2 positioned in the polymer",n_NH2)
print("number oh NH positioned in the polymer",n_NH)
n_ACID = n_NH2 + n_NH
#system.part[1].add_bond((angle_harmonic, 0, 2))
#system.part[1:-1].add_bond((angle_harmonic, np.arange(N_MON)[:-2], np.arange(N_MON)[2:]))
max_sigma = 1.0  # en nm
min_dist = 0.0
#
try:
    energy = system.analysis.energy()
    print("Before Minimization: E_total = {:.3e}".format(energy['total']))
except:
    print('ERROR Minimization 1')

try:
    energy = system.analysis.energy()
    print("Before Minimization: E_total = {:.3e}".format(energy['total']))
except:
    print('ERROR Minimization 2')



system.integrator.set_vv()
system.minimize_energy.init(f_max=100, gamma=10.0, max_steps=100,max_displacement=max_sigma * 0.01)
Ultim_ID=system.part[:].id[-1]

for i in range(50):
    system.minimize_energy.minimize()
    if i%10==0:
        print('Position last part. {}'.format(system.part[Ultim_ID].pos))


energy = system.analysis.energy()
print("After Minimization: E_total = {:.3e}".format(energy['total']))

ADD_HIDRO_IONS = True
if ADD_HIDRO_IONS:
#add the corresponding number of B+ ions
    system.part.add(pos=np.random.random((n_ACID, 3)) * BOX_L_UNITLESS,
                    type=[TYPES["B"]] * n_ACID,
                    q=[CHARGES["B"]] * n_ACID)

    system.part.add(pos=np.random.random((n_ACID, 3)) * BOX_L_UNITLESS,
                type=[TYPES["Cl"]] * n_ACID,
                q=[CHARGES["Cl"]] * n_ACID)


# add salt ion pairs
ADD_SALT = True
if ADD_SALT:
    system.part.add(pos=np.random.random((N_SALT, 3)) * BOX_L_UNITLESS,
                type=[TYPES["Na"]] * N_SALT,
                q=[CHARGES["Na"]] * N_SALT)

    system.part.add(pos=np.random.random((N_SALT, 3)) * BOX_L_UNITLESS,
                type=[TYPES["Cl"]] * N_SALT,
                q=[CHARGES["Cl"]] * N_SALT)


# Check charge neutrality
assert np.abs(np.sum(system.part[:].q)) < 1E-10

# Set the parameters
if USE_WCA:
    for type_1, type_2 in ((x, y) for x in TYPES.values() for y in TYPES.values()):
        lj_sig = combination_rule_sigma("Berthelot", lj_sigmas[str(type_1)], lj_sigmas[str(type_2)])
        lj_eps = combination_rule_epsilon("Lorentz", lj_epsilons[str(type_1)], lj_epsilons[str(type_2)])
        system.non_bonded_inter[type_1, type_2].wca.set_params(epsilon=lj_eps, sigma=0.5*lj_sig)

    # relax the overlaps with steepest descent
    system.integrator.set_steepest_descent(f_max=0, gamma=0.1, max_displacement=0.1)
    system.integrator.run(1000)
    system.integrator.set_vv()  # to switch back to velocity Verlet

# save_vxyz(system,'polymer1.xyz',mode='a',aplicar_PBC=True)
# add thermostat and short integration to let the system relax
system.thermostat.set_langevin(kT=KT.to("sim_energy").magnitude, gamma=1.0, seed=7)
system.integrator.run(steps=1000)

#outfile2 = 'polymer2.xyz'
#save_vxyz(system,outfile2,mode='w',aplicar_PBC=True)


# Prefactor in reduced units, if you want to make production run, better with a lower accuracy
if USE_ELECTROSTATICS:
    if USE_P3M:
        coulomb = espressomd.electrostatics.P3M(prefactor = (BJERRUM_LENGTH * KT / (ureg.elementary_charge ** 2)
                   ).to("sim_length * sim_energy / sim_charge^2").magnitude,
                                                accuracy=1e-4)
    else:
        coulomb = espressomd.electrostatics.DH(prefactor = (BJERRUM_LENGTH * KT / (ureg.elementary_charge ** 2)
                   ).to("sim_length * sim_energy / sim_charge^2").magnitude,
                                               kappa = KAPPA_REDUCED,
                                               r_cut = 1. / KAPPA_REDUCED)

    system.actors.add(coulomb)
else:
    # this speeds up the simulation of dilute systems with small particle numbers
    system.cell_system.set_n_square()


#p3m.tune()

system.integrator.run(steps=1000)

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

RE.set_non_interacting_type(len(TYPES)) # this parameter helps speed up the calculation in an interacting system
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
RE2.set_non_interacting_type(len(TYPES)) # this parameter helps speed up the calculation in an interacting system
print(RE.get_status())
print(RE2.get_status())
system.setup_type_map([0,1,2,3,4,5,6,7])

# Calculate the averaged rdfs
r_bins = 300
r_min = 0.0
r_max = system.box_l[0] / 2.0

rdf_HA_Cl_avg = np.zeros((len(pHs), r_bins))
rdf_HA2_Cl_avg = np.zeros((len(pHs), r_bins))
rdf_HA_HA_avg = np.zeros((len(pHs), r_bins))
rdf_HA_N_avg = np.zeros((len(pHs), r_bins))
rdf_HA_Na_avg = np.zeros((len(pHs), r_bins))
rdf_Na_Cl_avg = np.zeros((len(pHs), r_bins))
qdist = np.zeros((len(pHs), N_MON))

print('Index of particles:')
indices_TYPE_HA=np.where(system.part[:].type==TYPES["A"])[0]
indices_TYPE_HA2=np.where(system.part[:].type==TYPES["A2"])[0]
indices_TYPE_N=np.where(system.part[:].type==TYPES["N"])[0]


print('HA ',indices_TYPE_HA)
print('HA2 ',indices_TYPE_HA2)
print('N ',indices_TYPE_N)


llista_centres=[]
for ele in indices_TYPE_HA:
    llista_centres.append(ele-1)  # Rotació previ al OJ
    llista_centres.append(ele)    # Rotació després del OJ

llista_ids_dih=[]
for ele in llista_centres:
    llista_ids_dih.append([ele-1,ele,ele+1,ele+2])
dihAc=tp.dihedres(llista_ids_dih,system)
dihAc.acum()  # Valors a temps 0

def equilibrate_pH():
    RE.reaction(reaction_steps=20 * N_ACID + 1)
    RE2.reaction(reaction_steps=40 * N_ACID + 1)
    if USE_WCA:
        system.integrator.run(steps=1000)

times = np.zeros(NUM_SAMPLES)
T_inst = np.zeros(NUM_SAMPLES)
e_kin = np.zeros(NUM_SAMPLES)

outfile2 = "traj_polymer.xyz"
def perform_sampling(npH,num_samples, num_As: np.ndarray, num_As2: np.ndarray, num_B: np.ndarray, rad_gyr: np.ndarray, end_2end: np.ndarray,en_total: np.ndarray):
    system.time = 0.

    global rdf_HA_Cl_avg, rdf_HA2_Cl_avg,rdf_HA_HA_avg,rdf_HA_N_avg,rdf_HA_Na_avg,rdf_Na_Cl_avg, r1, r2, r3, r4, r5, r6, c
    c = 0
    k = 0
    for i in range(num_samples):
        if np.random.random() < PROB_REACTION:
            # should be at least one reaction attempt per particle
            RE.reaction(reaction_steps=n_NH + 1)

        if np.random.random() < PROB_REACTION*(2.0/(N_MON-2)):
            # should be at least one reaction attempt per particle
            RE2.reaction(reaction_steps=n_NH2 + 1)

        #if USE_WCA:
        system.integrator.run(steps=1000)
        num_As[i] = system.number_of_particles(type=TYPES["A"])
        num_As2[i] = system.number_of_particles(type=TYPES["A2"])
        num_B[i] = system.number_of_particles(type=TYPES["B"])
        c = c + 1

        for n in range(N_MON):
            qn = system.part[n].q
            qdist[npH,n] = qdist[npH,n] + qn

        if i%n_iter_prop==0 and i>begin_sample:
            k = k + 1
            if COMPUTE_RDF:
                r1, rdf_HA_Cl = system.analysis.rdf(rdf_type="rdf", type_list_a=[TYPES["HA"]], type_list_b=[TYPES["Cl"]], r_min=r_min, r_max=r_max, r_bins=r_bins)
                r2, rdf_HA2_Cl = system.analysis.rdf(rdf_type="rdf", type_list_a=[TYPES["HA2"]], type_list_b=[TYPES["Cl"]], r_min=r_min, r_max=r_max, r_bins=r_bins)
                r3, rdf_HA_HA = system.analysis.rdf(rdf_type="rdf", type_list_a=[TYPES["HA"]], type_list_b=[TYPES["HA"]], r_min=r_min, r_max=r_max, r_bins=r_bins)
                r4, rdf_HA_N = system.analysis.rdf(rdf_type="rdf", type_list_a=[TYPES["HA"]], type_list_b=[TYPES["N"]], r_min=r_min, r_max=r_max, r_bins=r_bins)
                r5, rdf_HA_Na = system.analysis.rdf(rdf_type="rdf", type_list_a=[TYPES["HA"]], type_list_b=[TYPES["Na"]], r_min=r_min, r_max=r_max, r_bins=r_bins)
                r6, rdf_Na_Cl = system.analysis.rdf(rdf_type="rdf", type_list_a=[TYPES["Na"]], type_list_b=[TYPES["Cl"]], r_min=r_min, r_max=r_max, r_bins=r_bins)
                rdf_HA_Cl_avg[npH,:] += rdf_HA_Cl/n_samp_iter
                rdf_HA2_Cl_avg[npH,:] += rdf_HA2_Cl/n_samp_iter
                rdf_HA_HA_avg[npH,:] += rdf_HA_HA/n_samp_iter
                rdf_HA_N_avg[npH,:] += rdf_HA_N/n_samp_iter
                rdf_HA_Na_avg[npH,:] += rdf_HA_Na/n_samp_iter
                rdf_Na_Cl_avg[npH,:] += rdf_Na_Cl/n_samp_iter

            rad_gyr[i] = system.analysis.calc_rg(chain_start=0,number_of_chains=1,chain_length=N_MON)[0]
            end_2end[i] = system.analysis.calc_re(chain_start=0,number_of_chains=1,chain_length=N_MON)[0]
        times[i] = system.time
        energy = system.analysis.energy()
        en_total[i] = energy['total']
        e_kin[i] = energy['kinetic']
        T_inst[i] = 2. / 3. * e_kin[i] / (n_poly*N_MON + 2*N_ACID + 2*N_SALT)
        if i%200 ==0:
            print("time: {0:.2f}, n_NH: {1:.2f}, n_NH2: {2:.2f}, pH: {3:.2f}, energy: {4:.3f}, Temperature: {5:.3f}".format(
                times[i], num_As[i], num_As2[i],pHs[npH],en_total[i],T_inst[i]))

            # Dihedres
        dihAc.acum()


        #print(times)
        if print_trajectory and i%100==0:
            tp.save_vxyz(system,outfile2,mode='a',aplicar_PBC=True)



# Observables for cheeck that the simulation is consistent with the LAngevin thermostat parameters

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
    print(f"measured number of NH: {np.mean(num_As_at_each_pH[ipH]):.2f}, (ideal: {n_NH*ideal_alpha(pH, pK):.2f})")
    print(f"measured number of NH2: {np.mean(num_As2_at_each_pH[ipH]):.2f}, (ideal: {n_NH2*ideal_alpha(pH, pK2):.2f})")
    print(f"measured number of NH2+NH: {np.mean(num_As2_at_each_pH[ipH])+np.mean(num_As_at_each_pH[ipH]):.2f}, (ideal: {n_NH*ideal_alpha(pH, pK)+n_NH2*ideal_alpha(pH, pK2):.2f})")
    print(f"measured number of B+: {np.mean(num_B_at_each_pH[ipH]):.2f})")
    #tp.save_vxyz(system,outfile2,mode='a',aplicar_PBC=True)

    print(f"radius of gyration : {np.mean(rad_gyr_at_each_pH[ipH]):.2f}")
    print(f"end to end distance : {np.mean(end_2e_at_each_pH[ipH]):.2f}")
    print(f"energy : {np.mean(energy_at_each_pH[ipH]):.2f}")

if print_trajectory :
    outfile3 = "polymer_traj_converted.xyz"
    tp.convert_vxyz(outfile2,outfile3)

# To check that the Langevin thermostat works correctly we compute the instantaneous temperature at each sample
# ionization degree alpha calculated from the Henderson-Hasselbalch equation for an ideal system
qdist = qdist / c

dihAc.save('dades_dih.dat')

temps_dh=dihAc.get_temps()
dh=dihAc.get_acum()
temps_dh=np.array(temps_dh)
dh=np.array(dh)

index_dh=dihAc.get_index()

plt.figure(figsize=(10, 6), dpi=80)

for i_rot in range(0,len(index_dh),4):
#for i_rot in range(10,15,1):

    indices=index_dh[i_rot]
    noms_tipus=''
    for i in range(4):  # 4 index
        noms_tipus=noms_tipus+tp.info_part[system.part[indices[i]].type].nom+'-'
    noms_tipus=noms_tipus[:-1] # treure l'ultim guionet
    #print(noms_tipus)

    plt.plot(temps_dh,dh[:,i_rot],'.',label='{} {}'.format(index_dh[i_rot],noms_tipus))
plt.ylabel(r'dihedral', fontsize=16)
#plt.legend(fontsize=16)
plt.savefig('dihedral.png')


# estimate the statistical error and the autocorrelation time
av_num_As, err_num_As, tau, block_size = block_analyze(num_As_at_each_pH, N_BLOCKS)
av_num_As2, err_num_As2, tau5, block_size = block_analyze(num_As2_at_each_pH, N_BLOCKS)
av_num_B, err_num_B, tau6, block_size = block_analyze(num_B_at_each_pH, N_BLOCKS)

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
av_alpha = (av_num_As  + av_num_As2 ) / n_ACID
err_alpha = (err_num_As + err_num_As2) / n_ACID
NH_alpha = (av_num_As) / n_NH
err_NH_alpha = (err_num_As) / n_NH
NH2_alpha = (av_num_As2) / n_NH2
err_NH2_alpha = (err_num_As2) / n_NH2


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


if COMPUTE_RDF:
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
if COMPUTE_RDF:
    for k in range(len(pHs)):
        with open('rdf_HA_Cl_avg'+str(k)+'.dat','w') as rdf_fp1:
            for i in range(r_bins):
                rdf_fp1.write("%1.5e %1.5e\n" % (r1[i], rdf_HA_Cl_avg[k][i]))

    for k in range(len(pHs)):
        with open('rdf_HA2_Cl_avg'+str(k)+'.dat','w') as rdf_fp2:
            for i in range(r_bins):
                rdf_fp2.write("%1.5e %1.5e\n" % (r2[i], rdf_HA2_Cl_avg[k][i]))

    for k in range(len(pHs)):
        with open('rdf_HA_HA_avg'+str(k)+'.dat','w') as rdf_fp3:
            for i in range(r_bins):
                rdf_fp3.write("%1.5e %1.5e\n" % (r3[i], rdf_HA_HA_avg[k][i]))

    for k in range(len(pHs)):
        with open('rdf_HA_N_avg'+str(k)+'.dat','w') as rdf_fp4:
            for i in range(r_bins):
                rdf_fp4.write("%1.5e %1.5e\n" % (r4[i], rdf_HA_N_avg[k][i]))

    for k in range(len(pHs)):
        with open('rdf_HA_Na_avg'+str(k)+'.dat','w') as rdf_fp5:
            for i in range(r_bins):
                rdf_fp5.write("%1.5e %1.5e\n" % (r1[i], rdf_HA_Na_avg[k][i]))

    for k in range(len(pHs)):
        with open('rdf_Na_Cl_avg'+str(k)+'.dat','w') as rdf_fp6:
            for i in range(r_bins):
                rdf_fp6.write("%1.5e %1.5e\n" % (r1[i], rdf_Na_Cl_avg[k][i]))


with open('alpha_avg.dat','w') as avg_alpha:
    for ipH, pH in enumerate(pHs):
        avg_alpha.write("%1.5e %1.5e\n" % (pH, av_alpha[ipH]))

with open('rad_gyr.dat','w') as rad_gyr:
    for ipH, pH in enumerate(pHs):
        rad_gyr.write("%1.5e %1.5e\n" % (pH, av_rad_gyr[ipH]))

with open('end_2e.dat','w') as end_2e:
    for ipH, pH in enumerate(pHs):
        end_2e.write("%1.5e %1.5e\n" % (pH, av_end_2e[ipH]))

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
plt.savefig('ionic_strength.png')

with open('ideal_ionic.dat','w') as ideal_ionic:
    for ipH, pH in enumerate(full_pH_range):
        ideal_ionic.write("%1.5e %1.5e\n" % (pH, ideal_ionic_strength[ipH]))

with open('cpH_ionic.dat','w') as cpH_ionic:
    for ipH, pH in enumerate(full_pH_range):
        cpH_ionic.write("%1.5e %1.5e\n" % (pH, cpH_ionic_strength[ipH]))

with open('cpH_ionic.dat','w') as cpH_ionic_measured:
    for ipH, pH in enumerate(pHs):
        cpH_ionic_measured.write("%1.5e %1.5e %1.5e\n" % (pH, cpH_ionic_strength_measured[ipH],cpH_error_ionic_strength_measured[ipH]))
