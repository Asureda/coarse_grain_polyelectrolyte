import numpy as np
from espressomd import observables, accumulators, analyze
import espressomd.reaction_ensemble

def use_interactions(WCA,ELECTROSTATICS,FENE,BENDING,DIHEDRAL):
    USE_WCA = WCA
    USE_ELECTROSTATICS = ELECTROSTATICS
    USE_FENE = FENE
    USE_BENDING = BENDING
    USE_DIHEDRAL_POT = DIHEDRAL
    return USE_WCA, USE_ELECTROSTATICS, USE_FENE, USE_BENDING, USE_DIHEDRAL_POT

# USE_WCA, USE_ELECTROSTATICS, USE_FENE, USE_BENDING, USE_DIHEDRAL_POT = use_interactions(True,False,True,True,False)

def ideal_alpha(pH, pK):
    return 1. / (1 + 10**(pK - pH))


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
