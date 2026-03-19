import numpy as np
from ase import units

# TODO:
#   - make functions understand that they need to use a parameter only if its given (ex: weights, masks, 2D)
#   - for bootstrap and block add the possibility to choose specify a function to apply to the data
#   - be initialized with temp
#   - generalize bootstrap block and pop to N dim

def Emec(Epot, Ekin):
    Emec = Epot + Ekin
    av, std = np.average(Emec), np.std(Emec)
    return Emec, av, std

def T(Ekin, nb_atoms):
    T = Ekin/(1.5*nb_atoms*units.kB)
    av, std = np.average(T), np.std(T)
    return T, av, std

def logw_to_w(logw, kT):
    return np.exp(logw/kT)

def fes(pop, kT):
    return -kT*np.ma.log(pop)

def bin_to_grid(bins):
    return (bins[1:]+bins[:-1])/2

def cum_average(data, weights=None, use_weights=False):
    if use_weights:
        av, av2 = np.cumsum(data*weights)/np.cumsum(weights), np.cumsum(data**2)/np.cumsum(weights)
        std = av2-av**2
    else:
        N = len(data)
        N_list = np.arange(1,N+1)
        av, av2 = np.cumsum(data)/N_list, np.cumsum(data**2)/N_list
        std = av2-av**2
    return av, std

def population(data, bins, weights=None, use_weights=False): 
    if use_weights:
        pop = np.histogram(data, bins, density=True, weights=weights)[0] # test if weights=None is okay
    else:
        pop = np.histogram(data, bins, density=True)[0]
    return pop

def population_2d(data1, data2, bins, weights=None, use_weights=False): 
    if use_weights:
        pop = np.histogram2d(data1, data2, bins, density=True, weights=weights)[0] # test if weights=None is okay
    else:
        pop = np.histogram2d(data1, data2, bins, density=True)[0]
    return pop

def bootstrap(data, nb_bootstraps, weights=None, use_weights=False):
    N = len(data)
    bootstrap_size = N//nb_bootstraps
    data_bootstraps = np.empty(nb_bootstraps, dtype=np.float64)
    for i in range(nb_bootstraps):
        indexes = np.random.randint(0, N, bootstrap_size)
        if use_weights:
            data_bootstraps[i] = np.average(data[indexes], weights[indexes])
        else:
            data_bootstraps[i] = np.average(data[indexes])
    av, std = np.average(data_bootstraps), np.std(data_bootstraps)
    return av, std, data_bootstraps

def bootstrap_pop(data, bins, nb_bootstraps, weights=None, use_weights=False):
    N, nb_bins = len(data), len(bins)
    bootstrap_size = N//nb_bootstraps
    pop_bootstraps = np.empty((nb_bootstraps, nb_bins-1), dtype=np.float64)
    for i in range(nb_bootstraps):
        indexes = np.random.randint(0, N, bootstrap_size)
        if use_weights:
            pop_bootstraps[i] = population(data[indexes], bins, weights[indexes], use_weights)
        else:
            pop_bootstraps[i] = population(data[indexes], bins)
    av, std = np.average(pop_bootstraps, axis=0), np.std(pop_bootstraps, axis=0)
    return av, std, pop_bootstraps

def bootstrap_pop_2d(data1, data2, bins, nb_bootstraps, weights=None, use_weights=False):
    N, nb_bins1, nb_bins2 = len(data1), len(bins[0]), len(bins[1])
    bootstrap_size = N//nb_bootstraps
    pop_bootstraps = np.empty((nb_bootstraps, nb_bins1-1, nb_bins2-1), dtype=np.float64)
    for i in range(nb_bootstraps):
        indexes = np.random.randint(0, N, bootstrap_size)
        if use_weights:
            pop_bootstraps[i] = population_2d(data1[indexes], data2[indexes], bins, weights[indexes], use_weights)
        else:
            pop_bootstraps[i] = population_2d(data1[indexes], data2[indexes], bins)
    av, std = np.average(pop_bootstraps, axis=0), np.std(pop_bootstraps, axis=0)
    return av, std, pop_bootstraps

def block(data, nb_blocks, weights=None, use_weights=False):
    N = len(data)
    block_size = N//nb_blocks
    data_blocks = np.empty(nb_blocks, dtype=np.float64)
    for i in range(nb_blocks):
        indexes = np.arange(i*block_size, (i+1)*block_size)
        if use_weights:
            data_blocks[i] = np.average(data[indexes], weights[indexes])
        else:
            data_blocks[i] = np.average(data[indexes])
    av, std = np.average(data_blocks), np.std(data_blocks)
    return av, std, data_blocks

def block_pop(data, bins, nb_blocks, weights=None, use_weights=False):
    N, nb_bins = len(data), len(bins)
    block_size = N//nb_blocks
    pop_blocks = np.empty((nb_blocks, nb_bins-1), dtype=np.float64)
    for i in range(nb_blocks):
        indexes = np.arange(i*block_size, (i+1)*block_size)
        if use_weights:
            pop_blocks[i] = population(data[indexes], bins, weights[indexes], use_weights)
        else:
            pop_blocks[i] = population(data[indexes], bins)
    av, std = np.average(pop_blocks, axis=0), np.std(pop_blocks, axis=0)
    return av, std, pop_blocks

def error_fes(pop_list, kT):
    fes_list = fes(pop_list, kT)
    av, std = np.average(fes_list, axis=0), np.std(fes_list, axis=0)
    return av, std, fes_list