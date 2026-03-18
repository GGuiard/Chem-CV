import numpy as np

# TODO:
# - make functions understand that they need to use a parameter only if its given (ex: weights, masks)
# - for bootstrap and block add the possibility to choose specify a function to apply to the data

def logw_to_w(logw, KBT):
    return np.exp(logw/KBT)

def fes(pop, KBT):
    return -KBT*np.ma.log(pop)

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

def population(data, bins, weights=None, use_weight=False): 
    if use_weight:
        pop = np.histogram(data, bins, density=True, weights=weights) # test if weights=None is okay
    else:
        pop = np.histogram(data, bins, density=True)
    return pop

def bootstrap(data, nb_bootstraps, weights=None, use_weight=False):
    N = len(data)
    bootstrap_size = N//nb_bootstraps
    data_bootstraps = np.empty(nb_bootstraps, dtype=np.float64)
    for i in range(nb_bootstraps):
        indexes = np.random.randint(N, bootstrap_size)
        if use_weight:
            data_bootstraps[i] = np.average(data[indexes], weights[indexes])
        else:
            data_bootstraps[i] = np.average(data[indexes])
    av, std = np.average(data_bootstraps), np.std(data_bootstraps)
    return av, std, data_bootstraps

def bootstrap_pop(data, bins, nb_bootstraps, weights=None, use_weight=False):
    N, nb_bins = len(data), len(bins)
    bootstrap_size = N//nb_bootstraps
    pop_bootstraps = np.empty((nb_bootstraps, nb_bins), dtype=np.float64)
    for i in range(nb_bootstraps):
        indexes = np.random.randint(N, bootstrap_size)
        if use_weight:
            pop_bootstraps[i] = population(data[indexes], bins[indexes], weights[indexes], use_weight)
        else:
            pop_bootstraps[i] = population(data[indexes], bins[indexes])
    av, std = np.average(pop_bootstraps, axis=0), np.std(pop_bootstraps, axis=0)
    return av, std, pop_bootstraps

def block(data, nb_blocks, weights=None, use_weight=False):
    N = len(data)
    block_size = N//nb_blocks
    data_blocks = np.empty(nb_blocks, dtype=np.float64)
    for i in range(nb_blocks):
        indexes = np.arange(i*block_size, (i+1)*block_size)
        if use_weight:
            data_blocks[i] = np.average(data[indexes], weights[indexes])
        else:
            data_blocks[i] = np.average(data[indexes])
    av, std = np.average(data_blocks), np.std(data_blocks)
    return av, std, data_blocks

def block_pop(data, bins, nb_blocks, weights=None, use_weight=False):
    N, nb_bins = len(data), len(bins)
    block_size = N//nb_blocks
    pop_blocks = np.empty((nb_blocks, nb_bins), dtype=np.float64)
    for i in range(nb_blocks):
        indexes = np.arange(i*block_size, (i+1)*block_size)
        if use_weight:
            pop_blocks[i] = population(data[indexes], bins[indexes], weights[indexes], use_weight)
        else:
            pop_blocks[i] = population(data[indexes], bins[indexes])
    av, std = np.average(pop_blocks, axis=0), np.std(pop_blocks, axis=0)
    return av, std, pop_blocks

def error_fes(pop_list, KBT):
    fes_list = fes(pop_list, KBT)
    av, std = np.average(fes_list, axis=0), np.std(fes_list, axis=0)
    return av, std, fes_list