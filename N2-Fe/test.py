import numpy as np

print(np.outer(np.concatenate((-np.ones(72)/36, np.ones(2))), np.array([0, 1, 2])))
