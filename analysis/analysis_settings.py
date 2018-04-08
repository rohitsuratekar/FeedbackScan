import numpy as np

from constants.namespace import *

SYSTEM = S_OPEN_2
PERCENTAGE_DEPLETION = 85

# First recovery point is 16 which is immediately after 15% depletion
# 76.5 is 90% of whatever is left after 15% depletion
RECOVERY_POINTS = [20, 30, 50, 76.5, 90, 100]  # Get data at these points

# Initialization and some base constants and their ranges
RANGE_HILL_COEFFICIENT = [0.5, 1, 2]

# We used log scale because for larger multiplication factor lipid levels
# will saturate in the MM type kinetics
RANGE_MULTIPLICATION_FACTOR = np.logspace(np.log10(1), np.log10(10), 5)
# We used log scale because concentrations are near 1
RANGE_CARRY = np.logspace(np.log10(0.1), np.log10(10), num=5)

RANGE_FEED_TYPE = [FEEDBACK_POSITIVE, FEEDBACK_NEGATIVE]
RANGE_SUBSTRATE = [I_PMPI, I_PI4P, I_PIP2, I_DAG, I_PMPA, I_ERPA, I_CDPDAG,
                   I_ERPI]
RANGE_ENZYMES = [E_PITP, E_PI4K, E_PIP5K, E_PLC, E_DAGK, E_LAZA, E_PATP, E_CDS,
                 E_PIS, E_SINK, E_SOURCE]

# Timings
recovery_time = np.linspace(0, 100, 2000)
