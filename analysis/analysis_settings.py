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
RANGE_MULTIPLICATION_FACTOR = np.logspace(np.log10(1), np.log10(10), 15)
# We used log scale because concentrations are near 1
RANGE_CARRY = np.logspace(np.log10(0.1), np.log10(10), num=15)

RANGE_FEED_TYPE = [FEEDBACK_POSITIVE, FEEDBACK_NEGATIVE]
RANGE_SUBSTRATE = [I_PMPI, I_PI4P, I_PIP2, I_DAG, I_PMPA, I_ERPA, I_CDPDAG,
                   I_ERPI]
RANGE_ENZYMES = [E_PITP, E_PI4K, E_PIP5K, E_PLC, E_DAGK, E_LAZA, E_PATP, E_CDS,
                 E_PIS, E_SINK, E_SOURCE]

COLORS_PRIMARY = ["#CDDC39", "#E91E63", "#9C27B0", "#673AB7",
                  "#3F51B5", "#2196F3", "#03A9F4", "#00BCD4", "#009688",
                  "#4CAF50", "#F44336",
                  "#8BC34A"]

COLORS_SECONDARY = ["#F0F4C3", "#F8BBD0", "#E1BEE7", "#D1C4E9",
                    "#C5CAE9", "#BBDEFB", "#B3E5FC", "#B2EBF2", "#B2DFDB",
                    "#C8E6C9", "#FFCDD2",
                    "#DCEDC8"]
