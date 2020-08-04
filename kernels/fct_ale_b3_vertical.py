
from kernel_tuner import tune_kernel
import numpy
import argparse

def reference(nodes, levels, max_levels, fct_adf_v, fct_plus, fct_minus, iter_yn):
    for node in range(0, nodes):
        ae = 1.0
        flux = fct_adf_v[(node * max_levels)]
        if flux >= 0:
            ae = min(ae, fct_plus[(node * max_levels)])
        else:
            ae = min(ae, fct_minus[(node * max_levels)])
        fct_adf_v[(node * max_levels)] = ae * fct_adf_v[(node * max_levels)]
        for level in range(1, levels[node] - 1):
            ae = 1.0
            flux = fct_adf_v[(node * max_levels) + level]
            if flux >= 0:
                ae = min(ae, fct_minus[(node * max_levels) + (level - 1)])
                ae = min(ae, fct_plus[(node * max_levels) + level])
            else:
                ae = min(ae, fct_plus[(node * max_levels) + (level - 1)])
                ae = min(ae, fct_minus[(node * max_levels) + level])
            fct_adf_v[(node * max_levels) + level] = ae * fct_adf_v[(node * max_levels) + level]
