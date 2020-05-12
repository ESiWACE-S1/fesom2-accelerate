
from kernel_tuner import tune_kernel
import numpy
import argparse

def generate_code(tuning_parameters):
    pass

def reference(vlimit, nodes, levels, max_levels, uv_rhs, fct_ttf_max, fct_ttf_min, fct_lo, real_type):
    tvert_max = 0
    tvert_min = 0
    if vlimit == 1:
        for node in range(0, nodes):
            for level in range(0, levels[node] - 1):
                tvert_max[level] = None
                tvert_min[level] = None
            # Surface level
            fct_ttf_max[(node * max_levels)] = tvert_max[0] - fct_lo[(node * max_levels)]
            fct_ttf_min[(node * max_levels)] = tvert_min[0] - fct_lo[(node * max_levels)]
            # Intermediate levels
            for level in range(1, levels[node] - 2):
                temp = max(tvert_max[level - 1], tvert_max[level])
                temp = max(temp, tvert_max[level + 1])
                fct_ttf_max[(node * max_levels) + level] = temp - fct_lo[(node * max_levels) + level]
                temp = min(tvert_min[level - 1], tvert_min[level])
                temp = min(temp, tvert_min[level + 1])
                fct_ttf_min[(node * max_levels) + level] = temp - fct_lo[(node * max_levels) + level]
            # Bottom level
            fct_ttf_max[(node * max_levels) + levels[node] - 1] = tvert_max[levels[node] - 1] - fct_lo[(node * max_levels) + levels[node] - 1]
            fct_ttf_min[(node * max_levels) + levels[node] - 1] = tvert_min[levels[node] - 1] - fct_lo[(node * max_levels) + levels[node] - 1]
    elif vlimit == 2:
        pass
    elif vlimit == 3:
        pass
    else:
        raise ValueError

def verify(control_data, data, atol=None):
    return numpy.allclose(control_data, data, atol)

def tune(nodes, max_levels, vlimit, max_tile, real_type):
    pass

def parse_command_line():
    parser = argparse.ArgumentParser(description="FESOM2 FCT ALE A3")
    parser.add_argument("--nodes", help="The number of nodes.", type=int, required=True)
    parser.add_argument("--max_levels", help="The maximum number of vertical levels per element.", type=int, required=True)
    parser.add_argument("--vlimit", help="The version of vertical limit.", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--max_tile", help="The maximum tiling factor.", type=int, default=2)
    parser.add_argument("--real_type", help="The floating point type to use.", choices=["float", "double"], type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    command_line = parse_command_line()
    results = tune(command_line.nodes, command_line.max_levels, command_line.vlimit, command_line.max_tile, command_line.real_type)
    best_configuration = min(results, key=lambda x : x["time"])
    print("/* Block size X: {} */".format(best_configuration["block_size_x"]))
    print(generate_code(best_configuration))