
from kernel_tuner import tune_kernel
import numpy
import argparse

def generate_code(tuning_parameters):
    pass

def reference(elements, levels, max_levels, nodes, UV_rhs, fct_ttf_max, fct_ttf_min, real_type):
    pass

def verify(control_data, data, atol=None):
    return numpy.allclose(control_data, data, atol)

def tune(elements, nodes, max_levels, max_tile, real_type):
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
    results = tune(command_line.elements, command_line.nodes, command_line.max_levels, command_line.max_tile, command_line.real_type)
    best_configuration = min(results, key=lambda x : x["time"])
    print("/* Block size X: {} */".format(best_configuration["block_size_x"]))
    print(generate_code(best_configuration))