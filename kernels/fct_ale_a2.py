
from kernel_tuner import tune_kernel
import numpy
import argparse

def generate_code(tuning_parameters):
    pass

def reference(elements, levels, max_levels, nodes, UV_rhs, fct_ttf_max, fct_ttf_min, real_type):
    for element in range(0, elements):
        for level in range(0, levels[element]):
            item = (element * max_levels * 2) + (level * 2)
            UV_rhs[item] = max(fct_ttf_max[(nodes[element * 3] * max_levels) + level], fct_ttf_max[(nodes[(element * 3) + 1] * max_levels) + level], fct_ttf_max[(nodes[(element * 3) + 2] * max_levels) + level])
            UV_rhs[item + 1] = min(fct_ttf_min[(nodes[element * 3] * max_levels) + level], fct_ttf_min[(nodes[(element * 3) + 1] * max_levels) + level], fct_ttf_min[(nodes[(element * 3) + 2] * max_levels) + level])
        if levels[element] <= max_levels - 1:
            for level in range(levels[element], max_levels - 1):
                item = (element * max_levels * 2) + (level * 2)
                if real_type == "float":
                    UV_rhs[item] = numpy.finfo(numpy.float32).min
                    UV_rhs[item + 1] = numpy.finfo(numpy.float32).max
                elif real_type == "double":
                    UV_rhs[item] = numpy.finfo(numpy.float64).min
                    UV_rhs[item + 1] = numpy.finfo(numpy.float64).max
                else:
                    raise ValueError

def verify(control_data, data, atol=None):
    return numpy.allclose(control_data, data, atol)

def tune(nodes, max_levels, max_tile, real_type):
    pass

def parse_command_line():
    parser = argparse.ArgumentParser(description="FESOM2 FCT ALE A2")
    parser.add_argument("--elements", help="The number of elements.", type=int, required=True)
    parser.add_argument("--max_levels", help="The maximum number of vertical levels per element.", type=int, required=True)
    parser.add_argument("--max_tile", help="The maximum tiling factor.", type=int, default=2)
    parser.add_argument("--real_type", help="The floating point type to use.", choices=["float", "double"], type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    command_line = parse_command_line()
    results = tune(command_line.elements, command_line.max_levels, command_line.max_tile, command_line.real_type)
    best_configuration = min(results, key=lambda x : x["time"])
    print("/* Block size X: {} */".format(best_configuration["block_size_x"]))
    print(generate_code(best_configuration))