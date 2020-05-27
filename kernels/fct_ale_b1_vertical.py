
from kernel_tuner import tune_kernel
import numpy
import argparse

def generate_code(tuning_parameters):
    code = \
        "__global__ void fct_ale_b1_vertical(const int maxLevels, const int * __restrict__ nLevels, const <%REAL_TYPE%> * __restrict__ fct_adf_v, <%REAL_TYPE%> * __restrict__ fct_plus, <%REAL_TYPE%> * __restrict__ fct_minus)\n" \
        "{\n" \
        "const <%INT_TYPE%> node = (blockIdx.x * maxLevels);\n" \
        "\n" \
        "for ( <%INT_TYPE%> level = threadIdx.x; level < nLevels[blockIdx.x] - 1; level += <%BLOCK_SIZE%> )\n" \
        "{\n" \
        "<%REAL_TYPE%> fct_adf_v_level = 0.0;\n" \
        "<%REAL_TYPE%> fct_adf_v_nlevel = 0.0;\n" \
        "<%COMPUTE_BLOCK%>" \
        "}\n" \
        "}\n"
    compute_block = \
        "fct_adf_v_level = fct_adf_v[node + level + <%OFFSET%>];\n" \
        "fct_adf_v_nlevel = fct_adf_v[node + (level + 1) + <%OFFSET%>];\n" \
        "fct_plus[node + level + <%OFFSET%>] = 0.0 + (fmax(0.0, fct_adf_v_level) + fmax(0.0, -fct_adf_v_nlevel));\n" \
        "fct_minus[node + level + <%OFFSET%>] = 0.0 + (fmin(0.0, fct_adf_v_level) + fmin(0.0, -fct_adf_v_nlevel));\n"
    if tuning_parameters["tiling_x"] > 1:
        code = code.replace("<%BLOCK_SIZE%>", str(tuning_parameters["block_size_x"] * tuning_parameters["tiling_x"]))
    else:
        code = code.replace("<%BLOCK_SIZE%>", str(tuning_parameters["block_size_x"]))
    compute = str()
    for tile in range(0, tuning_parameters["tiling_x"]):
        if tile == 0:
            compute = compute + compute_block.replace(" + <%OFFSET%>", "")
        else:
            offset = tuning_parameters["block_size_x"] * tile
            compute = compute + "if (level + {} < nLevels[blockIdx.x])\n{{\n{}}}\n".format(str(offset), compute_block.replace("<%OFFSET%>", str(offset)))
    code = code.replace("<%COMPUTE_BLOCK%>", compute)
    code = code.replace("<%INT_TYPE%>", tuning_parameters["int_type"].replace("_", " "))
    code = code.replace("<%REAL_TYPE%>", tuning_parameters["real_type"])
    return code

def reference(nodes, levels, max_levels, fct_adf_v, fct_plus, fct_minus):
    for node in range(0, nodes):
        for level in range(0, levels[node] - 1):
            item = (node * max_levels) + level
            fct_plus[item] = 0.0
            fct_minus[item] = 0.0
    for node in range(0, nodes):
        for level in range(0, levels[node] - 1):
            item = (node * max_levels) + level
            fct_plus[item] = (max(0.0, fct_adf_v[item]) + max(0.0, -fct_adf_v[(node * max_levels) + level + 1]))
            fct_minus[item] = (min(0.0, fct_adf_v[item]) + min(0.0, -fct_adf_v[(node * max_levels) + level + 1]))

def tune(nodes, max_levels, max_tile, real_type):
    numpy_real_type = None
    if real_type == "float":
        numpy_real_type = numpy.float32
    elif real_type == "double":
        numpy_real_type = numpy.float64
    else:
        raise ValueError
    # Tuning and code generation parameters
    tuning_parameters = dict()
    tuning_parameters["int_type"] = ["unsigned_int", "int"]
    tuning_parameters["real_type"] = [real_type]
    tuning_parameters["max_levels"] = [str(max_levels)]
    tuning_parameters["block_size_x"] = [32 * i for i in range(1, 33)]
    tuning_parameters["tiling_x"] = [i for i in range(1, max_tile)]
    constraints = list()
    constraints.append("block_size_x * tiling_x <= max_levels")
    # Memory allocation and initialization
    fct_adf_v = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    fct_plus = numpy.zeros(nodes * max_levels).astype(numpy_real_type)
    fct_minus = numpy.zeros_like(fct_plus).astype(numpy_real_type)
    fct_plus_control = numpy.zeros_like(fct_plus).astype(numpy_real_type)
    fct_minus_control = numpy.zeros_like(fct_minus).astype(numpy_real_type)
    levels = numpy.zeros(nodes).astype(numpy.int32)
    for node in range(0, nodes):
        levels[node] = numpy.random.randint(3, max_levels)
    arguments = [numpy.int32(max_levels), levels, fct_adf_v, fct_plus, fct_minus]
    # Reference
    reference(nodes, levels, max_levels, fct_adf_v, fct_plus, fct_minus)
    arguments_control = [None, None, None, fct_plus_control, fct_minus_control]
    # Tuning
    results, environment = tune_kernel("fct_ale_b1_vertical", generate_code, "{} * block_size_x".format(nodes), arguments, tuning_parameters, lang="CUDA", answer=arguments_control, restrictions=constraints, quiet=True)
    return results

def parse_command_line():
    parser = argparse.ArgumentParser(description="FESOM2 FCT ALE B1 VERTICAL")
    parser.add_argument("--nodes", help="The number of nodes.", type=int, required=True)
    parser.add_argument("--max_levels", help="The maximum number of vertical levels per node.", type=int, required=True)
    parser.add_argument("--max_tile", help="The maximum tiling factor.", type=int, default=2)
    parser.add_argument("--real_type", help="The floating point type to use.", choices=["float", "double"], type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    command_line = parse_command_line()
    results = tune(command_line.nodes, command_line.max_levels, command_line.max_tile, command_line.real_type)
    best_configuration = min(results, key=lambda x : x["time"])
    print("/* Block size X: {} */".format(best_configuration["block_size_x"]))
    print(generate_code(best_configuration))