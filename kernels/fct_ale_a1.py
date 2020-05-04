
from kernel_tuner import tune_kernel
import numpy
import argparse

def generate_code(tuning_parameters):
    code = \
        "__global__ void fct_ale_a1(const <%REAL_TYPE%> * __restrict__ fct_low_order, const <%REAL_TYPE%> * __restrict__ ttf, const int * __restrict__ nLevels, <%REAL_TYPE%> * fct_ttf_max, <%REAL_TYPE%> * fct_ttf_min)\n" \
        "{\n" \
        "const <%INT_TYPE%> node = (blockIdx.x * <%MAX_LEVELS%>);\n" \
        "\n" \
        "for ( <%INT_TYPE%> level = threadIdx.x; level < nLevels[blockIdx.x]; level += <%BLOCK_SIZE%> )\n" \
        "{\n" \
        "<%REAL_TYPE%> fct_low_order_item = 0;\n" \
        "<%REAL_TYPE%> ttf_item = 0;\n" \
        "<%COMPUTE_BLOCK%>" \
        "}\n" \
        "}\n"
    compute_block = "fct_low_order_item = fct_low_order[node + level + <%OFFSET%>];\n" \
        "ttf_item = ttf[node + level + <%OFFSET%>];\n" \
        "fct_ttf_max[node + level + <%OFFSET%>] = fmax(fct_low_order_item, ttf_item);\n" \
        "fct_ttf_min[node + level + <%OFFSET%>] = fmin(fct_low_order_item, ttf_item);\n"
    code = code.replace("<%INT_TYPE%>", tuning_parameters["int_type"].replace("_", " "))
    code = code.replace("<%REAL_TYPE%>", tuning_parameters["real_type"])
    code = code.replace("<%MAX_LEVELS%>", str(tuning_parameters["max_levels"]))
    if tuning_parameters["tiling_x"] > 1:
        code = code.replace("<%BLOCK_SIZE%>", str(tuning_parameters["block_size_x"] * tuning_parameters["tiling_x"]))
    else:
        code = code.replace("<%BLOCK_SIZE%>", str(tuning_parameters["block_size_x"]))
    compute = str()
    for tile in range(0, tuning_parameters["tiling_x"]):
        if tile == 0:
            compute = compute + compute_block.replace(" + <%OFFSET%>", "")
        else:
            compute = compute + compute_block.replace("<%OFFSET%>", tuning_parameters["block_size_x"] * tile)
    code = code.replace("<%COMPUTE_BLOCK%>", compute)
    return code

def reference(nodes, levels, max_levels, fct_low_order, ttf, fct_ttf_max, fct_ttf_min):
    for node in range(0, nodes):
        for level in range(0, levels[node]):
            item = (node * max_levels) + level
            fct_ttf_max[item] = max(fct_low_order[item], ttf[item])
            fct_ttf_min[item] = min(fct_low_order[item], ttf[item])

def verify(control_data, data, atol=None):
    return numpy.allclose(control_data, data, atol)

def tune(nodes, max_levels, max_tile, real_type):
    numpy_real_type = None
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
    if real_type == "float":
        numpy_real_type = numpy.float32
    elif real_type == "double":
        numpy_real_type = numpy.float64
    else:
        raise ValueError
    fct_low_order = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    ttf = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    fct_ttf_max = numpy.zeros(nodes * max_levels)
    fct_ttf_min = numpy.zeros_like(fct_ttf_max)
    fct_ttf_max_control = numpy.zeros_like(fct_ttf_max)
    fct_ttf_min_control = numpy.zeros_like(fct_ttf_min)
    levels = numpy.zeros(nodes).astype(numpy.int32)
    for node in range(0, nodes):
        levels[node] = numpy.random.randint(0, max_levels)
    arguments = [fct_low_order, ttf, levels, fct_ttf_max, fct_ttf_min]
    # Reference
    reference(nodes, levels, max_levels, fct_low_order, ttf, fct_ttf_max_control, fct_ttf_min_control)
    arguments_control = [None, None, None, fct_ttf_max_control, fct_ttf_min_control]
    # Tuning
    results, environment = tune_kernel("fct_ale_a1", generate_code, "{} * block_size_x".format(nodes), arguments, tuning_parameters, lang="CUDA", answer=arguments_control, verify=verify, restrictions=constraints, quiet=True)
    return results

def parse_command_line():
    parser = argparse.ArgumentParser(description="FESOM2 FCT ALE A1")
    parser.add_argument("--nodes", help="The number of nodes.", type=int, required=True)
    parser.add_argument("--max_levels", help="The maximum number of vertical levels for nodes.", type=int, required=True)
    parser.add_argument("--max_tile", help="The maximum level of tiling.", type=int, default=2)
    parser.add_argument("--real_type", help="The floating point type to use.", choices=["float", "double"], type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    command_line = parse_command_line()
    results = tune(command_line.nodes, command_line.max_levels, command_line.max_tile, command_line.real_type)
    best_configuration = min(results, key=lambda x : x["time"])
    print("Block size X: {}".format(best_configuration["block_size_x"]))
    print(generate_code(best_configuration))