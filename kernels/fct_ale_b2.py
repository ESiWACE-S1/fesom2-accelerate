
from kernel_tuner import tune_kernel
import numpy
import argparse

def generate_code(tuning_parameters):
    code = \
        "__global__ void fct_ale_b2(const int maxLevels, const <%REAL_TYPE%> dt, const <%REAL_TYPE%> fluxEpsilon, const int * __restrict__ nLevels, const <%REAL_TYPE%> * __restrict__ area, const <%REAL_TYPE%> * __restrict__ fct_ttf_max, const <%REAL_TYPE%> * __restrict__ fct_ttf_min, <%REAL_TYPE%> * __restrict__ fct_plus, <%REAL_TYPE%> * __restrict__ fct_minus)\n" \
        "{\n" \
        "const <%INT_TYPE%> maxNodeLevel = nLevels[blockIdx.x] - 1;\n" \
        "<%INT_TYPE%> index = 0;\n" \
        "<%REAL_TYPE%> area_item = 0;\n" \
        "\n" \
        "for ( <%INT_TYPE%> level = threadIdx.x; level < maxNodeLevel; level += <%BLOCK_SIZE%> )\n" \
        "{\n" \
        "<%COMPUTE_BLOCK%>" \
        "}\n" \
        "}\n"
    compute_block = \
        "index = (blockIdx.x * maxLevels) + level + <%OFFSET%>;\n" \
        "area_item = area[index];\n" \
        "fct_plus[index] = <%FMIN%>(1.0, fct_ttf_max[index] / (((fct_plus[index] * dt) / area_item) + fluxEpsilon));\n" \
        "fct_minus[index] = <%FMIN%>(1.0, fct_ttf_min[index] / (((fct_minus[index] * dt) / area_item) - fluxEpsilon));\n"
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
            compute = compute + "if ( level + {} < maxNodeLevel )\n{{\n{}}}\n".format(str(offset), compute_block.replace("<%OFFSET%>", str(offset)))
    code = code.replace("<%COMPUTE_BLOCK%>", compute)
    if tuning_parameters["real_type"] == "float":
        code = code.replace("<%FMAX%>", "fmaxf")
        code = code.replace("<%FMIN%>", "fminf")
    elif tuning_parameters["real_type"] == "double":
        code = code.replace("<%FMAX%>", "fmax")
        code = code.replace("<%FMIN%>", "fmin")
    else:
        raise ValueError
    code = code.replace("<%INT_TYPE%>", tuning_parameters["int_type"].replace("_", " "))
    code = code.replace("<%REAL_TYPE%>", tuning_parameters["real_type"])
    return code

def reference(nodes, dt, flux_epsilon, levels, max_levels, area, fct_ttf_max, fct_ttf_min, fct_plus, fct_minus):
    for node in range(0, nodes):
        for level in range(0, levels[node] - 1):
            index = (node * max_levels) + level
            temp = fct_ttf_max[index] / (((fct_plus[index] * dt) / area[index]) + flux_epsilon)
            fct_plus[index] = min(1.0, temp)
            temp = fct_ttf_min[index] / (((fct_minus[index] * dt) / area[index]) - flux_epsilon)
            fct_minus[index] = min(1.0, temp)

def tune(nodes, max_levels, max_tile, real_type, quiet=True):
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
    dt = numpy.random.random()
    flux_epsilon = numpy.random.random()
    fct_ttf_max = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    fct_ttf_min = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    area = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    fct_plus = numpy.zeros(nodes * max_levels).astype(numpy_real_type)
    fct_minus = numpy.zeros_like(fct_plus).astype(numpy_real_type)
    fct_plus_control = numpy.zeros_like(fct_plus).astype(numpy_real_type)
    fct_minus_control = numpy.zeros_like(fct_minus).astype(numpy_real_type)
    levels = numpy.zeros(nodes).astype(numpy.int32)
    used_levels = 0
    for node in range(0, nodes):
        levels[node] = numpy.random.randint(3, max_levels)
        used_levels = used_levels + (levels[node] - 1)
    if real_type == "float":
        arguments = [numpy.int32(max_levels), numpy.float32(dt), numpy.float32(flux_epsilon), levels, area, fct_ttf_max, fct_ttf_min, fct_plus, fct_minus]
    elif real_type == "double":
        arguments = [numpy.int32(max_levels), numpy.float64(dt), numpy.float64(flux_epsilon), levels, area, fct_ttf_max, fct_ttf_min, fct_plus, fct_minus]
    else:
        raise ValueError
    # Reference
    reference(nodes, dt, flux_epsilon, levels, max_levels, area, fct_ttf_max, fct_ttf_min, fct_plus_control, fct_minus_control)
    arguments_control = [None, None, None, None, None, None, None, fct_plus_control, fct_minus_control]
    # Tuning
    results, environment = tune_kernel("fct_ale_b2", generate_code, "{} * block_size_x".format(nodes), arguments, tuning_parameters, lang="CUDA", answer=arguments_control, restrictions=constraints, quiet=quiet)
    # Memory bandwidth
    memory_bytes = ((nodes * 4) + (used_levels * 7 * numpy.dtype(numpy_real_type).itemsize))
    for result in results:
        result["memory_bandwidth"] = memory_bytes / (result["time"] / 10**3)
    return results

def parse_command_line():
    parser = argparse.ArgumentParser(description="FESOM2 FCT ALE B2")
    parser.add_argument("--nodes", help="The number of nodes.", type=int, required=True)
    parser.add_argument("--max_levels", help="The maximum number of vertical levels per node.", type=int, required=True)
    parser.add_argument("--max_tile", help="The maximum tiling factor.", type=int, default=2)
    parser.add_argument("--real_type", help="The floating point type to use.", choices=["float", "double"], type=str, required=True)
    parser.add_argument("--verbose", help="Print all kernel configurations.", default=True, action="store_false")
    return parser.parse_args()

if __name__ == "__main__":
    command_line = parse_command_line()
    results = tune(command_line.nodes, command_line.max_levels, command_line.max_tile, command_line.real_type, command_line.verbose)
    best_configuration = min(results, key=lambda x : x["time"])
    print("/* Memory bandwidth: {:.2f} GB/s */".format(best_configuration["memory_bandwidth"] / 10**9))
    print("/* Block size X: {} */".format(best_configuration["block_size_x"]))
    print(generate_code(best_configuration))