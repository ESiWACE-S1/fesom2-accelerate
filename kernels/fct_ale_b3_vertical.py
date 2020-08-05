
from kernel_tuner import tune_kernel
import numpy
import argparse

def generate_code(tuning_parameters):
    code = \
        "__global__ void fct_ale_b3_vertical(const int maxLevels, const int * __restrict__ nLevels, <%REAL_TYPE%> * __restrict__ fct_adf_v, const <%REAL_TYPE%> * __restrict__ fct_plus, const <%REAL_TYPE%> * __restrict__ fct_minus)\n" \
        "{\n" \
        "const <%INT_TYPE%> node = (blockIdx.x * maxLevels);\n" \
        "const <%INT_TYPE%> maxNodeLevel = nLevels[blockIdx.x] - 1;\n" \
        "\n" \
        "/* Intermediate levels */\n" \
        "for ( <%INT_TYPE%> level = threadIdx.x + 1; level < maxNodeLevel; level += <%BLOCK_SIZE%> )\n" \
        "{\n" \
        "<%REAL_TYPE%> flux = 0.0;\n" \
        "<%REAL_TYPE%> ae_plus = 0.0;\n" \
        "<%REAL_TYPE%> ae_minus = 0.0;\n" \
        "<%COMPUTE_BLOCK%>" \
        "}\n" \
        "/* Top level */\n" \
        "if ( threadIdx.x == 0 )\n" \
        "{\n" \
        "<%REAL_TYPE%> flux = fct_adf_v[node];\n" \
        "<%REAL_TYPE%> ae = 1.0;\n" \
        "if ( signbit(flux) == 0 )\n" \
        "{\n" \
        "ae = <%FMIN%>(ae, fct_plus[node]);\n" \
        "}\n" \
        "else\n" \
        "{\n" \
        "ae = <%FMIN%>(ae, fct_minus[node]);\n" \
        "}\n" \
        "fct_adf_v[node] = ae * flux;\n" \
        "}\n" \
        "}\n"
    compute_block = \
        "flux = fct_adf_v[node + level + <%OFFSET%>];\n" \
        "ae_plus = 1.0;\n" \
        "ae_minus = 1.0;\n" \
        "ae_plus = <%FMIN%>(ae_plus, fct_minus[node + (level + <%OFFSET%>) - 1]);\n" \
        "ae_minus = <%FMIN%>(ae_minus, fct_minus[node + (level + <%OFFSET%>)]);\n" \
        "ae_plus = <%FMIN%>(ae_plus, fct_plus[node + (level + <%OFFSET%>)]);\n" \
        "ae_minus = <%FMIN%>(ae_minus, fct_plus[node + (level + <%OFFSET%>) - 1]);\n" \
        "if ( signbit(flux) == 0 )\n" \
        "{\n" \
        "flux *= ae_plus;\n" \
        "}\n" \
        "else\n" \
        "{\n" \
        "flux *= ae_minus;\n" \
        "}\n" \
        "fct_adf_v[node + level + <%OFFSET%>] = flux;\n"
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

def reference(nodes, levels, max_levels, fct_adf_v, fct_plus, fct_minus):
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
    tuning_parameters["shared_memory"] = [False]
    tuning_parameters["int_type"] = ["unsigned_int", "int"]
    tuning_parameters["real_type"] = [real_type]
    tuning_parameters["max_levels"] = [str(max_levels)]
    tuning_parameters["block_size_x"] = [32 * i for i in range(1, 33)]
    tuning_parameters["tiling_x"] = [i for i in range(1, max_tile)]
    constraints = list()
    constraints.append("block_size_x * tiling_x <= max_levels")
    # Memory allocation and initialization
    fct_adf_v = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    fct_adf_v_control = numpy.copy(fct_adf_v)
    fct_plus = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    fct_minus = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    levels = numpy.zeros(nodes).astype(numpy.int32)
    used_levels = 0
    for node in range(0, nodes):
        levels[node] = numpy.random.randint(3, max_levels)
        used_levels = used_levels + (levels[node] - 2)
    arguments = [numpy.int32(max_levels), levels, fct_adf_v, fct_plus, fct_minus]
    # Reference
    reference(nodes, levels, max_levels, fct_adf_v_control, fct_plus, fct_minus)
    arguments_control = [None, None, fct_adf_v_control, None, None]
    # Tuning
    results, _ = tune_kernel("fct_ale_b3_vertical", generate_code, "{} * block_size_x".format(nodes), arguments, tuning_parameters, lang="CUDA", answer=arguments_control, restrictions=constraints, quiet=quiet)
    # Memory bandwidth
    memory_bytes = ((nodes * 4) + (nodes * 3 * numpy.dtype(numpy_real_type).itemsize) + (used_levels * 6 * numpy.dtype(numpy_real_type).itemsize))
    for result in results:
        result["memory_bandwidth"] = memory_bytes / (result["time"] / 10**3)
    return results

def parse_command_line():
    parser = argparse.ArgumentParser(description="FESOM2 FCT ALE B3 VERTICAL")
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