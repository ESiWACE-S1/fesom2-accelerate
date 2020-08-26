
from kernel_tuner import tune_kernel
import numpy
import argparse

def generate_code(tuning_parameters):
    code = \
        "__global__ void fct_ale_c_vertical(const int maxLevels, const int * __restrict__ nLevels, <%REAL_TYPE%> * __restrict__ del_ttf_advvert, const <%REAL_TYPE%> * __restrict__ ttf, const <%REAL_TYPE%> * __restrict__ hnode, const <%REAL_TYPE%> * __restrict__ fct_LO, const <%REAL_TYPE%> * __restrict__ hnode_new, const <%REAL_TYPE%> * __restrict__ fct_adf_v, const <%REAL_TYPE%> dt, const <%REAL_TYPE%> * __restrict__ area)\n" \
        "{\n" \
        "const <%INT_TYPE%> node = (blockIdx.x * maxLevels);\n" \
        "const <%INT_TYPE%> maxNodeLevel = nLevels[blockIdx.x] - 1;\n" \
        "\n" \
        "for ( <%INT_TYPE%> level = threadIdx.x; level < maxNodeLevel; level += <%BLOCK_SIZE%> )\n" \
        "{\n" \
        "<%REAL_TYPE%> temp = 0;\n" \
        "<%COMPUTE_BLOCK%>" \
        "}\n" \
        "}\n"
    compute_block = \
        "temp = del_ttf_advvert[node + level + <%OFFSET%>] - (ttf[node + level + <%OFFSET%>] * hnode[node + level + <%OFFSET%>]);\n" \
        "temp += fct_LO[node + level + <%OFFSET%>] * hnode_new[node + level + <%OFFSET%>];\n" \
        "temp += (fct_adf_v[node + level + <%OFFSET%>] - fct_adf_v[node + level + <%OFFSET%> + 1]) * (dt / area[node + level + <%OFFSET%>]);\n" \
        "del_ttf_advvert[node + level + <%OFFSET%>] = temp;\n"
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
    code = code.replace("<%INT_TYPE%>", tuning_parameters["int_type"].replace("_", " "))
    code = code.replace("<%REAL_TYPE%>", tuning_parameters["real_type"])
    return code

def reference(nodes, levels, max_levels, del_ttf_advvert, ttf, hnode, fct_LO, hnode_new, fct_adf_v, dt, area):
    for node in range(0, nodes):
        for level in range(0, levels[node] - 1):
            del_ttf_advvert[(node * max_levels) + level] = del_ttf_advvert[(node * max_levels) + level] - (ttf[(node * max_levels) + level] * hnode[(node * max_levels) + level]) + (fct_LO[(node * max_levels) + level] * hnode_new[(node * max_levels) + level]) + ((fct_adf_v[(node * max_levels) + level] - fct_adf_v[(node * max_levels) + (level + 1)]) * (dt / area[(node * max_levels) + level]))

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
    del_ttf_advvert = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    del_ttf_advvert_control = numpy.copy(del_ttf_advvert)
    ttf = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    hnode = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    fct_LO = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    hnode_new = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    fct_adf_v = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    area = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    dt = numpy.random.random()
    levels = numpy.zeros(nodes).astype(numpy.int32)
    used_levels = 0
    for node in range(0, nodes):
        levels[node] = numpy.random.randint(3, max_levels)
        used_levels = used_levels + (levels[node] - 1)
    if real_type == "float":
        arguments = [numpy.int32(max_levels), levels, del_ttf_advvert, ttf, hnode, fct_LO, hnode_new, fct_adf_v, numpy.float32(dt), area]
    elif real_type == "double":
        arguments = [numpy.int32(max_levels), levels, del_ttf_advvert, ttf, hnode, fct_LO, hnode_new, fct_adf_v, numpy.float64(dt), area]
    else:
        raise ValueError
    # Reference
    reference(nodes, levels, max_levels, del_ttf_advvert_control, ttf, hnode, fct_LO, hnode_new, fct_adf_v, dt, area)
    arguments_control = [None, None, del_ttf_advvert_control, None, None, None, None, None, None, None]
    # Tuning
    results, _ = tune_kernel("fct_ale_c_vertical", generate_code, "{} * block_size_x".format(nodes), arguments, tuning_parameters, lang="CUDA", answer=arguments_control, restrictions=constraints, quiet=quiet)
    # Memory bandwidth
    memory_bytes = (nodes * 4) + (used_levels * 9 * numpy.dtype(numpy_real_type).itemsize)
    for result in results:
        result["memory_bandwidth"] = memory_bytes / (result["time"] / 10**3)
    return results

def parse_command_line():
    parser = argparse.ArgumentParser(description="FESOM2 FCT ALE C VERTICAL")
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