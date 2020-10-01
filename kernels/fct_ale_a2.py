
from kernel_tuner import tune_kernel
import numpy
import argparse
import json

def generate_code(tuning_parameters):
    code = \
        "__global__ void fct_ale_a2(const int maxLevels, const int * __restrict__ nLevels, const int * __restrict__ elementNodes, <%REAL_TYPE%><%VECTOR_SIZE%> * __restrict__ UV_rhs, const <%REAL_TYPE%> * __restrict__ fct_ttf_max, const <%REAL_TYPE%> * __restrict__ fct_ttf_min)\n" \
        "{\n" \
        "const <%INT_TYPE%> elementIndex = (blockIdx.x * maxLevels * 2);\n" \
        "const <%INT_TYPE%> nodeOneIndex = (elementNodes[(blockIdx.x * 3)] - 1) * maxLevels;\n" \
        "const <%INT_TYPE%> nodeTwoIndex = (elementNodes[(blockIdx.x * 3) + 1] - 1) * maxLevels;\n" \
        "const <%INT_TYPE%> nodeThreeIndex = (elementNodes[(blockIdx.x * 3) + 2] - 1) * maxLevels;\n" \
        "const <%INT_TYPE%> maxElementLevel = nLevels[blockIdx.x] - 1;\n" \
        "\n" \
        "for ( <%INT_TYPE%> level = threadIdx.x; level < maxLevels - 1; level += <%BLOCK_SIZE%> )\n" \
        "{\n" \
        "<%COMPUTE_BLOCK%>" \
        "}\n" \
        "}\n"
    compute_block = \
        "if ( level + <%OFFSET%> < maxElementLevel )\n" \
        "{\n" \
        "<%REAL_TYPE%> temp = 0.0;\n" \
        "temp = <%FMAX%>(fct_ttf_max[nodeOneIndex + level + <%OFFSET%>], fct_ttf_max[nodeTwoIndex + level + <%OFFSET%>]);\n" \
        "temp = <%FMAX%>(temp, fct_ttf_max[nodeThreeIndex + level + <%OFFSET%>]);\n" \
        "UV_rhs[elementIndex + ((level + <%OFFSET%>) * 2)] = temp;\n" \
        "temp = <%FMIN%>(fct_ttf_min[nodeOneIndex + level + <%OFFSET%>], fct_ttf_min[nodeTwoIndex + level + <%OFFSET%>]);\n" \
        "temp = <%FMIN%>(temp, fct_ttf_min[nodeThreeIndex + level + <%OFFSET%>]);\n" \
        "UV_rhs[elementIndex + ((level + <%OFFSET%>) * 2) + 1] = temp;\n" \
        "}\n" \
        "else if ( (level + <%OFFSET%> > maxElementLevel) && (level + <%OFFSET%> < maxLevels - 1) )\n" \
        "{\n" \
        "UV_rhs[elementIndex + ((level + <%OFFSET%>) * 2)] = <%MIN%>;\n" \
        "UV_rhs[elementIndex + ((level + <%OFFSET%>) * 2) + 1] = <%MAX%>;\n" \
        "}\n"
    compute_block_vector = \
        "if ( level + <%OFFSET%> < maxElementLevel )\n" \
        "{\n" \
        "<%REAL_TYPE%><%VECTOR_SIZE%> temp = make_<%REAL_TYPE%>2(0.0, 0.0);\n" \
        "temp.x = <%FMAX%>(fct_ttf_max[nodeOneIndex + level + <%OFFSET%>], fct_ttf_max[nodeTwoIndex + level + <%OFFSET%>]);\n" \
        "temp.x = <%FMAX%>(temp.x, fct_ttf_max[nodeThreeIndex + level + <%OFFSET%>]);\n" \
        "temp.y = <%FMIN%>(fct_ttf_min[nodeOneIndex + level + <%OFFSET%>], fct_ttf_min[nodeTwoIndex + level + <%OFFSET%>]);\n" \
        "temp.y = <%FMIN%>(temp.y, fct_ttf_min[nodeThreeIndex + level + <%OFFSET%>]);\n" \
        "UV_rhs[elementIndex + level + <%OFFSET%>] = temp;\n" \
        "}\n" \
        "else if ( (level + <%OFFSET%> > maxElementLevel) && (level + <%OFFSET%> < maxLevels - 1) )\n" \
        "{\n" \
        "UV_rhs[elementIndex + level + <%OFFSET%>] = make_<%REAL_TYPE%>2(<%MIN%>, <%MAX%>);\n" \
        "}\n"
    if tuning_parameters["tiling_x"] > 1:
        code = code.replace("<%BLOCK_SIZE%>", str(tuning_parameters["block_size_x"] * tuning_parameters["tiling_x"]))
    else:
        code = code.replace("<%BLOCK_SIZE%>", str(tuning_parameters["block_size_x"]))
    compute = str()
    for tile in range(0, tuning_parameters["tiling_x"]):
        if tile == 0:
            if tuning_parameters["vector_size"] == 1:
                compute = compute + compute_block.replace(" + <%OFFSET%>", "")
            else:
                compute = compute + compute_block_vector.replace(" + <%OFFSET%>", "")
        else:
            if tuning_parameters["vector_size"] == 1:
                compute = compute + compute_block.replace("<%OFFSET%>", str(tuning_parameters["block_size_x"] * tile))
            else:
                compute = compute + compute_block_vector.replace("<%OFFSET%>", str(tuning_parameters["block_size_x"] * tile))
    if tuning_parameters["real_type"] == "float":
        compute = compute.replace("<%MIN%>", str(numpy.finfo(numpy.float32).min))
        compute = compute.replace("<%MAX%>", str(numpy.finfo(numpy.float32).max))
    elif tuning_parameters["real_type"] == "double":
        compute = compute.replace("<%MIN%>", str(numpy.finfo(numpy.float64).min))
        compute = compute.replace("<%MAX%>", str(numpy.finfo(numpy.float64).max))
    else:
        raise ValueError
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
    if tuning_parameters["vector_size"] == 1:
        code = code.replace("<%VECTOR_SIZE%>", "")
    else:
        code = code.replace("<%VECTOR_SIZE%>", str(tuning_parameters["vector_size"]))
        code = code.replace("maxLevels * 2", "maxLevels")
    return code

def reference(elements, levels, max_levels, nodes, UV_rhs, fct_ttf_max, fct_ttf_min, real_type):
    numpy_real_type = None
    if real_type == "float":
        numpy_real_type = numpy.float32
    elif real_type == "double":
        numpy_real_type = numpy.float64
    else:
        raise ValueError
    memory_bytes = elements * 16
    for element in range(0, elements):
        for level in range(0, levels[element] - 1):
            memory_bytes = memory_bytes + (8 * numpy.dtype(numpy_real_type).itemsize)
            item = (element * max_levels * 2) + (level * 2)
            UV_rhs[item] = max(fct_ttf_max[((nodes[element * 3] - 1) * max_levels) + level], fct_ttf_max[((nodes[(element * 3) + 1] - 1) * max_levels) + level], fct_ttf_max[((nodes[(element * 3) + 2] - 1) * max_levels) + level])
            UV_rhs[item + 1] = min(fct_ttf_min[((nodes[element * 3] - 1) * max_levels) + level], fct_ttf_min[((nodes[(element * 3) + 1] - 1) * max_levels) + level], fct_ttf_min[((nodes[(element * 3) + 2] - 1) * max_levels) + level])
        if levels[element] <= max_levels - 1:
            for level in range(levels[element], max_levels - 1):
                memory_bytes = memory_bytes + (2 * numpy.dtype(numpy_real_type).itemsize)
                item = (element * max_levels * 2) + (level * 2)
                UV_rhs[item] = numpy.finfo(numpy_real_type).min
                UV_rhs[item + 1] = numpy.finfo(numpy_real_type).max
    return memory_bytes

def tune(elements, nodes, max_levels, max_tile, real_type, quiet=True):
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
    tuning_parameters["vector_size"] = [1, 2]
    constraints = list()
    constraints.append("block_size_x * tiling_x <= max_levels")
    # Memory allocation and initialization
    uv_rhs = numpy.zeros(elements * max_levels * 2).astype(numpy_real_type)
    uv_rhs_control = numpy.zeros_like(uv_rhs).astype(numpy_real_type)
    fct_ttf_max = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    fct_ttf_min = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    levels = numpy.zeros(elements).astype(numpy.int32)
    element_nodes = numpy.zeros(elements * 3).astype(numpy.int32)
    for element in range(0, elements):
        levels[element] = numpy.random.randint(3, max_levels)
        element_nodes[(element * 3)] = numpy.random.randint(1, nodes + 1)
        element_nodes[(element * 3) + 1] = numpy.random.randint(1, nodes + 1)
        element_nodes[(element * 3) + 2] = numpy.random.randint(1, nodes + 1)
    arguments = [numpy.int32(max_levels), levels, element_nodes, uv_rhs, fct_ttf_max, fct_ttf_min]
    # Reference
    memory_bytes = reference(elements, levels, max_levels, element_nodes, uv_rhs_control, fct_ttf_max, fct_ttf_min, real_type)
    arguments_control = [None, None, None, uv_rhs_control, None, None]
    # Tuning
    results, _ = tune_kernel("fct_ale_a2", generate_code, "{} * block_size_x".format(elements), arguments, tuning_parameters, lang="CUDA", answer=arguments_control, restrictions=constraints, quiet=quiet)
    # Memory bandwidth
    for result in results:
        result["memory_bandwidth"] = memory_bytes / (result["time"] / 10**3)
    return results

def parse_command_line():
    parser = argparse.ArgumentParser(description="FESOM2 FCT ALE A2")
    parser.add_argument("--elements", help="The number of elements.", type=int, required=True)
    parser.add_argument("--nodes", help="The number of nodes.", type=int, required=True)
    parser.add_argument("--max_levels", help="The maximum number of vertical levels per element.", type=int, required=True)
    parser.add_argument("--max_tile", help="The maximum tiling factor.", type=int, default=2)
    parser.add_argument("--real_type", help="The floating point type to use.", choices=["float", "double"], type=str, required=True)
    parser.add_argument("--verbose", help="Print all kernel configurations.", default=True, action="store_false")
    parser.add_argument("--store", help="Store performance results in a JSON file.", default=False, action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    command_line = parse_command_line()
    results = tune(command_line.elements, command_line.nodes, command_line.max_levels, command_line.max_tile, command_line.real_type, command_line.verbose)
    best_configuration = min(results, key=lambda x : x["time"])
    print("/* Memory bandwidth: {:.2f} GB/s */".format(best_configuration["memory_bandwidth"] / 10**9))
    print("/* Block size X: {} */".format(best_configuration["block_size_x"]))
    print(generate_code(best_configuration))
    if command_line.store:
        try:
            with open("fct_ale_a2_{}_{}_{}_{}.json".format(command_line.nodes, command_line.elements, command_line.max_levels, command_line.real_type), "x") as fp:
                json.dump(results, fp)
        except FileExistsError:
            print("Impossible to save the results, a results file already exists for a similar experiment.")