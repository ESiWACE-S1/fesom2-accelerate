
from kernel_tuner import tune_kernel
import numpy
import argparse
import json

def generate_code(tuning_parameters):
    code = \
        "__global__ void fct_ale_c_horizontal(const int maxLevels, const int * __restrict__ nLevels, const int * __restrict__ nodesPerEdge, const int * __restrict__ elementsPerEdge, <%REAL_TYPE%> * __restrict__ del_ttf_advhoriz, const <%REAL_TYPE%> * __restrict__ fct_adf_h, const <%REAL_TYPE%> dt, const <%REAL_TYPE%> * __restrict__ area)\n" \
        "{\n" \
        "const <%INT_TYPE%> edge = blockIdx.x * 2;\n" \
        "<%INT_TYPE%> levelBound = 0;\n" \
        "const <%INT_TYPE%> nodeOne = (nodesPerEdge[edge] - 1) * maxLevels;\n" \
        "const <%INT_TYPE%> nodeTwo = (nodesPerEdge[edge + 1] - 1) * maxLevels;\n" \
        "\n" \
        "/* Compute the upper bound for the level */\n" \
        "levelBound = elementsPerEdge[edge + 1];\n" \
        "if ( levelBound > 0 )\n" \
        "{\n" \
        "levelBound = max(nLevels[(elementsPerEdge[edge]) - 1], nLevels[levelBound - 1]);\n" \
        "}\n" \
        "else\n" \
        "{\n" \
        "levelBound = max(nLevels[(elementsPerEdge[edge]) - 1], 0);\n" \
        "}\n" \
        "\n" \
        "for ( <%INT_TYPE%> level = threadIdx.x; level < levelBound - 1; level += <%BLOCK_SIZE%> )\n" \
        "{\n" \
        "<%REAL_TYPE%> fct_adf_h_item = 0;\n" \
        "<%COMPUTE_BLOCK%>" \
        "}\n" \
        "}\n"
    compute_block = \
        "fct_adf_h_item = fct_adf_h[(blockIdx.x * maxLevels) + level + <%OFFSET%>];\n" \
        "atomicAdd(&(del_ttf_advhoriz[nodeOne + level + <%OFFSET%>]), (fct_adf_h_item * (dt / area[nodeOne + level + <%OFFSET%>])));\n" \
        "atomicAdd(&(del_ttf_advhoriz[nodeTwo + level + <%OFFSET%>]), -(fct_adf_h_item * (dt / area[nodeTwo + level + <%OFFSET%>])));\n"
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
            compute = compute + "if ( level + {} < (levelBound - 1) )\n{{\n{}}}\n".format(str(offset), compute_block.replace("<%OFFSET%>", str(offset)))
    code = code.replace("<%COMPUTE_BLOCK%>", compute)
    code = code.replace("<%INT_TYPE%>", tuning_parameters["int_type"].replace("_", " "))
    code = code.replace("<%REAL_TYPE%>", tuning_parameters["real_type"])
    return code

def reference(edges, nodes_per_edge, elements_per_edge, levels, max_levels, del_ttf_advhoriz, fct_adf_h, dt, area, numpy_real_type):
    memory_bytes = 0
    for edge in range(0, edges):
        memory_bytes = memory_bytes + (3 * 4)
        node_one = nodes_per_edge[edge * 2] - 1
        node_two = nodes_per_edge[(edge * 2) + 1] - 1
        element_one = elements_per_edge[edge * 2] - 1
        element_two = elements_per_edge[(edge * 2) + 1] - 1
        if element_two < 0:
            memory_bytes = memory_bytes + (4)
            number_levels = max(levels[element_one], 0)
        else:
            memory_bytes = memory_bytes + (2 * 4)
            number_levels = max(levels[element_one], levels[element_two])
        for level in range(0, number_levels - 1):
            memory_bytes = memory_bytes + (7 * numpy.dtype(numpy_real_type).itemsize)
            del_ttf_advhoriz[(node_one * max_levels) + level] = del_ttf_advhoriz[(node_one * max_levels) + level] + (fct_adf_h[(edge * max_levels) + level] * (dt / area[(node_one * max_levels) + level]))
            del_ttf_advhoriz[(node_two * max_levels) + level] = del_ttf_advhoriz[(node_two * max_levels) + level] - (fct_adf_h[(edge * max_levels) + level] * (dt / area[(node_two * max_levels) + level]))
    return memory_bytes

def tune(nodes, edges, elements, max_levels, max_tile, real_type, quiet=True):
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
    del_ttf_advhoriz = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    del_ttf_advhoriz_control = numpy.copy(del_ttf_advhoriz)
    fct_adf_h = numpy.random.randn(edges * max_levels).astype(numpy_real_type)
    area = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    dt = numpy.random.random()
    levels = numpy.zeros(elements).astype(numpy.int32)
    for element in range(0, elements):
        levels[element] = numpy.random.randint(3, max_levels)
    nodes_per_edge = numpy.zeros(edges * 2).astype(numpy.int32)
    elements_per_edge = numpy.zeros(edges * 2).astype(numpy.int32)
    for edge in range(0, edges):
        nodes_per_edge[edge * 2] = numpy.random.randint(1, nodes + 1)
        nodes_per_edge[(edge * 2) + 1] = numpy.random.randint(1, nodes + 1)
        elements_per_edge[edge * 2] = numpy.random.randint(1, elements + 1)
        elements_per_edge[(edge * 2) + 1] = numpy.random.randint(0, elements + 1)
    if real_type == "float":
        arguments = [numpy.int32(max_levels), levels, nodes_per_edge, elements_per_edge, del_ttf_advhoriz, fct_adf_h, numpy.float32(dt), area]
    elif real_type == "double":
        arguments = [numpy.int32(max_levels), levels, nodes_per_edge, elements_per_edge, del_ttf_advhoriz, fct_adf_h, numpy.float64(dt), area]
    else:
        raise ValueError
    # Reference
    memory_bytes = reference(edges, nodes_per_edge, elements_per_edge, levels, max_levels, del_ttf_advhoriz_control, fct_adf_h, dt, area, numpy_real_type)
    arguments_control = [None, None, None, None, del_ttf_advhoriz_control, None, None, None]
    # Tuning
    results, _ = tune_kernel("fct_ale_c_horizontal", generate_code, "{} * block_size_x".format(edges), arguments, tuning_parameters, lang="CUDA", answer=arguments_control, restrictions=constraints, quiet=quiet, atol=1e-03)
    # Memory bandwidth
    for result in results:
        result["memory_bandwidth"] = memory_bytes / (result["time"] / 10**3)
    return results

def parse_command_line():
    parser = argparse.ArgumentParser(description="FESOM2 FCT ALE C HORIZONTAL")
    parser.add_argument("--nodes", help="The number of nodes.", type=int, required=True)
    parser.add_argument("--edges", help="The number of edges.", type=int, required=True)
    parser.add_argument("--elements", help="The number of elements.", type=int, required=True)
    parser.add_argument("--max_levels", help="The maximum number of horizontal levels per node.", type=int, required=True)
    parser.add_argument("--max_tile", help="The maximum tiling factor.", type=int, default=2)
    parser.add_argument("--real_type", help="The floating point type to use.", choices=["float", "double"], type=str, required=True)
    parser.add_argument("--verbose", help="Print all kernel configurations.", default=True, action="store_false")
    parser.add_argument("--store", help="Store performance results in a JSON file.", default=False, action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    command_line = parse_command_line()
    results = tune(command_line.nodes, command_line.edges, command_line.elements, command_line.max_levels, command_line.max_tile, command_line.real_type, command_line.verbose)
    best_configuration = min(results, key=lambda x : x["time"])
    print("/* Memory bandwidth: {:.2f} GB/s */".format(best_configuration["memory_bandwidth"] / 10**9))
    print("/* Block size X: {} */".format(best_configuration["block_size_x"]))
    print(generate_code(best_configuration))
    if command_line.store:
        try:
            with open("fct_ale_c_horizontal_{}_{}_{}_{}_{}.json".format(command_line.nodes, command_line.elements, command_line.edges, command_line.max_levels, command_line.real_type), "x") as fp:
                json.dump(results, fp)
        except FileExistsError:
            print("Impossible to save the results, a results file already exists for a similar experiment.")