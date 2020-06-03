
from kernel_tuner import tune_kernel
import numpy
import argparse

def generate_code(tuning_parameters):
    code = \
        "__global__ void fct_ale_b1_horizontal(const int maxLevels, const int * __restrict__ nLevels, const int * __restrict__ nodesPerEdge, const int * __restrict__ elementsPerEdge, const <%REAL_TYPE%> * __restrict__ fct_adf_h, <%REAL_TYPE%> * __restrict__ fct_plus, <%REAL_TYPE%> * __restrict__ fct_minus)\n" \
        "{\n" \
        "<%INT_TYPE%> levelBound = 0;\n" \
        "const <%INT_TYPE%> nodeOne = nodesPerEdge[(blockIdx.x * 2)] * maxLevels;\n" \
        "const <%INT_TYPE%> nodeTwo = nodesPerEdge[(blockIdx.x * 2) + 1] * maxLevels;\n" \
        "\n" \
        "/* Compute the upper bound for the level */\n" \
        "levelBound = elementsPerEdge[(blockIdx.x * 2) + 1];\n" \
        "if ( levelBound > 0 )\n" \
        "{\n" \
        "levelBound = max(nLevels[elementsPerEdge[(blockIdx.x * 2)], nLevels[levelBound]);\n" \
        "}\n" \
        "else" \
        "{\n" \
        "levelBound = max(nLevels[elementsPerEdge[(blockIdx.x * 2)], 0);\n" \
        "}\n" \
        "/* Compute fct_plus and fct_minus */\n" \
        "for ( <%INT_TYPE%> level = threadIdx.x; level < levelBound; level += <%BLOCK_SIZE%> )\n" \
        "{\n" \
        "<%REAL_TYPE%> fct_adf_h_value = 0.0;\n" \
        "<%COMPUTE_BLOCK%>" \
        "}\n" \
        "}\n"
    compute_block = \
        "fct_adf_h_value = fct_adf_h[(blockIdx.x * maxLevels) + level + <%OFFSET%>];\n" \
        "fct_plus[nodeOne + level + <%OFFSET%>] = fct_plus[nodeOne + level + <%OFFSET%>] + <%MAX%>(0.0, fct_adf_h_value);\n" \
        "fct_minus[nodeOne + level + <%OFFSET%>] = fct_minus[nodeOne + level + <%OFFSET%>] + <%MIN%>(0.0, fct_adf_h_value);\n" \
        "fct_plus[nodeTwo + level + <%OFFSET%>] = fct_plus[nodeTwo + level + <%OFFSET%>] + <%MAX%>(0.0, -fct_adf_h_value);\n" \
        "fct_minus[nodeTwo + level + <%OFFSET%>] = fct_minus[nodeTwo + level + <%OFFSET%>] + <%MIN%>(0.0, -fct_adf_h_value);\n"
    return code

def reference(edges, nodes_per_edge, elements_per_edge, levels, max_levels, fct_adf_h, fct_plus, fct_minus):
    for edge in range(0, edges):
        node_one = nodes_per_edge[edge * 2]
        node_two = nodes_per_edge[(edge * 2) + 1]
        element_one = elements_per_edge[edge * 2]
        element_two = elements_per_edge[(edge * 2) + 1]
        if element_two <= 0:
            number_levels = max(levels(element_one), 0)
        else:
            number_levels = max(levels(element_one), levels(element_two))
        for level in range(0, number_levels):
            fct_plus[(node_one * max_levels) + level] = fct_plus[(node_one * max_levels) + level] + max(0.0, fct_adf_h[(edge * max_levels) + level])
            fct_minus[(node_one * max_levels) + level] = fct_minus[(node_one * max_levels) + level] + min(0.0, fct_adf_h[(edge * max_levels) + level])
            fct_plus[(node_two * max_levels) + level] = fct_plus[(node_two * max_levels) + level] + max(0.0, -fct_adf_h[(edge * max_levels) + level])
            fct_minus[(node_two * max_levels) + level] = fct_minus[(node_two * max_levels) + level] + min(0.0, -fct_adf_h[(edge * max_levels) + level])

def tune(nodes, edges, elements, max_levels, max_tile, real_type):
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
    fct_adf_h = numpy.random.randn(edges * max_levels).astype(numpy_real_type)
    fct_plus = numpy.zeros(nodes * max_levels).astype(numpy_real_type)
    fct_minus = numpy.zeros_like(fct_plus).astype(numpy_real_type)
    fct_plus_control = numpy.zeros_like(fct_plus).astype(numpy_real_type)
    fct_minus_control = numpy.zeros_like(fct_minus).astype(numpy_real_type)
    levels = numpy.zeros(elements).astype(numpy.int32)
    for element in range(0, elements):
        levels[element] = numpy.random.randint(3, max_levels)
    nodes_per_edge = numpy.zeros(edges * 2).astype(numpy.int32)
    elements_per_edge = numpy.zeros(edges * 2).astype(numpy.int32)
    for edge in range(0, edges):
        nodes_per_edge[edge * 2] = numpy.random.randint(0, nodes)
        nodes_per_edge[(edge * 2) + 1] = numpy.random.randint(0, nodes)
        elements_per_edge[edge * 2] = numpy.random.randint(0, elements)
        elements_per_edge[(edge * 2) + 1] = numpy.random.randint(0, elements)
    arguments = []
    # Reference
    reference(edges, nodes_per_edge, elements_per_edge, levels, max_levels, fct_adf_h, fct_plus_control, fct_minus_control)
    arguments_control = []
    # Tuning
    results, environment = tune_kernel("fct_ale_b1_vertical", generate_code, "{} * block_size_x".format(nodes), arguments, tuning_parameters, lang="CUDA", answer=arguments_control, restrictions=constraints, quiet=True)
    return results

def parse_command_line():
    parser = argparse.ArgumentParser(description="FESOM2 FCT ALE B1 HORIZONTAL")
    parser.add_argument("--nodes", help="The number of nodes.", type=int, required=True)
    parser.add_argument("--edges", help="The number of edges.", type=int, required=True)
    parser.add_argument("--elements", help="The number of elements.", type=int, required=True)
    parser.add_argument("--max_levels_node", help="The maximum number of vertical levels.", type=int, required=True)
    parser.add_argument("--max_tile", help="The maximum tiling factor.", type=int, default=2)
    parser.add_argument("--real_type", help="The floating point type to use.", choices=["float", "double"], type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    command_line = parse_command_line()
    results = tune()
    best_configuration = min(results, key=lambda x : x["time"])
    print("/* Block size X: {} */".format(best_configuration["block_size_x"]))
    print(generate_code(best_configuration))