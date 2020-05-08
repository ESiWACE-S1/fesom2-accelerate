
from kernel_tuner import tune_kernel
import numpy
import argparse

def generate_code(tuning_parameters):
    code = \
        "__global__ void fct_ale_a2(const int * __restrict__ nLevels, const int * __restrict__ elementNodes, <%REAL_TYPE%>2 * __restrict__ UV_rhs, const <%REAL_TYPE%> * __restrict__ fct_ttf_max, const <%REAL_TYPE%> * __restrict__ fct_ttf_min)\n" \
        "{\n" \
        "const <%INT_TYPE%> element_index = (blockIdx.x * <%MAX_LEVELS%>);\n" \
        "const <%INT_TYPE%> element_node0_index = elementNodes[(blockIdx.x * 3)] * <%MAX_LEVELS%>;\n" \
        "const <%INT_TYPE%> element_node1_index = elementNodes[(blockIdx.x * 3) + 1] * <%MAX_LEVELS%>;\n" \
        "const <%INT_TYPE%> element_node2_index = elementNodes[(blockIdx.x * 3) + 2] * <%MAX_LEVELS%>;\n" \
        "for ( <%INT_TYPE%> level = threadIdx.x; level < <%MAX_LEVELS%> - 1; level += <%BLOCK_SIZE%> )\n" \
        "{\n" \
        "<%COMPUTE_BLOCK%>" \
        "}\n" \
        "}\n"
    compute_block = \
        "if ( level + <%OFFSET%> < nLevels[blockIdx.x] ) {\n" \
        "<%REAL_TYPE%>2 temp = make_<%REAL_TYPE%>2(0.0, 0.0);\n" \
        "temp.x = fmax(fct_ttf_max[element_node0_index + level + <%OFFSET%>], fct_ttf_max[element_node1_index + level + <%OFFSET%>]);\n" \
        "temp.x = fmax(temp.x, fct_ttf_max[element_node2_index + level + <%OFFSET%>]);\n" \
        "temp.y = fmin(fct_ttf_min[element_node0_index + level + <%OFFSET%>], fct_ttf_min[element_node1_index + level + <%OFFSET%>]);\n" \
        "temp.y = fmin(temp.y, fct_ttf_min[element_node2_index + level + <%OFFSET%>]);\n" \
        "UV_rhs[element_index + level + <%OFFSET%>] = temp;\n" \
        "}\n" \
        "else {\n" \
        "UV_rhs[element_index + level + <%OFFSET%>] = make_<%REAL_TYPE%>2(<%MIN%>, <%MAX%>);\n" \
        "}\n"
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
            compute = compute + compute_block.replace(" <%OFFSET%>", str(tuning_parameters["block_size_x"] * tile))
    if tuning_parameters["real_type"] == "float":
        compute = compute.replace("<%MIN%>", str(numpy.finfo(numpy.float32).min))
        compute = compute.replace("<%MAX%>", str(numpy.finfo(numpy.float32).max))
    elif tuning_parameters["real_type"] == "double":
        compute = compute.replace("<%MIN%>", str(numpy.finfo(numpy.float64).min))
        compute = compute.replace("<%MAX%>", str(numpy.finfo(numpy.float64).max))
    else:
        raise ValueError
    compute = compute.replace("<%REAL_TYPE%>", tuning_parameters["real_type"])
    code = code.replace("<%COMPUTE_BLOCK%>", compute)
    return code

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

def tune(elements, nodes, max_levels, max_tile, real_type):
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
    uv_rhs = numpy.zeros(elements * max_levels * 2).astype(numpy_real_type)
    uv_rhs_control = numpy.zeros_like(uv_rhs).astype(numpy_real_type)
    fct_ttf_max = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    fct_ttf_min = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    levels = numpy.zeros(elements).astype(numpy.int32)
    element_nodes = numpy.zeros(elements * 3).astype(numpy.int32)
    for element in range(0, elements):
        levels[element] = numpy.random.randint(0, max_levels)
        element_nodes[(element * 3)] = numpy.random.randint(0, nodes)
        element_nodes[(element * 3) + 1] = numpy.random.randint(0, nodes)
        element_nodes[(element * 3) + 2] = numpy.random.randint(0, nodes)
    arguments = [levels, element_nodes, uv_rhs, fct_ttf_max, fct_ttf_min]
    # Reference
    reference(elements, levels, max_levels, element_nodes, uv_rhs_control, fct_ttf_max, fct_ttf_min, real_type)
    arguments_control = [None, None, uv_rhs, None, None]
    # Tuning
    results, environment = tune_kernel("fct_ale_a2", generate_code, "{} * block_size_x".format(elements), arguments, tuning_parameters, lang="CUDA", answer=arguments_control, verify=verify, restrictions=constraints, quiet=True)
    return results

def parse_command_line():
    parser = argparse.ArgumentParser(description="FESOM2 FCT ALE A2")
    parser.add_argument("--elements", help="The number of elements.", type=int, required=True)
    parser.add_argument("--nodes", help="The number of nodes.", type=int, required=True)
    parser.add_argument("--max_levels", help="The maximum number of vertical levels per element.", type=int, required=True)
    parser.add_argument("--max_tile", help="The maximum tiling factor.", type=int, default=2)
    parser.add_argument("--real_type", help="The floating point type to use.", choices=["float", "double"], type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    command_line = parse_command_line()
    results = tune(command_line.elements, command_line.nodes, command_line.max_levels, command_line.max_tile, command_line.real_type)
    best_configuration = min(results, key=lambda x : x["time"])
    print("/* Block size X: {} */".format(best_configuration["block_size_x"]))
    print(generate_code(best_configuration))