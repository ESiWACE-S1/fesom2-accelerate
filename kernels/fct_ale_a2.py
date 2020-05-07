
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