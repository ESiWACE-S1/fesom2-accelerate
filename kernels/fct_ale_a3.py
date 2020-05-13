
from kernel_tuner import tune_kernel
import numpy
import argparse

def generate_code(tuning_parameters):
    numpy_real_type = None
    if tuning_parameters["real_type"] == "float":
        numpy_real_type = numpy.float32
    elif tuning_parameters["real_type"] == "double":
        numpy_real_type = numpy.float64
    else:
        raise ValueError
    code = \
        "__global__ void fct_ale_a3(const int maxLevels, const int maxElements, const int * __restrict__ nLevels, const int * __restrict__ elements_in_node, const int * __restrict__ number_elements_in_node, const <%REAL_TYPE%><%VECTOR_SIZE%> * __restrict__ UV_rhs, <%REAL_TYPE%> * __restrict__ fct_ttf_max, <%REAL_TYPE%> * __restrict__ fct_ttf_min, const <%REAL_TYPE%> * __restrict__ fct_lo)\n" \
        "{\n" \
        "<%INT_TYPE%> item = 0;\n" \
        "extern __shared__ <%REAL_TYPE%> tvert_max[];\n" \
        "extern __shared__ <%REAL_TYPE%> tvert_min[];\n" \
        "/* Compute tvert_max and tvert_min per level */\n" \
        "for ( <%INT_TYPE%> level = threadIdx.x; level < nLevels[blockIdx.x]; level += <%BLOCK_SIZE%> )\n" \
        "{\n" \
        "<%REAL_TYPE%> tvert_max_temp = 0.0;" \
        "<%REAL_TYPE%> tvert_min_temp = 0.0;" \
        "<%REDUCTION%>" \
        "}\n" \
        "__syncthreads();\n" \
        "/* Update fct_ttf_max and fct_ttf_min per level */\n" \
        "item = blockIdx.x * maxLevels;\n" \
        "for ( <%INT_TYPE%> level = threadIdx.x + 1; level < nLevels[blockIdx.x] - 2; level += <%BLOCK_SIZE%> )\n" \
        "{\n" \
        "<%REAL_TYPE%> temp = 0.0;" \
        "<%UPDATE%>" \
        "}\n" \
        "if ( threadIdx.x == 0 )\n" \
        "{\n" \
        "fct_ttf_max[item] = tvert_max[0] - fct_lo[item];\n" \
        "fct_ttf_min[item] = tvert_min[0] - fct_lo[item];\n" \
        "fct_ttf_max[item + (nLevels[blockIdx.x] - 1)] = tvert_max[nLevels[blockIdx.x] - 1] - fct_lo[item + (nLevels[blockIdx.x] - 1)];\n" \
        "fct_ttf_min[item + (nLevels[blockIdx.x] - 1)] = tvert_min[nLevels[blockIdx.x] - 1] - fct_lo[item + (nLevels[blockIdx.x] - 1)];\n" \
        "}\n" \
        "}\n"
    reduction = \
        "item = (elements_in_node[(blockIdx.x * maxElements)] * maxLevels * 2) + ((level + <%OFFSET%>) * 2);\n" \
        "tvert_max_temp = UV_rhs[item];\n" \
        "tvert_min_temp = UV_rhs[item + 1];\n" \
        "for ( <%INT_TYPE%> element = 1; element < number_elements_in_node[blockIdx.x]; element++ )\n" \
        "{\n" \
        "item = (elements_in_node[(blockIdx.x * maxElements) + element] * maxLevels * 2) + ((level + <%OFFSET%>) * 2);\n" \
        "tvert_max_temp = fmax(tvert_max_temp, UV_rhs[item]);\n" \
        "tvert_min_temp = fmin(tvert_min_temp, UV_rhs[item + 1]);\n" \
        "}\n" \
        "tvert_max[level + <%OFFSET%>] = tvert_max_temp;\n" \
        "tvert_min[level + <%OFFSET%>] = tvert_min_temp;\n"
    update = \
        "temp = fmax(tvert_max[(level + <%OFFSET%>) - 1], tvert_max[level + <%OFFSET%>]);\n" \
        "temp = fmax(temp, tvert_max[(level + <%OFFSET%>) + 1]);\n" \
        "fct_ttf_max[item + level + <%OFFSET%>] = temp - fct_lo[item + level + <%OFFSET%>];\n" \
        "temp = fmin(tvert_min[(level + <%OFFSET%>) - 1], tvert_min[level + <%OFFSET%>]);\n" \
        "temp = fmin(temp, tvert_min[(level + <%OFFSET%>) + 1]);\n" \
        "fct_ttf_min[item + level + <%OFFSET%>] = temp - fct_lo[item + level + <%OFFSET%>];\n"

def reference(vlimit, nodes, levels, max_levels, elements_in_node, number_elements_in_node, max_elements_in_node, uv_rhs, fct_ttf_max, fct_ttf_min, fct_lo, real_type):
    tvert_max = 0
    tvert_min = 0
    if vlimit == 1:
        for node in range(0, nodes):
            for level in range(0, levels[node] - 1):
                max_temp = numpy.finfo(real_type).min
                min_temp = numpy.finfo(real_type).max
                for element in range(0, number_elements_in_node[node]):
                    item = (elements_in_node[(node * max_elements_in_node) + element] * max_levels * 2) + (level * 2)
                    max_temp = max(max_temp, uv_rhs[item])
                    min_temp = min(min_temp, uv_rhs[item + 1])
                tvert_max[level] = max_temp
                tvert_min[level] = min_temp
            # Surface level
            fct_ttf_max[(node * max_levels)] = tvert_max[0] - fct_lo[(node * max_levels)]
            fct_ttf_min[(node * max_levels)] = tvert_min[0] - fct_lo[(node * max_levels)]
            # Intermediate levels
            for level in range(1, levels[node] - 2):
                temp = max(tvert_max[level - 1], tvert_max[level])
                temp = max(temp, tvert_max[level + 1])
                fct_ttf_max[(node * max_levels) + level] = temp - fct_lo[(node * max_levels) + level]
                temp = min(tvert_min[level - 1], tvert_min[level])
                temp = min(temp, tvert_min[level + 1])
                fct_ttf_min[(node * max_levels) + level] = temp - fct_lo[(node * max_levels) + level]
            # Bottom level
            fct_ttf_max[(node * max_levels) + levels[node] - 1] = tvert_max[levels[node] - 1] - fct_lo[(node * max_levels) + levels[node] - 1]
            fct_ttf_min[(node * max_levels) + levels[node] - 1] = tvert_min[levels[node] - 1] - fct_lo[(node * max_levels) + levels[node] - 1]
    elif vlimit == 2:
        pass
    elif vlimit == 3:
        pass
    else:
        raise ValueError

def verify(control_data, data, atol=None):
    return numpy.allclose(control_data, data, atol)

def tune(nodes, max_levels, vlimit, max_tile, real_type):
    numpy_real_type = None
    if real_type == "float":
        numpy_real_type = numpy.float32
    elif real_type == "double":
        numpy_real_type = numpy.float64
    else:
        raise ValueError
    pass

def parse_command_line():
    parser = argparse.ArgumentParser(description="FESOM2 FCT ALE A3")
    parser.add_argument("--nodes", help="The number of nodes.", type=int, required=True)
    parser.add_argument("--max_levels", help="The maximum number of vertical levels per element.", type=int, required=True)
    parser.add_argument("--vlimit", help="The version of vertical limit.", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--max_tile", help="The maximum tiling factor.", type=int, default=2)
    parser.add_argument("--real_type", help="The floating point type to use.", choices=["float", "double"], type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    command_line = parse_command_line()
    results = tune(command_line.nodes, command_line.max_levels, command_line.vlimit, command_line.max_tile, command_line.real_type)
    best_configuration = min(results, key=lambda x : x["time"])
    print("/* Block size X: {} */".format(best_configuration["block_size_x"]))
    print(generate_code(best_configuration))