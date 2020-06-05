
from kernel_tuner import tune_kernel
import numpy
import argparse

def generate_code(tuning_parameters):
    code = \
        "__global__ void fct_ale_a3(const int maxLevels, const int maxElements, const int * __restrict__ nLevels, const int * __restrict__ elements_in_node, const int * __restrict__ number_elements_in_node, const <%REAL_TYPE%><%VECTOR_SIZE%> * __restrict__ UV_rhs, <%REAL_TYPE%> * __restrict__ fct_ttf_max, <%REAL_TYPE%> * __restrict__ fct_ttf_min, const <%REAL_TYPE%> * __restrict__ fct_lo)\n" \
        "{\n" \
        "<%INT_TYPE%> item = 0;\n" \
        "const <%INT_TYPE%> maxNodeLevel = nLevels[blockIdx.x];\n" \
        "extern __shared__ <%REAL_TYPE%> sharedBuffer[];\n" \
        "<%REAL_TYPE%> * tvert_max = (<%REAL_TYPE%> *)(sharedBuffer);\n" \
        "<%REAL_TYPE%> * tvert_min = (<%REAL_TYPE%> *)(&sharedBuffer[maxLevels]);\n" \
        "\n" \
        "/* Compute tvert_max and tvert_min per level */\n" \
        "for ( <%INT_TYPE%> level = threadIdx.x; level < maxNodeLevel - 1; level += <%BLOCK_SIZE%> )\n" \
        "{\n" \
        "<%REAL_TYPE%> tvert_max_temp = 0.0;\n" \
        "<%REAL_TYPE%> tvert_min_temp = 0.0;\n" \
        "<%REDUCTION%>" \
        "}\n" \
        "if ( threadIdx.x == 0 )\n" \
        "{\n" \
        "tvert_max[maxNodeLevel - 1] = 0;\n" \
        "tvert_min[maxNodeLevel - 1] = 0;\n" \
        "}\n" \
        "__syncthreads();\n" \
        "/* Update fct_ttf_max and fct_ttf_min per level */\n" \
        "item = blockIdx.x * maxLevels;\n" \
        "for ( <%INT_TYPE%> level = threadIdx.x + 1; level < maxNodeLevel - 2; level += <%BLOCK_SIZE%> )\n" \
        "{\n" \
        "<%REAL_TYPE%> temp = 0.0;\n" \
        "<%UPDATE%>" \
        "}\n" \
        "/* Special case for top and bottom levels */\n" \
        "if ( threadIdx.x == 0 )\n" \
        "{\n" \
        "fct_ttf_max[item] = tvert_max[0] - fct_lo[item];\n" \
        "fct_ttf_min[item] = tvert_min[0] - fct_lo[item];\n" \
        "fct_ttf_max[item + (maxNodeLevel - 1)] = tvert_max[maxNodeLevel - 1] - fct_lo[item + (maxNodeLevel - 1)];\n" \
        "fct_ttf_min[item + (maxNodeLevel - 1)] = tvert_min[maxNodeLevel - 1] - fct_lo[item + (maxNodeLevel - 1)];\n" \
        "}\n" \
        "}\n"
    reduction_block = \
        "item = ((elements_in_node[(blockIdx.x * maxElements)] - 1) * maxLevels * 2) + ((level + <%OFFSET%>) * 2);\n" \
        "tvert_max_temp = UV_rhs[item];\n" \
        "tvert_min_temp = UV_rhs[item + 1];\n" \
        "for ( <%INT_TYPE%> element = 1; element < number_elements_in_node[blockIdx.x]; element++ )\n" \
        "{\n" \
        "item = ((elements_in_node[(blockIdx.x * maxElements) + element] - 1) * maxLevels * 2) + ((level + <%OFFSET%>) * 2);\n" \
        "tvert_max_temp = <%FMAX%>(tvert_max_temp, UV_rhs[item]);\n" \
        "tvert_min_temp = <%FMIN%>(tvert_min_temp, UV_rhs[item + 1]);\n" \
        "}\n" \
        "tvert_max[level + <%OFFSET%>] = tvert_max_temp;\n" \
        "tvert_min[level + <%OFFSET%>] = tvert_min_temp;\n"
    reduction_block_vector = \
        "item = ((elements_in_node[(blockIdx.x * maxElements)] - 1) * maxLevels) + (level + <%OFFSET%>);\n" \
        "tvert_max_temp = (UV_rhs[item]).x;\n" \
        "tvert_min_temp = (UV_rhs[item]).y;\n" \
        "for ( <%INT_TYPE%> element = 1; element < number_elements_in_node[blockIdx.x]; element++ )\n" \
        "{\n" \
        "item = ((elements_in_node[(blockIdx.x * maxElements) + element] - 1) * maxLevels) + (level + <%OFFSET%>);\n" \
        "tvert_max_temp = <%FMAX%>(tvert_max_temp, (UV_rhs[item]).x);\n" \
        "tvert_min_temp = <%FMIN%>(tvert_min_temp, (UV_rhs[item]).y);\n" \
        "}\n" \
        "tvert_max[level + <%OFFSET%>] = tvert_max_temp;\n" \
        "tvert_min[level + <%OFFSET%>] = tvert_min_temp;\n"
    update_block = \
        "temp = <%FMAX%>(tvert_max[(level + <%OFFSET%>) - 1], tvert_max[level + <%OFFSET%>]);\n" \
        "temp = <%FMAX%>(temp, tvert_max[(level + <%OFFSET%>) + 1]);\n" \
        "fct_ttf_max[item + level + <%OFFSET%>] = temp - fct_lo[item + level + <%OFFSET%>];\n" \
        "temp = <%FMIN%>(tvert_min[(level + <%OFFSET%>) - 1], tvert_min[level + <%OFFSET%>]);\n" \
        "temp = <%FMIN%>(temp, tvert_min[(level + <%OFFSET%>) + 1]);\n" \
        "fct_ttf_min[item + level + <%OFFSET%>] = temp - fct_lo[item + level + <%OFFSET%>];\n"
    reduction = str()
    update = str()
    for tile in range(0, tuning_parameters["tiling_x"]):
        if tile == 0:
            if tuning_parameters["vector_size"] == 1:
                reduction = reduction + reduction_block.replace(" + <%OFFSET%>", "")
            else:
                reduction = reduction + reduction_block_vector.replace(" + <%OFFSET%>", "")
            update = update + update_block.replace(" + <%OFFSET%>", "")
        else:
            offset = tuning_parameters["block_size_x"] * tile
            if tuning_parameters["vector_size"] == 1:
                reduction = reduction + "if (level + {} < maxNodeLevel - 1)\n{{\n{}}}\n".format(str(offset), reduction_block.replace("<%OFFSET%>", str(offset)))
            else:
                reduction = reduction + "if (level + {} < maxNodeLevel - 1)\n{{\n{}}}\n".format(str(offset), reduction_block_vector.replace("<%OFFSET%>", str(offset)))
            update = update + "if (level + {} < maxNodeLevel - 2)\n{{\n{}}}\n".format(str(offset), update_block.replace("<%OFFSET%>", str(offset)))
    code = code.replace("<%REDUCTION%>", reduction)
    code = code.replace("<%UPDATE%>", update)
    if tuning_parameters["tiling_x"] > 1:
        code = code.replace("<%BLOCK_SIZE%>", str(tuning_parameters["block_size_x"] * tuning_parameters["tiling_x"]))
    else:
        code = code.replace("<%BLOCK_SIZE%>", str(tuning_parameters["block_size_x"]))
    if tuning_parameters["real_type"] == "float":
        code = code.replace("<%FMAX%>", "fmaxf")
        code = code.replace("<%FMIN%>", "fminf")
    elif tuning_parameters["real_type"] == "double":
        code = code.replace("<%FMAX%>", "fmax")
        code = code.replace("<%FMIN%>", "fmin")
    else:
        raise ValueError
    code = code.replace("<%REAL_TYPE%>", tuning_parameters["real_type"])
    code = code.replace("<%INT_TYPE%>", tuning_parameters["int_type"].replace("_", " "))
    if tuning_parameters["vector_size"] == 1:
        code = code.replace("<%VECTOR_SIZE%>", "")
    else:
        code = code.replace("<%VECTOR_SIZE%>", str(tuning_parameters["vector_size"]))
    return code

def reference(vlimit, nodes, levels, max_levels, elements_in_node, number_elements_in_node, max_elements_in_node, uv_rhs, fct_ttf_max, fct_ttf_min, fct_lo, real_type):
    if vlimit == 1:
        for node in range(0, nodes):
            tvert_max = list()
            tvert_min = list()
            for level in range(0, levels[node] - 1):
                max_temp = numpy.finfo(real_type).min
                min_temp = numpy.finfo(real_type).max
                for element in range(0, number_elements_in_node[node]):
                    item = ((elements_in_node[(node * max_elements_in_node) + element] - 1) * max_levels * 2) + (level * 2)
                    max_temp = max(max_temp, uv_rhs[item])
                    min_temp = min(min_temp, uv_rhs[item + 1])
                tvert_max.append(max_temp)
                tvert_min.append(min_temp)
            tvert_max.append(0)
            tvert_min.append(0)
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

def tune(elements, nodes, max_elements, max_levels, vlimit, max_tile, real_type):
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
    shared_memory_args = dict()
    shared_memory_args["size"] = 2 * max_levels * numpy.dtype(numpy_real_type).itemsize
    constraints = list()
    constraints.append("block_size_x * tiling_x <= max_levels")
    # Memory allocation and initialization
    fct_ttf_max = numpy.zeros(nodes * max_levels).astype(numpy_real_type)
    fct_ttf_max_control = numpy.zeros_like(fct_ttf_max).astype(numpy_real_type)
    fct_ttf_min = numpy.zeros_like(fct_ttf_max).astype(numpy_real_type)
    fct_ttf_min_control = numpy.zeros_like(fct_ttf_min).astype(numpy_real_type)
    uv_rhs = numpy.random.randn(elements * max_levels * 2).astype(numpy_real_type)
    fct_lo = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    levels = numpy.zeros(nodes).astype(numpy.int32)
    elements_in_node = numpy.zeros(nodes * max_elements).astype(numpy.int32)
    number_elements_in_node = numpy.zeros(nodes).astype(numpy.int32)
    for node in range(0, nodes):
        levels[node] = numpy.random.randint(3, max_levels)
        number_elements_in_node[node] = numpy.random.randint(3, max_elements)
        for element in range(0, number_elements_in_node[node]):
            elements_in_node[(node * max_elements) + element] = numpy.random.randint(1, elements + 1)
    arguments = [numpy.int32(max_levels), numpy.int32(max_elements), levels, elements_in_node, number_elements_in_node, uv_rhs, fct_ttf_max, fct_ttf_min, fct_lo]
    # Reference
    reference(vlimit, nodes, levels, max_levels, elements_in_node, number_elements_in_node, max_elements, uv_rhs, fct_ttf_max_control, fct_ttf_min_control, fct_lo, numpy_real_type)
    arguments_control = [None, None, None, None, None, None, fct_ttf_max_control, fct_ttf_min_control, None]
    # Tuning
    results, environment = tune_kernel("fct_ale_a3", generate_code, "{} * block_size_x".format(nodes), arguments, tuning_parameters, smem_args=shared_memory_args, lang="CUDA", answer=arguments_control, restrictions=constraints, quiet=True)
    return results

def parse_command_line():
    parser = argparse.ArgumentParser(description="FESOM2 FCT ALE A3")
    parser.add_argument("--elements", help="The number of elements.", type=int, required=True)
    parser.add_argument("--nodes", help="The number of nodes.", type=int, required=True)
    parser.add_argument("--max_elements", help="The maximum number of elements a node is part of.", type=int, required=True)
    parser.add_argument("--max_levels", help="The maximum number of vertical levels per element.", type=int, required=True)
    parser.add_argument("--vlimit", help="The version of vertical limit.", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--max_tile", help="The maximum tiling factor.", type=int, default=2)
    parser.add_argument("--real_type", help="The floating point type to use.", choices=["float", "double"], type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    command_line = parse_command_line()
    results = tune(command_line.elements, command_line.nodes, command_line.max_elements, command_line.max_levels, command_line.vlimit, command_line.max_tile, command_line.real_type)
    best_configuration = min(results, key=lambda x : x["time"])
    print("/* Block size X: {} */".format(best_configuration["block_size_x"]))
    print(generate_code(best_configuration))