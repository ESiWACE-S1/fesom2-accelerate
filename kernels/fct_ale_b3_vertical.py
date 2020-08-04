
from kernel_tuner import tune_kernel
import numpy
import argparse

def generate_code(tuning_parameters):
    code = \
        "__global__ void fct_ale_b3_vertical(const int maxLevels, const int * __restrict__ nLevels, const <%REAL_TYPE%> * __restrict__ fct_adf_v, <%REAL_TYPE%> * __restrict__ fct_plus, <%REAL_TYPE%> * __restrict__ fct_minus)\n" \
        "{\n" \
        "const <%INT_TYPE%> node = (blockIdx.x * maxLevels);\n" \
        "const <%INT_TYPE%> maxNodeLevel = nLevels[blockIdx.x] - 1;\n" \
        "\n" \
        "/* Intermediate levels */" \
        "for ( <%INT_TYPE%> level = threadIdx.x + 1; level < maxNodeLevel; level += <%BLOCK_SIZE%> )\n" \
        "{\n" \
        "<%REAL_TYPE%> flux = 0.0;\n" \
        "<%REAL_TYPE%> ae_plus = 0.0;\n" \
        "<%REAL_TYPE%> ae_minus = 0.0;\n" \
        "<%COMPUTE_BLOCK%>" \
        "}\n" \
        "/* Top level */" \
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

def reference(nodes, levels, max_levels, fct_adf_v, fct_plus, fct_minus, iter_yn):
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

def parse_command_line():
    parser = argparse.ArgumentParser(description="FESOM2 FCT ALE B3 VERTICAL")
    parser.add_argument("--nodes", help="The number of nodes.", type=int, required=True)
    parser.add_argument("--max_levels", help="The maximum number of vertical levels per node.", type=int, required=True)
    parser.add_argument("--max_tile", help="The maximum tiling factor.", type=int, default=2)
    parser.add_argument("--real_type", help="The floating point type to use.", choices=["float", "double"], type=str, required=True)
    parser.add_argument("--verbose", help="Print all kernel configurations.", default=True, action="store_false")
    return parser.parse_args()
