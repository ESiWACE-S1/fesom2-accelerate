
from kernel_tuner import tune_kernel
import numpy

def generate_code(tuning_parameters):
    code = \
        "__global__ void fct_ale_a1(const <%REAL_TYPE%> * __restrict__ fct_low_order, const <%REAL_TYPE%> * __restrict__ ttf, const int * __restrict__ nLevels, <%REAL_TYPE%> * fct_ttf_max, <%REAL_TYPE%> * fct_ttf_min)\n" \
        "{\n" \
        "<%INT_TYPE%> item = (blockIdx.x * <%MAX_LEVELS%>) + threadIdx.x;\n" \
        "<%REAL_TYPE%> fct_low_order_item = 0;\n" \
        "<%REAL_TYPE%> ttf_item = 0;\n" \
        "\n" \
        "if ( threadIdx.x < nLevels[blockIdx.x] )\n" \
        "{\n" \
        "fct_low_order_item = fct_low_order[item];\n" \
        "ttf_item = ttf[item];\n" \
        "fct_ttf_max[item] = fmax(fct_low_order_item, ttf_item);\n" \
        "fct_ttf_min[item] = fmin(fct_low_order_item, ttf_item);\n" \
        "}\n" \
        "}\n"
    code = code.replace("<%INT_TYPE%>", tuning_parameters["int_type"].replace("_", " "))
    code = code.replace("<%REAL_TYPE%>", tuning_parameters["real_type"])
    code = code.replace("<%MAX_LEVELS%>", tuning_parameters["max_levels"])
    return code

def reference(nodes, levels, max_levels, fct_low_order, ttf, fct_ttf_max, fct_ttf_min):
    for node in range(0, nodes):
        for level in range(0, levels[node]):
            item = (node * max_levels) + level
            fct_ttf_max[item] = numpy.max(fct_low_order[item], ttf[item])
            fct_ttf_min[item] = numpy.min(fct_low_order[item], ttf[item])

def verify(control_data, data, atol=None):
    return numpy.allclose(control_data, data, atol)

def tune(nodes, max_levels, real_type):
    numpy_real_type = None
    # Tuning and code generation parameters
    tuning_parameters = dict()
    tuning_parameters["max_levels"] = [str(max_levels)]
    tuning_parameters["block_size_x"] = [32 * i for i in range(1, 33)]
    tuning_parameters["int_type"] = ["unsigned_int", "int"]
    tuning_parameters["real_type"] = [real_type]
    # Memory allocation and initialization
    if real_type == "float":
        numpy_real_type = numpy.float32
    elif real_type == "double":
        numpy_real_type = numpy.float64
    else:
        raise ValueError
    fct_low_order = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    ttf = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    fct_ttf_max = numpy.zeros(nodes * max_levels)
    fct_ttf_min = numpy.zeros_like(fct_ttf_max)
    fct_ttf_max_control = numpy.zeros_like(fct_ttf_max)
    fct_ttf_min_control = numpy.zeros_like(fct_ttf_min)
    levels = numpy.zeros(nodes).astype(numpy.int32)
    for node in range(0, nodes):
        levels[node] = numpy.random.randint(0, max_levels)
    arguments = [fct_low_order, ttf, levels, fct_ttf_max, fct_ttf_min]
    # Reference
    reference(nodes, levels, max_levels, fct_low_order, ttf, fct_ttf_max_control, fct_ttf_min_control)
    arguments_control = [None, None, None, fct_ttf_max_control, fct_ttf_min_control]
    # Tuning
    results = tune_kernel("fct_ale_a1", generate_code, nodes * max_levels, arguments, tuning_parameters, verbose=True, lang="CUDA", answer=arguments_control, verify=verify)
    return results