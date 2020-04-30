
from kernel_tuner import tune_kernel
import numpy


def generate_code(configuration):
    code = \
        "__global__ void fct_ale_a1(<%REAL_TYPE%> * fct_low_order, <%REAL_TYPE%> * ttf, int * nLevels, <%REAL_TYPE%> * fct_ttf_max, <%REAL_TYPE%> * fct_ttf_min)\n" \
        "{\n" \
        "<%INT_TYPE%> item = (blockIdx.x * <%MAX_LEVELS%>) + threadIdx.x;\n" \
        "<%REAL_TYPE%> fct_low_order_item = 0;" \
        "<%REAL_TYPE%> ttf_item = 0;" \
        "" \
        "if ( threadIdx.x < nLevels[blockIdx.x] )" \
        "{\n" \
        "fct_low_order_item = fct_low_order[item];" \
        "ttf_item = ttf[item];" \
        "fct_ttf_max[item] = fmax(fct_low_order_item, ttf_item);" \
        "fct_ttf_min[item] = fmin(fct_low_order_item, ttf_item);" \
        "}\n" \
        "}\n"
    code.replace("<%INT_TYPE%>", configuration["int_type"])
    code.replace("<%REAL_TYPE%>", configuration["real_type"])
    code.replace("<%MAX_LEVELS%>", configuration["max_levels"])
    return code

def tune(nodes, max_levels, real_type, numpy_real_type):
    # Tuning and code generation parameters
    tuning_parameters = dict()
    tuning_parameters["max_levels"] = max_levels
    tuning_parameters["block_size_x"] = [32 * i for i in range(1, 33)]
    tuning_parameters["int_type"] = ["unsigned int", "int"]
    tuning_parameters["real_type"] = real_type
    # Memory allocation and initialization
    fct_low_order = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    ttf = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    fct_ttf_max = numpy.zeros(nodes * max_levels)
    fct_ttf_min = numpy.zeros_like(fct_ttf_max)
    levels = numpy.zeros(nodes).astype(numpy.np.int32)
    for node in range(0, nodes):
        levels[node] = numpy.random.randint(0, max_levels)
    arguments = [fct_low_order, ttf, levels, fct_ttf_max, fct_ttf_min]
    # Tuning
    results = tune_kernel("fct_ale_a1", generate_code, nodes * max_levels, arguments, tuning_parameters, verbose=True)
    return results
