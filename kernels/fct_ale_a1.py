
from kernel_tuner import tune_kernel
import numpy

class Kernel:
    def __init__(self):
        self.tuning_parameters = dict()

    def generate_code(self):
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
        code.replace("<%INT_TYPE%>", self.tuning_parameters["int_type"])
        code.replace("<%REAL_TYPE%>", self.tuning_parameters["real_type"])
        code.replace("<%MAX_LEVELS%>", self.tuning_paramenters["max_levels"])
        return code

    def tune(self, nodes, max_levels, real_type, numpy_real_type):
        # Tuning and code generation parameters
        self.tuning_parameters = dict()
        self.tuning_parameters["max_levels"] = [str(max_levels)]
        self.tuning_parameters["block_size_x"] = [32 * i for i in range(1, 33)]
        self.tuning_parameters["int_type"] = ["unsigned int", "int"]
        self.tuning_parameters["real_type"] = [real_type]
        # Memory allocation and initialization
        fct_low_order = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
        ttf = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
        fct_ttf_max = numpy.zeros(nodes * max_levels)
        fct_ttf_min = numpy.zeros_like(fct_ttf_max)
        levels = numpy.zeros(nodes).astype(numpy.int32)
        for node in range(0, nodes):
            levels[node] = numpy.random.randint(0, max_levels)
        arguments = [fct_low_order, ttf, levels, fct_ttf_max, fct_ttf_min]
        # Tuning
        results = tune_kernel("fct_ale_a1", self.generate_code, nodes * max_levels, arguments, self.tuning_parameters, verbose=True, lang="CUDA")
        return results
