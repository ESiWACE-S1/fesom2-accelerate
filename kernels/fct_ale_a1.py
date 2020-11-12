
from kernel_tuner import tune_kernel
from jinja2 import Environment, FileSystemLoader
import numpy
import argparse
import json


def generate_code(tuning_parameters):
    template_loader = FileSystemLoader("./templates")
    template_environment = Environment(loader=template_loader)
    template = template_environment.get_template("fct_ale_a1_template.cu")
    if tuning_parameters["tiling_x"] > 1:
        block_size = tuning_parameters["block_size_x"] * tuning_parameters["tiling_x"]
    else:
        block_size = tuning_parameters["block_size_x"]
    if tuning_parameters["real_type"] == "float":
        fmax = "fmaxf"
        fmin = "fminf"
    elif tuning_parameters["real_type"] == "double":
        fmax = "fmax"
        fmin = "fmin"
    else:
        raise ValueError
    int_type = tuning_parameters["int_type"].replace("_", " ")
    real_type = tuning_parameters["real_type"]
    return template.render(real_type=real_type, int_type=int_type, block_size_x = tuning_parameters["block_size_x"], block_size=block_size, tiling_x = tuning_parameters["tiling_x"], fmax=fmax, fmin=fmin)


def reference(nodes, levels, max_levels, fct_low_order, ttf, fct_ttf_max, fct_ttf_min):
    for node in range(0, nodes):
        for level in range(0, levels[node] - 1):
            item = (node * max_levels) + level
            fct_ttf_max[item] = max(fct_low_order[item], ttf[item])
            fct_ttf_min[item] = min(fct_low_order[item], ttf[item])


def tune(nodes, max_levels, max_tile, real_type, quiet=True):
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
    if (max_levels <= 1024) and (max_levels not in tuning_parameters["block_size_x"]):
        tuning_parameters["block_size_x"].append(max_levels)
    tuning_parameters["tiling_x"] = [i for i in range(1, max_tile)]
    constraints = list()
    constraints.append("block_size_x * tiling_x <= max_levels")
    # Memory allocation and initialization
    fct_low_order = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    ttf = numpy.random.randn(nodes * max_levels).astype(numpy_real_type)
    fct_ttf_max = numpy.zeros(nodes * max_levels).astype(numpy_real_type)
    fct_ttf_min = numpy.zeros_like(fct_ttf_max).astype(numpy_real_type)
    fct_ttf_max_control = numpy.zeros_like(fct_ttf_max).astype(numpy_real_type)
    fct_ttf_min_control = numpy.zeros_like(fct_ttf_min).astype(numpy_real_type)
    levels = numpy.zeros(nodes).astype(numpy.int32)
    used_levels = 0
    for node in range(0, nodes):
        levels[node] = numpy.random.randint(3, max_levels)
        used_levels = used_levels + (levels[node] - 1)
    arguments = [numpy.int32(max_levels), fct_low_order, ttf, levels, fct_ttf_max, fct_ttf_min]
    # Reference
    reference(nodes, levels, max_levels, fct_low_order, ttf, fct_ttf_max_control, fct_ttf_min_control)
    arguments_control = [None, None, None, None, fct_ttf_max_control, fct_ttf_min_control]
    # Tuning
    tuning_results, _ = tune_kernel("fct_ale_a1", generate_code, "{} * block_size_x".format(nodes), arguments, tuning_parameters, lang="CUDA", answer=arguments_control, restrictions=constraints, quiet=quiet)
    # Memory bandwidth
    memory_bytes = ((nodes * 4) + (used_levels * 4 * numpy.dtype(numpy_real_type).itemsize))
    for result in tuning_results:
        result["memory_bandwidth"] = memory_bytes / (result["time"] / 10**3)
    return tuning_results


def parse_command_line():
    parser = argparse.ArgumentParser(description="FESOM2 FCT ALE A1")
    parser.add_argument("--nodes", help="The number of nodes.", type=int, required=True)
    parser.add_argument("--max_levels", help="The maximum number of vertical levels per node.", type=int, required=True)
    parser.add_argument("--max_tile", help="The maximum tiling factor.", type=int, default=2)
    parser.add_argument("--real_type", help="The floating point type to use.", choices=["float", "double"], type=str, required=True)
    parser.add_argument("--verbose", help="Print all kernel configurations.", default=True, action="store_false")
    parser.add_argument("--store", help="Store performance results in a JSON file.", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    command_line = parse_command_line()
    results = tune(command_line.nodes, command_line.max_levels, command_line.max_tile, command_line.real_type, command_line.verbose)
    best_configuration = min(results, key=lambda x: x["time"])
    print("/* Memory bandwidth: {:.2f} GB/s */".format(best_configuration["memory_bandwidth"] / 10**9))
    print("/* Block size X: {} */".format(best_configuration["block_size_x"]))
    print(generate_code(best_configuration))
    if command_line.store:
        try:
            with open("fct_ale_a1_{}_{}_{}.json".format(command_line.nodes, command_line.max_levels, command_line.real_type), "x") as fp:
                json.dump(results, fp)
        except FileExistsError:
            print("Impossible to save the results, a results file already exists for a similar experiment.")
