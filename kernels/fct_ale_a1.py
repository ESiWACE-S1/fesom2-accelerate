
from kernel_tuner import tune_kernel


def generate_code():
    code = \
        "__global__ fct_ale_a1()\n" \
        "{\n" \
        "<%INT_TYPE%> item = 0;\n" \
        "<%REAL_TYPE%> fct_low_order_item = 0;" \
        "<%REAL_TYPE%> ttf_item = 0;" \
        "" \
        "if ( <%CONDITION%> )" \
        "{\n" \
        "fct_low_order_item = fct_low_order[item];" \
        "ttf_item = ttf[item];" \
        "fct_ttf_max[item] = fmax(fct_low_order_item, ttf_item);" \
        "fct_ttf_min[item] = fmin(fct_low_order_item, ttf_item);" \
        "}\n" \
        "}\n"
    return code

def tune_code():
    pass
