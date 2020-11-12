__global__ void fct_ale_a1(const int maxLevels, const {{ real_type }} * __restrict__ fct_low_order, const {{ real_type }} * __restrict__ ttf, const int * __restrict__ nLevels, {{ real_type }} * __restrict__ fct_ttf_max, {{ real_type }} * __restrict__ fct_ttf_min)
{
const {{ int_type }} node = (blockIdx.x * maxLevels);
const int maxNodeLevel = nLevels[blockIdx.x] - 1;
{{ real_type }} fct_low_order_item = 0;
{{ real_type }} ttf_item = 0;

for ( {{ int_type }} level = threadIdx.x; level < maxNodeLevel; level += {{ block_size }} )
{
    {% for i in range(tiling_x) %}
    {% set offset = block_size_x * i %}
    if ( level + {{ offset}} < maxNodeLevel ) {
        fct_low_order_item = fct_low_order[node + level + {{ offset }}];
        ttf_item = ttf[node + level + {{ offset }}];
        fct_ttf_max[node + level + {{ offset }}] = {{ fmax }}(fct_low_order_item, ttf_item);
        fct_ttf_min[node + level + {{ offset }}] = {{ fmin }}(fct_low_order_item, ttf_item);
    }
    {% endfor %}
}
}