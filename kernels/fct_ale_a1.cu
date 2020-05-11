/* Block size X: 32 */
__global__ void fct_ale_a1(const int maxLevels, const double * __restrict__ fct_low_order, const double * __restrict__ ttf, const int * __restrict__ nLevels, double * __restrict__ fct_ttf_max, double * __restrict__ fct_ttf_min)
{
const int node = (blockIdx.x * maxLevels);

for ( int level = threadIdx.x; level < nLevels[blockIdx.x]; level += 32 )
{
double fct_low_order_item = 0;
double ttf_item = 0;
fct_low_order_item = fct_low_order[node + level];
ttf_item = ttf[node + level];
fct_ttf_max[node + level] = fmax(fct_low_order_item, ttf_item);
fct_ttf_min[node + level] = fmin(fct_low_order_item, ttf_item);
}
}