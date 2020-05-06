
/*
 Code generated running fct_ale_a1.py for 5576658 nodes and 48 maximum levels.
*/
__global__ void fct_ale_a1(const double * __restrict__ fct_low_order, const double * __restrict__ ttf, const int * __restrict__ nLevels, double * fct_ttf_max, double * fct_ttf_min)
{
const unsigned int node = (blockIdx.x * 48);

for ( unsigned int level = threadIdx.x; level < nLevels[blockIdx.x]; level += 32 )
{
double fct_low_order_item = 0;
double ttf_item = 0;
fct_low_order_item = fct_low_order[node + level];
ttf_item = ttf[node + level];
fct_ttf_max[node + level] = fmax(fct_low_order_item, ttf_item);
fct_ttf_min[node + level] = fmin(fct_low_order_item, ttf_item);
}
}