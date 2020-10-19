/* Block size X: 32 */
__global__ void fct_ale_b1_vertical(const int maxLevels, const int * __restrict__ nLevels, const double * __restrict__ fct_adf_v, double * __restrict__ fct_plus, double * __restrict__ fct_minus)
{
const int node = (blockIdx.x * maxLevels);

for ( int level = threadIdx.x; level < nLevels[blockIdx.x] - 1; level += 32 )
{
    double fct_adf_v_level = 0.0;
    double fct_adf_v_nlevel = 0.0;
    int item = blockIdx.x * (maxLevels + 1) + level;
    fct_adf_v_level = fct_adf_v[item];
    fct_adf_v_nlevel = fct_adf_v[item + 1];
    fct_plus[node + level] = fmax(0.0, fct_adf_v_level) + fmax(0.0, -fct_adf_v_nlevel);
    fct_minus[node + level] = fmin(0.0, fct_adf_v_level) + fmin(0.0, -fct_adf_v_nlevel);
}
}