/* Block size X: 32 */
__global__ void fct_ale_b2(const int maxLevels, const double dt, const double fluxEpsilon, const int * __restrict__ nLevels, const double * __restrict__ area_inv, const double * __restrict__ fct_ttf_max, const double * __restrict__ fct_ttf_min, double * __restrict__ fct_plus, double * __restrict__ fct_minus)
{
int index = 0;
double area_item = 0;
for ( int level = threadIdx.x; level < nLevels[blockIdx.x]; level += 32 )
{
    index = (blockIdx.x * maxLevels) + level;
    area_item = area_inv[index];
    fct_plus[index] = fmin(1.0, fct_ttf_max[index] / (fct_plus[index] * dt * area_item + fluxEpsilon));
    fct_minus[index] = fmin(1.0, fct_ttf_min[index] / (fct_minus[index] * dt * area_item - fluxEpsilon));
}
}