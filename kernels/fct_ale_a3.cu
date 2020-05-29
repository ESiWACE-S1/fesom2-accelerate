/* Block size X: 32 */
__global__ void fct_ale_a3(const int maxLevels, const int maxElements, const int * __restrict__ nLevels, const int * __restrict__ elements_in_node, const int * __restrict__ number_elements_in_node, const double2 * __restrict__ UV_rhs, double * __restrict__ fct_ttf_max, double * __restrict__ fct_ttf_min, const double * __restrict__ fct_lo)
{
int item = 0;                                   
extern __shared__ double sharedBuffer[];
double * tvert_max = (double *)(sharedBuffer);
double * tvert_min = (double *)(&sharedBuffer[maxLevels]);
/* Compute tvert_max and tvert_min per level */
for ( int level = threadIdx.x; level < nLevels[blockIdx.x]; level += 32 )
{
    double tvert_max_temp = 0.0;
    double tvert_min_temp = 0.0;
    item = (elements_in_node[(blockIdx.x * maxElements)] * maxLevels) + (level);
    tvert_max_temp = (UV_rhs[item]).x;
    tvert_min_temp = (UV_rhs[item]).y;
    for ( int element = 1; element < number_elements_in_node[blockIdx.x]; element++ )
    {
        item = (elements_in_node[(blockIdx.x * maxElements) + element] * maxLevels) + (level);
        tvert_max_temp = fmax(tvert_max_temp, (UV_rhs[item]).x);
        tvert_min_temp = fmin(tvert_min_temp, (UV_rhs[item]).y);
    }
    tvert_max[level] = tvert_max_temp;
    tvert_min[level] = tvert_min_temp;
}
__syncthreads();
/* Update fct_ttf_max and fct_ttf_min per level */
item = blockIdx.x * maxLevels;
for ( int level = threadIdx.x + 1; level < nLevels[blockIdx.x] - 2; level += 32 )
{
    double temp = 0.0;
    temp = fmax(tvert_max[(level) - 1], tvert_max[level]);
    temp = fmax(temp, tvert_max[(level) + 1]); 
    fct_ttf_max[item + level] = temp - fct_lo[item + level];
    temp = fmin(tvert_min[(level) - 1], tvert_min[level]);
    temp = fmin(temp, tvert_min[(level) + 1]); 
    fct_ttf_min[item + level] = temp - fct_lo[item + level];
}
if ( threadIdx.x == 0 )
{
    fct_ttf_max[item] = tvert_max[0] - fct_lo[item];
    fct_ttf_min[item] = tvert_min[0] - fct_lo[item];
    fct_ttf_max[item + (nLevels[blockIdx.x] - 1)] = tvert_max[nLevels[blockIdx.x] - 1] - fct_lo[item + (nLevels[blockIdx.x] - 1)];
    fct_ttf_min[item + (nLevels[blockIdx.x] - 1)] = tvert_min[nLevels[blockIdx.x] - 1] - fct_lo[item + (nLevels[blockIdx.x] - 1)];
}
}