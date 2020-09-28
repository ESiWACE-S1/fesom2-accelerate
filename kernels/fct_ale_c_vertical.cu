/* Block size X: 32 */
__global__ void fct_ale_c_vertical(const int maxLevels, const int * __restrict__ nLevels, double * __restrict__ del_ttf_advvert, const double * __restrict__ ttf, const double * __restrict__ hnode, const double * __restrict__ fct_LO, const double * __restrict__ hnode_new, const double * __restrict__ fct_adf_v, const double dt, const double * __restrict__ area)
{
const int node = (blockIdx.x * maxLevels);
const int maxNodeLevel = nLevels[blockIdx.x] - 1;

for ( int level = threadIdx.x; level < maxNodeLevel; level += 32 )
{
    double temp = 0;
    temp = del_ttf_advvert[node + level] - (ttf[node + level] * hnode[node + level]);
    temp += fct_LO[node + level] * hnode_new[node + level];
    temp += (fct_adf_v[node + level] - fct_adf_v[node + level + 1]) * (dt / area[node + level]);
    del_ttf_advvert[node + level] = temp;
}
}