#include <atomicAddFix.h>
/* Block size X: 32 */
__global__ void fct_ale_c_horizontal(const int maxLevels, const int * __restrict__ nLevels, const int * __restrict__ nodesPerEdge, const int * __restrict__ elementsPerEdge, double * __restrict__ del_ttf_advhoriz, const double * __restrict__ fct_adf_h, const double dt, const double * __restrict__ area)
{
const int edge = blockIdx.x * 2;
int levelBound = 0;
const int nodeOne = (nodesPerEdge[edge] - 1) * maxLevels;
const int nodeTwo = (nodesPerEdge[edge + 1] - 1) * maxLevels;

/* Compute the upper bound for the level */
levelBound = elementsPerEdge[edge + 1];
if ( levelBound > 0 )
{
    levelBound = max(nLevels[(elementsPerEdge[edge]) - 1], nLevels[levelBound - 1]);
}
else
{
    levelBound = max(nLevels[(elementsPerEdge[edge]) - 1], 0);
}

for ( int level = threadIdx.x; level < levelBound - 1; level += 32 )
{
    double fct_adf_h_item = 0;
    fct_adf_h_item = fct_adf_h[(blockIdx.x * maxLevels) + level];
    atomicAdd(&(del_ttf_advhoriz[nodeOne + level]), (fct_adf_h_item * (dt / area[nodeOne + level])));
    atomicAdd(&(del_ttf_advhoriz[nodeTwo + level]), -(fct_adf_h_item * (dt / area[nodeTwo + level])));
}
}