#include <atomicAddFix.h>
/* Block size X: 32 */
__global__ void fct_ale_b1_horizontal(const int maxLevels, const int * __restrict__ nLevels, const int * __restrict__ nodesPerEdge, const int * __restrict__ elementsPerEdge, const double * __restrict__ fct_adf_h, double * __restrict__ fct_plus, double * __restrict__ fct_minus)
{
int levelBound = 0;
const int nodeOne = (nodesPerEdge[(blockIdx.x * 2)] - 1) * maxLevels;
const int nodeTwo = (nodesPerEdge[(blockIdx.x * 2) + 1] - 1) * maxLevels;

/* Compute the upper bound for the level */
levelBound = elementsPerEdge[(blockIdx.x * 2) + 1] - 1;
if ( levelBound > 0 )
{
    levelBound = max(nLevels[elementsPerEdge[(blockIdx.x * 2)] - 1], nLevels[levelBound]);
}
else
{
    levelBound = max(nLevels[elementsPerEdge[(blockIdx.x * 2)] - 1], 0);
}
/* Compute fct_plus and fct_minus */
for ( int level = threadIdx.x; level < levelBound; level += 32 )
{
    double fct_adf_h_value = 0.0;
    fct_adf_h_value = fct_adf_h[(blockIdx.x * maxLevels) + level];
    atomicAdd(&(fct_plus[nodeOne + level]), fmax(0.0, fct_adf_h_value));
    atomicAdd(&(fct_minus[nodeOne + level]), fmin(0.0, fct_adf_h_value));
    atomicAdd(&(fct_plus[nodeTwo + level]), fmax(0.0, -fct_adf_h_value));
    atomicAdd(&(fct_minus[nodeTwo + level]), fmin(0.0, -fct_adf_h_value));
}
}