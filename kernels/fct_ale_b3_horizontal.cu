/* Block size X: 32 */
__global__ void fct_ale_b3_horizontal(const int maxLevels, const int * __restrict__ nLevels, const int * __restrict__ nodesPerEdge, const int * __restrict__ elementsPerEdge, double * __restrict__ fct_adf_h, const double * __restrict__ fct_plus, const double * __restrict__ fct_minus)
{
int levelBound = 0;
const int nodeOne = (nodesPerEdge[blockIdx.x * 2] - 1) * maxLevels;
const int nodeTwo = (nodesPerEdge[blockIdx.x * 2 + 1] - 1) * maxLevels;

/* Compute the upper bound for the level */
const int elemOne = elementsPerEdge[blockIdx.x * 2];
const int elemTwo = elementsPerEdge[blockIdx.x * 2 + 1];
if ( elemTwo > 0 )
{
    levelBound = max(nLevels[elemOne - 1] - 1, nLevels[elemTwo - 1] - 1);
}
else
{
    levelBound = max(nLevels[elemOne - 1] - 1, 0);
}

for ( int level = threadIdx.x; level < levelBound; level += 32 )
{
    double flux = 0.0;
    double ae_plus = 0.0;
    double ae_minus = 0.0;
    flux = fct_adf_h[(blockIdx.x * maxLevels) + level];
    ae_plus = 1.0;
    ae_minus = 1.0;
    ae_plus = fmin(ae_plus, fct_plus[nodeOne + (level)]);
    ae_minus = fmin(ae_minus, fct_plus[nodeTwo + (level)]);
    ae_minus = fmin(ae_minus, fct_minus[nodeOne + (level)]);
    ae_plus = fmin(ae_plus, fct_minus[nodeTwo + (level)]);
    if ( signbit(flux) == 0 )
    {
        flux *= ae_plus;
    }
    else
    {
        flux *= ae_minus;
    }
    fct_adf_h[(blockIdx.x * maxLevels) + level] = flux;
}
}
