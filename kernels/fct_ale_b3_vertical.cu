/* Block size X: 32 */      
__global__ void fct_ale_b3_vertical(const int maxLevels, const int * __restrict__ nLevels, double * __restrict__ fct_adf_v, const double * __restrict__ fct_plus, const double * __restrict__ fct_minus)
{
const int node = (blockIdx.x * maxLevels);
const int flux_index = (blockIdx.x * (maxLevels + 1));
const int maxNodeLevel = nLevels[blockIdx.x] - 1;

/* Intermediate levels */
for ( int level = threadIdx.x + 1; level < maxNodeLevel; level += 32 )
{
    double flux = 0.0;
    double ae_plus = 0.0;
    double ae_minus = 0.0;
    flux = fct_adf_v[flux_index + level];
    ae_plus = 1.0;
    ae_minus = 1.0;
    ae_plus = fmin(ae_plus, fct_minus[node + (level) - 1]);
    ae_minus = fmin(ae_minus, fct_minus[node + (level)]);
    ae_plus = fmin(ae_plus, fct_plus[node + (level)]);
    ae_minus = fmin(ae_minus, fct_plus[node + (level) - 1]);
    if ( signbit(flux) == 0 )
    {
        flux *= ae_plus;
    }
    else
    {
        flux *= ae_minus;
    }
    fct_adf_v[flux_index + level] = flux;
}
/* Top level */
if ( threadIdx.x == 0 )
{
    double flux = fct_adf_v[flux_index];
    double ae = 1.0;
    if ( signbit(flux) == 0 )
    {
        ae = fmin(ae, fct_plus[node]);
    }
    else
    {
        ae = fmin(ae, fct_minus[node]);
    }
    fct_adf_v[flux_index] = ae * flux;
}
}