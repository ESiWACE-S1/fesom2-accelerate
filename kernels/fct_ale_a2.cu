/* Block size X: 32 */
__global__ void fct_ale_a2(const int maxLevels, const int * __restrict__ nLevels, const int * __restrict__ elementNodes, double2 * __restrict__ UV_rhs, const double * __restrict__ fct_ttf_max, const double * __restrict__ fct_ttf_min)
{
const unsigned int element_index = (blockIdx.x * maxLevels);
const unsigned int element_node0_index = (elementNodes[(blockIdx.x * 3)] - 1) * maxLevels;
const unsigned int element_node1_index = (elementNodes[(blockIdx.x * 3) + 1] - 1) * maxLevels;
const unsigned int element_node2_index = (elementNodes[(blockIdx.x * 3) + 2] - 1) * maxLevels;
for ( unsigned int level = threadIdx.x; level < maxLevels + 1; level += 32 )
{
if ( level < nLevels[blockIdx.x] - 1 )
{
double2 temp = make_double2(0.0, 0.0);
temp.x = fmax(fct_ttf_max[element_node0_index + level], fct_ttf_max[element_node1_index + level]);
temp.x = fmax(temp.x, fct_ttf_max[element_node2_index + level]);
temp.y = fmin(fct_ttf_min[element_node0_index + level], fct_ttf_min[element_node1_index + level]);
temp.y = fmin(temp.y, fct_ttf_min[element_node2_index + level]);
UV_rhs[element_index + level] = temp;
}
else if ( level < maxLevels - 1 )
{
UV_rhs[element_index + level] = make_double2(-1.e+3, 1.e+3);
}
}
}