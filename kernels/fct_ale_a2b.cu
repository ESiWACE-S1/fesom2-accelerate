/* Block size X: 32 */
__global__ void fct_ale_a2b(const int maxLevels, const int * __restrict__ nLevels, const int * __restrict__ elementNodes, double * __restrict__ UV_rhs, const double * __restrict__ fct_ttf_max, const double * __restrict__ fct_ttf_min, const double big_number)
{
    const unsigned int element_index = (blockIdx.x * maxLevels);
    const unsigned int element_node0_index = (elementNodes[(blockIdx.x * 3)] - 1) * maxLevels;
    const unsigned int element_node1_index = (elementNodes[(blockIdx.x * 3) + 1] - 1) * maxLevels;
    const unsigned int element_node2_index = (elementNodes[(blockIdx.x * 3) + 2] - 1) * maxLevels;
    for ( unsigned int level = threadIdx.x; level < maxLevels + 1; level += 32 )
    {
        if ( level < nLevels[blockIdx.x] - 1 )
        {
            double temp1 = fmax(fct_ttf_max[element_node0_index + level], fct_ttf_max[element_node1_index + level]);
            temp1 = fmax(temp1, fct_ttf_max[element_node2_index + level]);
            double temp2 = fmin(fct_ttf_min[element_node0_index + level], fct_ttf_min[element_node1_index + level]);
            temp2 = fmin(temp2, fct_ttf_min[element_node2_index + level]);
            UV_rhs[2*(element_index + level)] = temp1;
            UV_rhs[2*(element_index + level) + 1] = temp2;
        }
        else if ( level < maxLevels - 1 )
        {
            UV_rhs[2*(element_index + level)] = -big_number;
            UV_rhs[2*(element_index + level) + 1] = big_number;
        }
    }
}
