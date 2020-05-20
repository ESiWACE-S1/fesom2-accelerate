/* Block size X: 32 */
__global__ void fct_ale_pre_comm(   const int max_levels,
                                    const int max_num_elems,
                                    const int * __restrict__ node_levels,
                                    const int * __restrict__ elem_levels,
                                    const int * __restrict__ node_elems,
                                    const int * __restrict__ node_num_elems,
                                    const int * __restrict__ elem_nodes,
                                    const double * __restrict__ fct_low_order, 
                                    const double * __restrict__ ttf,
                                    const double * __restrict__ fct_adf_v,
                                    const double * __restrict__ fct_adf_h,
                                    double * __restrict__ UVrhs,
                                    double * __restrict__ fct_ttf_max, 
                                    double * __restrict__ fct_ttf_min,
                                    double * __restrict__ tvert_max,
                                    double * __restrict__ tvert_min,
                                    double * __restrict__ fct_plus,
                                    double * __restrict__ fct_minus)
{
    const int node = (blockIdx.x * max_levels);
    const int numelems = node_num_elems[blockIdx.x];

    for ( int level = threadIdx.x; level < node_levels[blockIdx.x]; level += 32 )
    {
        double tvmax = -bignr;
        double tvmin = bignr;
        for ( int elem = 0; elem < numelems ; elem++ )
        {
            elem_index = node_elems[blockIdx.x * max_num_elems + elem] - 1;
            int node_indices[3] = { (elem_nodes[3 * elem_index] - 1) * max_levels + level,
                                    (elem_nodes[3 * elem_index + 1] - 1) * max_levels + level,
                                    (elem_nodes[3 * elem_index + 2] - 1) * max_levels + level};
            double fctttfmax[3]  = {fmax(fct_low_order[node_indices[0]], ttf[node_indices[0]]), 
                                    fmax(fct_low_order[node_indices[1]], ttf[node_indices[1]]), 
                                    fmax(fct_low_order[node_indices[2]], ttf[node_indices[2]])};
            double fctttfmin[3]  = {fmin(fct_low_order[node_indices[0]], ttf[node_indices[0]]), 
                                    fmin(fct_low_order[node_indices[1]], ttf[node_indices[1]]), 
                                    fmin(fct_low_order[node_indices[2]], ttf[node_indices[2]])};
            double uvrhs1, uvrhs2;
            if(level < elem_levels[elem_index] - 1)
            {
                uvrhs1 = fmax(fctttfmax[0], fmax(fctttfmax[1], fctttfmax[2]));
                uvrhs2 = fmin(fctttfmin[0], fmin(fctttfmin[1], fctttfmin[2]));
            }
            else
            {
                uvrhs1 = bignr;
                uvrhs2 = -bignr;
            }
            tvmax = fmax(uvrhs1, tvmax);
            tvmin = fmin(uvrhs2, tvmin);
            UVrhs[2 * elem_index * max_levels + level] = uvrhs1;
            UVrhs[2 * elem_index * max_levels + level + 1] = uvrhs2;
        }
        tvert_max[node + level] = tvmax;
        tvert_min[node + level] = tvmin;
    }
    __syncthreads();
    for ( int level = threadIdx.x; level < node_levels[blockIdx.x]; level += 32 )
    {
        if(level == 0 or level == node_levels[blockIdx.x] - 2)
        {
            fct_ttf_max[node + level] = fmax(fct_low_order[node + level], ttf[node + level]);
            fct_ttf_min[node + level] = fmin(fct_low_order[node + level], ttf[node + level]);
        }
        else
        {
            fct_ttf_max[node + level] = fmax(tvert_max[node + level], fmax(tvert_max[node + level - 1],
                                                                           tvert_max[node + level + 1]));
            fct_ttf_min[node + level] = fmin(tvert_min[node + level], fmin(tvert_min[node + level - 1],
                                                                           tvert_min[node + level + 1]));
        }
        int adf_index = blockIdx.x * (max_levels + 1) + level;
        fct_plus[node + level]  = fmax(0.,fct_adf_v[adf_index]) + fmax(0.,-fct_adf_v[adf_index + 1]);
        fct_minus[node + level] = fmin(0.,fct_adf_v[adf_index]) + fmin(0.,-fct_adf_v[adf_index + 1]);
    }
}