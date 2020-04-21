
#include <fesom2-accelerate.h>

const int TODO_INT = 0;
const real_type TODO_REAL = 0.0;

/**
 3D Flux Corrected Transport scheme
*/
void fct_ale(int myDim_nod2D, int eDim_nod2D, int * nLevels_nod2D, real_type * fct_ttf_max, real_type * fct_ttf_min, real_type * fct_LO, real_type * ttf, int myDim_elem2D, int * nLevels, real_type * UV_rhs, int nl, int vLimit, real_type * tvert_max, real_type * tvert_min, real_type * fct_plus, real_type * fct_minus, real_type * fct_adf_v, int myDim_edge2D, int * edges, int * edge_tri, real_type * fct_adf_h, real_type dt, real_type * area, bool iter_yn, real_type * fct_adf_v2, real_type * fct_adf_h2, real_type * hnode_new, real_type * del_ttf_advvert, real_type * del_ttf_advhoriz, real_type * hnode, int * elem2D_nodes)
{
    real_type flux;
    const real_type flux_eps = 1e-16;
    
    // a1: max, min between old solution and updated low-order solution per node
    int nod2D = myDim_nod2D + eDim_nod2D;
    fct_ale_a1_reference_(&nod2D, nLevels_nod2D, fct_ttf_max, fct_ttf_min, fct_LO, ttf);
    // a2: Admissible increments on elements
    // (only layers below the first and above the last layer)
    // look for max, min bounds for each element --> UV_rhs here auxilary array
    fct_ale_a2_reference_(&myDim_elem2D, &nod2D, &nl, nLevels, UV_rhs, elem2D_nodes, fct_ttf_max, fct_ttf_min);
    if ( vLimit == 1 )
    {
        for ( unsigned int node = 0; node < myDim_nod2D; node++ )
        {
            unsigned int item = 0;

            for( unsigned int node_z = 0; node_z < nLevels_nod2D[node] - 1; node_z++ )
            {
                tvert_max[node_z] = TODO_REAL;
                tvert_min[node_z] = TODO_REAL;
            }
            fct_ttf_max[node] = tvert_max[0] - fct_LO[node];
            fct_ttf_min[node] = tvert_min[0] - fct_LO[node];
            for ( unsigned int node_z = 1; node_z < nLevels_nod2D[node] - 2; node_z++ )
            {
                item = (node_z * TODO_INT) + node;
                fct_ttf_max[item] = TODO_REAL;
                fct_ttf_min[item] = TODO_REAL;
            }
            item = ((nLevels_nod2D[node] - 1) * TODO_INT) + node;
            fct_ttf_max[item] = tvert_max[nLevels_nod2D[node] - 1] - fct_LO[item];
            fct_ttf_min[item] = tvert_min[nLevels_nod2D[node] - 1] - fct_LO[item];
        }
    }
    if ( vLimit == 2 )
    {
        for ( unsigned int node = 0; node < myDim_nod2D; node++ )
        {
            for( unsigned int node_z = 0; node_z < nLevels_nod2D[node] - 1; node_z++ )
            {
                tvert_max[node_z] = TODO_REAL;
                tvert_min[node_z] = TODO_REAL;
            }
            for ( unsigned int node_z = 1; node_z < nLevels_nod2D[node] - 2; node_z++ )
            {
                tvert_max[node_z] = std::max(TODO_REAL, TODO_REAL);
                tvert_min[node_z] = std::min(TODO_REAL, TODO_REAL);
            }
            for( unsigned int node_z = 0; node_z < nLevels_nod2D[node] - 1; node_z++ )
            {
                unsigned int item = (node_z * TODO_INT) + node;

                fct_ttf_max[item] = tvert_max[node_z] - fct_LO[item];
                fct_ttf_min[item] = tvert_min[node_z] - fct_LO[item];
            }
        }
    }
    if ( vLimit == 3 )
    {
        for ( unsigned int node = 0; node < myDim_nod2D; node++ )
        {
            for( unsigned int node_z = 0; node_z < nLevels_nod2D[node] - 1; node_z++ )
            {
                tvert_max[node_z] = TODO_REAL;
                tvert_min[node_z] = TODO_REAL;
            }
            for ( unsigned int node_z = 1; node_z < nLevels_nod2D[node] - 2; node_z++ )
            {
                tvert_max[node_z] = std::min(TODO_REAL, TODO_REAL);
                tvert_min[node_z] = std::max(TODO_REAL, TODO_REAL);
            }
            for( unsigned int node_z = 0; node_z < nLevels_nod2D[node] - 1; node_z++ )
            {
                unsigned int item = (node_z * TODO_INT) + node;

                fct_ttf_max[item] = tvert_max[node_z] - fct_LO[item];
                fct_ttf_min[item] = tvert_min[node_z] - fct_LO[item];
            }
        }
    }
    for ( unsigned int node = 0; node < myDim_nod2D; node++ )
    {
        for ( unsigned int node_z = 0; node_z < nLevels_nod2D[node] - 1; node_z++ )
        {
            unsigned int item = (node_z * TODO_INT) + node;

            fct_plus[item] = 0.0;
            fct_minus[item] = 0.0;
        }
    }
    for ( unsigned int node = 0; node < myDim_nod2D; node++ )
    {
        for ( unsigned int node_z = 0; node_z < nLevels_nod2D[node] - 1; node_z++ )
        {
            unsigned int item = (node_z * TODO_INT) + node;

            fct_plus[item] += std::max(0.0, fct_adf_v[item]) + std::max(0.0, -fct_adf_v[((node_z + 1) * TODO_INT) + node]);
            fct_minus[item] += std::min(0.0, fct_adf_v[item]) + std::min(0.0, -fct_adf_v[((node_z + 1) * TODO_INT) + node]);
        }
    }
    for ( unsigned int edge = 0; edge < myDim_edge2D; edge++ )
    {
        unsigned int node_l1 =0, node_l2 = 0;
        int edgeNodes [2] = {edges[edge], edges[2 * myDim_edge2D]};

        node_l1 = nLevels[edge_tri[edge]] - 1;
        if ( edge_tri[2 * edge] > 0 )
        {
            node_l2 = nLevels[edge_tri[2 * edge]] - 1;
        }
        for ( unsigned int node__z = 0; node__z < std::max(node_l1, node_l2); node__z++ )
        {
            unsigned int item = (node__z * TODO_INT) + edge;

            fct_plus[(node__z * TODO_INT) + edgeNodes[0]] += std::max(0.0, fct_adf_h[item]);
            fct_minus[(node__z * TODO_INT) + edgeNodes[0]] += std::min(0.0, fct_adf_h[item]);
            fct_plus[(node__z * TODO_INT) + edgeNodes[1]] += std::max(0.0, -fct_adf_h[item]);
            fct_minus[(node__z * TODO_INT) + edgeNodes[1]] += std::min(0.0, -fct_adf_h[item]);
        }
    }
    for ( unsigned int node = 0; node < myDim_nod2D; node++ )
    {
        for ( unsigned int node_z = 0; node_z < nLevels_nod2D[node] - 1; node_z++ )
        {
            unsigned int item = (node_z * TODO_INT) + node;

            flux = fct_plus[item] * (dt / area[item]) + flux_eps;
            fct_plus[item] = std::min(1.0, fct_ttf_max[item] / flux);
            flux = fct_minus[item] * (dt / area[item]) - flux_eps;
            fct_minus[item] = std::min(1.0, fct_ttf_min[item] / flux);
        }
    }
    // TODO: exchange between nodes: fct_plus, fct_minus
    for ( unsigned int node = 0; node <  myDim_nod2D; node++ )
    {
        unsigned int item = node;
        real_type ae = 1.0;

        flux = fct_adf_v[item];
        if ( flux >= 0.0 )
        {
            ae = std::min(ae, fct_plus[item]);
        }
        else
        {
            ae = std::min(ae, fct_minus[item]);
        }
        fct_adf_v[item] *= ae;
        for ( unsigned int node_z = 1; node_z < nLevels_nod2D[node] - 1; node_z++ )
        {
            item = (node_z * TODO_INT) + node;
            ae = 1.0;
            flux = fct_adf_v[item];
            if ( flux >= 0.0 )
            {
                ae = std::min(ae, fct_minus[((node_z - 1) * TODO_INT) + node]);
                ae = std::min(ae, fct_plus[item]);
            }
            else
            {
                ae = std::min(ae, fct_plus[((node_z - 1) * TODO_INT) + node]);
                ae = std::min(ae, fct_minus[item]);
            }
            if ( iter_yn )
            {
                fct_adf_v2[item] = (1.0 - ae) * fct_adf_v[item];
            }
            fct_adf_v[item] *= ae;
        }
    }
    // TODO: exchange between nodes: fct_plus, fct_minus
    for ( unsigned int edge = 0; edge < myDim_edge2D; edge++ )
    {
        unsigned int node_l1 =0, node_l2 = 0;
        int edgeNodes [2] = {edges[edge], edges[2 * myDim_edge2D]};

        node_l1 = nLevels[edge_tri[edge]] - 1;
        if ( edge_tri[2 * edge] > 0 )
        {
            node_l2 = nLevels[edge_tri[2 * edge]] - 1;
        }
        for ( unsigned int node_z = 0; node_z < std::max(node_l1, node_l2); node_z++ )
        {
            unsigned int item = (node_z * TODO_INT) + edge;
            real_type ae = 1.0;

            flux = fct_adf_h[item];
            if ( flux >= 0.0 )
            {
                ae = std::min(ae, fct_plus[(node_z * TODO_INT) + edgeNodes[0]]);
                ae = std::min(ae, fct_minus[(node_z * TODO_INT) + edgeNodes[1]]);
            }
            else
            {
                ae = std::min(ae, fct_minus[(node_z * TODO_INT) + edgeNodes[0]]);
                ae = std::min(ae, fct_plus[(node_z * TODO_INT) + edgeNodes[1]]);
            }
            if ( iter_yn )
            {
                fct_adf_h2[item] = (1.0 - ae) * fct_adf_h[item];
            }
            fct_adf_h[item] *= ae;
        }
    }
    if ( iter_yn )
    {
        for ( unsigned int node = 0; node <  myDim_nod2D; node++ )
        {
            for ( unsigned int node_z = 0; node_z < nLevels_nod2D[node] - 1; node_z++ )
            {
                unsigned int item = (node_z * TODO_INT) + node;

                fct_LO[item] += (fct_adf_v[item] - fct_adf_v[((node_z + 1) * TODO_INT) + node]) * (dt / area[item] / hnode_new[item]);
            }
        }
        for ( unsigned int edge = 0; edge < myDim_edge2D; edge++ )
        {
            unsigned int node_l1 =0, node_l2 = 0;
            int edgeNodes [2] = {edges[edge], edges[2 * myDim_edge2D]};

            node_l1 = nLevels[edge_tri[edge]] - 1;
            if ( edge_tri[2 * edge] > 0 )
            {
                node_l2 = nLevels[edge_tri[2 * edge]] - 1;
            }
            for ( unsigned int node_z = 0; node_z < std::max(node_l1, node_l2); node_z++ )
            {
                unsigned int item = (node_z * TODO_INT) + edgeNodes[0];
                
                fct_LO[item] += fct_adf_h[(node_z * TODO_INT) + edge] * dt / area[item] / hnode_new[item];

                item = (node_z * TODO_INT) + edgeNodes[1];
                fct_LO[item] -= fct_adf_h[(node_z * TODO_INT) + edge] * dt / area[item] / hnode_new[item];
            }
        }
        // TODO: need to properly swap pointers
        fct_adf_h = fct_adf_h2;
        fct_adf_v = fct_adf_v2;
    }
    else
    {
        for ( unsigned int node = 0; node < myDim_nod2D; node++ )
        {
            for ( unsigned int node_z = 0; node_z < nLevels_nod2D[node] - 1; node_z++ )
            {
                unsigned int item = (node_z * TODO_INT) + node;

                del_ttf_advvert[item] -= (ttf[item] * hnode[item]) + (fct_LO[item] * hnode_new[item]) + ((fct_adf_v[item] - fct_adf_v[((node_z + 1) * TODO_INT) + node]) * (dt / area[item]));
            }
        }
        for ( unsigned int edge = 0; edge < myDim_edge2D; edge++ )
        {
            unsigned int node_l1 =0, node_l2 = 0;
            int edgeNodes [2] = {edges[edge], edges[2 * myDim_edge2D]};

            node_l1 = nLevels[edge_tri[edge]] - 1;
            if ( edge_tri[2 * edge] > 0 )
            {
                node_l2 = nLevels[edge_tri[2 * edge]] - 1;
            }
            for ( unsigned int node_z = 0; node_z < std::max(node_l1, node_l2); node_z++ )
            {
                unsigned int item = (node_z * TODO_INT) + edgeNodes[0];

                del_ttf_advhoriz[item] += fct_adf_h[(node_z * TODO_INT) + edge] * (dt / area[item]);
                item = (node_z * TODO_INT) + edgeNodes[1];
                del_ttf_advhoriz[item] -= fct_adf_h[(node_z * TODO_INT) + edge] * (dt / area[item]);
            }
        }
    }
}

void fct_ale_a1_reference_(int * nNodes, int * nLevels_nod2D, real_type * fct_ttf_max, real_type * fct_ttf_min,  real_type * fct_low_order, real_type * ttf)
{
    for ( unsigned int node2D = 0; node2D < *nNodes; node2D++ )
    {
        for ( unsigned int node2D_z = 0; node2D_z < nLevels_nod2D[node2D] - 1; node2D_z++ )
        {
            unsigned int item = (node2D_z * (*nNodes)) + node2D;

            fct_ttf_max[item] = std::max(fct_low_order[item], ttf[item]);
            fct_ttf_min[item] = std::min(fct_low_order[item], ttf[item]);
        }
    }
}

void fct_ale_a2_reference_(int * nElements_ptr, int * nNodes_ptr, int * maxLevels_ptr, int * nLevels, real_type * UV_rhs, int * elem2D_nodes, real_type * fct_ttf_max, real_type * fct_ttf_min)
{
    int nElements = *nElements_ptr;
    int nNodes = *nNodes_ptr;
    int maxLevels = *maxLevels_ptr;
    for ( unsigned int element = 0; element < nElements; element++ )
    {
        for ( unsigned int element_z = 0; element_z < nLevels[element] - 1; element_z++ )
        {
            unsigned int item = (element_z * nNodes);

            UV_rhs[(element_z * nElements) + element] = std::max(fct_ttf_max[item], fct_ttf_max[item + 1]);
            UV_rhs[(element_z * nElements) + element] = std::max(UV_rhs[(element_z * nElements) + element], fct_ttf_max[item + 2]);
            UV_rhs[(maxLevels * nElements) + (element_z * nElements) + element] = std::min(fct_ttf_min[item], fct_ttf_min[item + 1]);
            UV_rhs[(maxLevels * nElements) + (element_z * nElements) + element] = std::min(UV_rhs[(maxLevels * nElements) + (element_z * nElements) + element], fct_ttf_min[item + 2]);
        }
        if ( nLevels[element] <= maxLevels - 2 )
        {
            for ( unsigned int element_z = nLevels[element] - 1; element_z < maxLevels - 1; element_z++ )
            {
                UV_rhs[(element_z * nElements) + element] = std::numeric_limits<real_type>::min();
                UV_rhs[(maxLevels * nElements) + (element_z * nElements) + element] = std::numeric_limits<real_type>::max();
            }
        }
    }
}

void stress2rhs(int myDim_nod2D, int myDim_elem2D, int elem2D_nodes_size, real_type * U_rhs_ice, real_type * V_rhs_ice, real_type * ice_strength, int * elem2D_nodes, real_type * elem_area, real_type * sigma11, real_type * sigma12, real_type * sigma22, real_type * gradient_sca, real_type * metric_factor, real_type * inv_areamass, real_type * rhs_a, real_type * rhs_m)
{
    const unsigned int elementNodes = 3;
    const unsigned int elementGradients = 6;
    const real_type one_third = 1.0 / 3.0;

    // Initialize to zero
    for ( unsigned int node = 0; node < myDim_nod2D; node++ )
    {
        U_rhs_ice[node] = 0.0;
        V_rhs_ice[node] = 0.0;
    }
    for ( unsigned int element = 0; element < myDim_elem2D; element++ )
    {
        if ( ice_strength[element] > 0.0 )
        {
            for ( unsigned int nodeID = 0; nodeID < elementNodes; nodeID++ )
            {
                unsigned int node = elem2D_nodes[(nodeID * elem2D_nodes_size) + element];

                U_rhs_ice[node] -= elem_area[element] * ((sigma11[element] * gradient_sca[(nodeID * elementGradients) + element]) + (sigma12[element] * gradient_sca[((nodeID + 3) * elementGradients) + element]) + (sigma12[element] * one_third * metric_factor[element]));
                V_rhs_ice[node] -= elem_area[element] * ((sigma12[element] * gradient_sca[(nodeID * elementGradients) + element]) + (sigma22[element] * gradient_sca[((nodeID + 3) * elementGradients) + element]) - (sigma11[element] * one_third * metric_factor[element]));
            }
        }
    }
    // Update solution
    for ( unsigned int node = 0; node < myDim_nod2D; node++ )
    {
        if ( inv_areamass[node] > 0.0 )
        {
            U_rhs_ice[node] = (U_rhs_ice[node] * inv_areamass[node]) + rhs_a[node];
            V_rhs_ice[node] = (V_rhs_ice[node] * inv_areamass[node]) + rhs_m[node]; 
        }
        else
        {
            U_rhs_ice[node] = 0.0;
            V_rhs_ice[node] = 0.0;
        }
        
    }
}
