
#include <fesom2-accelerate.h>

const int TODO_INT = 0;
const real_type TODO_REAL = 0.0;

/**
 3D Flux Corrected Transport scheme
*/
void fct_ale(unsigned int myDim_nod2D, unsigned int eDim_nod2D, int * nLevels_nod2D, real_type * fct_ttf_max, real_type * fct_ttf_min, real_type * fct_LO, real_type * ttf, unsigned int myDim_elem2D, int * nLevels, real_type * UV_rhs, int nl, int vLimit, real_type * tvert_max, real_type * tvert_min)
{
    /*
    ! a1. max, min between old solution and updated low-order solution per node
    do n=1,myDim_nod2D + edim_nod2d
        do nz=1, nlevels_nod2D(n)-1 
            fct_ttf_max(nz,n)=max(fct_LO(nz,n), ttf(nz,n))
            fct_ttf_min(nz,n)=min(fct_LO(nz,n), ttf(nz,n))
        end do
    end do   
    */
    for ( unsigned int node2D = 0; node2D < myDim_nod2D + eDim_nod2D; node2D++ )
    {
        for ( unsigned int node2D_z = 0; node2D_z < nLevels_nod2D[node2D] - 1; node2D_z++ )
        {
            unsigned int item = (node2D_z * TODO_INT) + node2D;
            fct_ttf_max[item] = std::max(fct_LO[item], ttf[item]);
            fct_ttf_min[item] = std::min(fct_LO[item], ttf[item]);
        }
    }
    /*
    ! a2. Admissible increments on elements
    !     (only layers below the first and above the last layer)
    !     look for max, min bounds for each element --> UV_rhs here auxilary array
    do elem=1, myDim_elem2D
        enodes=elem2D_nodes(:,elem)
        do nz=1, nlevels(elem)-1
            UV_rhs(1,nz,elem)=maxval(fct_ttf_max(nz,enodes))
            UV_rhs(2,nz,elem)=minval(fct_ttf_min(nz,enodes))
        end do
        if (nlevels(elem)<=nl-1) then
            do nz=nlevels(elem),nl-1
                UV_rhs(1,nz,elem)=-bignumber
                UV_rhs(2,nz,elem)= bignumber
            end do
        endif
    end do
    */
    for ( unsigned int element = 0; element < myDim_elem2D; element++ )
    {
        for ( unsigned int element_z = 0; element_z < nLevels[element] - 1; element_z++ )
        {
            unsigned int item = (element_z * TODO_INT) + element;
            UV_rhs[item] = TODO_REAL;
            UV_rhs[(TODO_INT * TODO_INT) + item] = TODO_REAL;
        }
        if ( nLevels[element] <= nl - 1 )
        {
            for ( unsigned int element_z = nLevels[element] - 1; element_z < nl - 1; element_z++ )
            {
                unsigned int item = (element_z * TODO_INT) + element;
                    UV_rhs[item] = std::numeric_limits<real_type>::min();
                    UV_rhs[(TODO_INT * TODO_INT) + item] = std::numeric_limits<real_type>::max();
            }
        }
    }
    /*
        ! a3. Bounds on clusters and admissible increments
        ! Vertical1: In this version we look at the bounds on the clusters
        !            above and below, which leaves wide bounds because typically 
        !            vertical gradients are larger.  
        if(vlimit==1) then
            !Horizontal
            do n=1, myDim_nod2D
                !___________________________________________________________________
                do nz=1,nlevels_nod2D(n)-1
                    ! max,min horizontal bound in cluster around node n in every 
                    ! vertical layer
                    ! nod_in_elem2D     --> elem indices of which node n is surrounded
                    ! nod_in_elem2D_num --> max number of surrounded elem 
                    tvert_max(nz)= maxval(UV_rhs(1,nz,nod_in_elem2D(1:nod_in_elem2D_num(n),n)))
                    tvert_min(nz)= minval(UV_rhs(2,nz,nod_in_elem2D(1:nod_in_elem2D_num(n),n)))
                end do
                
                !___________________________________________________________________
                ! calc max,min increment of surface layer with respect to low order 
                ! solution 
                fct_ttf_max(1,n)=tvert_max(1)-fct_LO(1,n)
                fct_ttf_min(1,n)=tvert_min(1)-fct_LO(1,n)
                
                ! calc max,min increment from nz-1:nz+1 with respect to low order 
                ! solution at layer nz
                do nz=2,nlevels_nod2D(n)-2  
                    fct_ttf_max(nz,n)=maxval(tvert_max(nz-1:nz+1))-fct_LO(nz,n)
                    fct_ttf_min(nz,n)=minval(tvert_min(nz-1:nz+1))-fct_LO(nz,n)
                end do
                ! calc max,min increment of bottom layer -1 with respect to low order 
                ! solution 
                nz=nlevels_nod2D(n)-1
                fct_ttf_max(nz,n)=tvert_max(nz)-fct_LO(nz,n)
                fct_ttf_min(nz,n)=tvert_min(nz)-fct_LO(nz,n)  
            end do
        end if
    */
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
    /*
    ! Vertical2: Similar to the version above, but the vertical bounds are more 
        ! local  
        if(vlimit==2) then
            do n=1, myDim_nod2D
                do nz=1,nlevels_nod2D(n)-1
                    tvert_max(nz)= maxval(UV_rhs(1,nz,nod_in_elem2D(1:nod_in_elem2D_num(n),n)))
                    tvert_min(nz)= minval(UV_rhs(2,nz,nod_in_elem2D(1:nod_in_elem2D_num(n),n)))
                end do
                do nz=2, nlevels_nod2D(n)-2
                    tvert_max(nz)=max(tvert_max(nz),maxval(fct_ttf_max(nz-1:nz+1,n)))
                    tvert_min(nz)=min(tvert_min(nz),minval(fct_ttf_max(nz-1:nz+1,n)))
                end do
                do nz=1,nlevels_nod2D(n)-1
                    fct_ttf_max(nz,n)=tvert_max(nz)-fct_LO(nz,n)
                    fct_ttf_min(nz,n)=tvert_min(nz)-fct_LO(nz,n)  
                end do
            end do
        end if
    */
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
                tvert_max[node_z] = std::max(TODO_REAL);
                tvert_min[node_z] = std::min(TODO_REAL);
            }
            for( unsigned int node_z = 0; node_z < nLevels_nod2D[node] - 1; node_z++ )
            {
                unsigned int item = (node_z * TODO_INT) + node;
                fct_ttf_max[item] = tvert_max[node_z] - fct_LO[item];
                fct_ttf_min[item] = tvert_min[node_z] - fct_LO[item];
            }
        }
    }
}
