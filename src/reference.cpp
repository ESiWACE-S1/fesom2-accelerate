
#include <fesom2-accelerate.h>

const int TODO_INT = 0;

/**
 3D Flux Corrected Transport scheme
*/
void fct_ale(unsigned int myDim_nod2D, unsigned int eDim_nod2D, int * nLevels_nod2D, real_type * fct_ttf_max, real_type * fct_ttf_min, real_type * fct_LO, real_type * ttf, unsigned int myDim_elem2D, int * nLevels, real_type * UV_rhs, int nl)
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
           UV_rhs[item] = std::max();
           UV_rhs[(TODO_INT * TODO_INT) + item] = std::min();
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
}
