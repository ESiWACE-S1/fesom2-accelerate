
#include <fesom2-accelerate.h>

const int TODO_INT = 0;
const real_type TODO_REAL = 0.0;

/**
 3D Flux Corrected Transport scheme
*/
void fct_ale(unsigned int myDim_nod2D, unsigned int eDim_nod2D, int * nLevels_nod2D, real_type * fct_ttf_max, real_type * fct_ttf_min, real_type * fct_LO, real_type * ttf, unsigned int myDim_elem2D, int * nLevels, real_type * UV_rhs, int nl, int vLimit, real_type * tvert_max, real_type * tvert_min, real_type * fct_plus, real_type * fct_minus, real_type * fct_adf_v, int myDim_edge2D, int * edges, int * edge_tri, real_type * fct_adf_h, real_type dt, real_type * area, bool iter_yn, real_type * fct_adf_v2, real_type * fct_adf_h2, real_type * hnode_new, real_type * del_ttf_advvert, real_type * del_ttf_advhoriz, real_type * hnode)
{
    real_type flux;
    const real_type flux_eps = 1e-16;
    
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
    /*
    ! Vertical3: Vertical bounds are taken into account only if they are narrower than the
    !            horizontal ones  
    if(vlimit==3) then
        do n=1, myDim_nod2D
            do nz=1,nlevels_nod2D(n)-1
                tvert_max(nz)= maxval(UV_rhs(1,nz,nod_in_elem2D(1:nod_in_elem2D_num(n),n)))
                tvert_min(nz)= minval(UV_rhs(2,nz,nod_in_elem2D(1:nod_in_elem2D_num(n),n)))
            end do
            do nz=2, nlevels_nod2D(n)-2
                tvert_max(nz)=min(tvert_max(nz),maxval(fct_ttf_max(nz-1:nz+1,n)))
                tvert_min(nz)=max(tvert_min(nz),minval(fct_ttf_max(nz-1:nz+1,n)))
            end do
            do nz=1,nlevels_nod2D(n)-1
                fct_ttf_max(nz,n)=tvert_max(nz)-fct_LO(nz,n)
                fct_ttf_min(nz,n)=tvert_min(nz)-fct_LO(nz,n)  
            end do
        end do
    end if
    */
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
    /*
    ! b1. Split positive and negative antidiffusive contributions
    ! --> sum all positive (fct_plus), negative (fct_minus) antidiffusive 
    !     horizontal element and vertical node contribution to node n and layer nz
    !     see. R. Löhner et al. "finite element flux corrected transport (FEM-FCT)
    !     for the euler and navier stoke equation
    do n=1, myDim_nod2D
        do nz=1,nlevels_nod2D(n)-1
            fct_plus(nz,n)=0._WP
            fct_minus(nz,n)=0._WP
        end do
    end do
    
    !Vertical
    do n=1, myDim_nod2D
        do nz=1,nlevels_nod2D(n)-1
            fct_plus(nz,n) =fct_plus(nz,n) +(max(0.0_WP,fct_adf_v(nz,n))+max(0.0_WP,-fct_adf_v(nz+1,n)))
            fct_minus(nz,n)=fct_minus(nz,n)+(min(0.0_WP,fct_adf_v(nz,n))+min(0.0_WP,-fct_adf_v(nz+1,n)))
        end do
    end do
    
    !Horizontal
    do edge=1, myDim_edge2D
        enodes(1:2)=edges(:,edge)   
        el=edge_tri(:,edge)
        nl1=nlevels(el(1))-1
        nl2=0
        if(el(2)>0) then
            nl2=nlevels(el(2))-1
        end if   
        do nz=1, max(nl1,nl2)
            fct_plus (nz,enodes(1))=fct_plus (nz,enodes(1)) + max(0.0_WP, fct_adf_h(nz,edge))
            fct_minus(nz,enodes(1))=fct_minus(nz,enodes(1)) + min(0.0_WP, fct_adf_h(nz,edge))  
            fct_plus (nz,enodes(2))=fct_plus (nz,enodes(2)) + max(0.0_WP,-fct_adf_h(nz,edge))
            fct_minus(nz,enodes(2))=fct_minus(nz,enodes(2)) + min(0.0_WP,-fct_adf_h(nz,edge)) 
        end do
    end do
    */
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
    /*
    ! b2. Limiting factors
    do n=1,myDim_nod2D
        do nz=1,nlevels_nod2D(n)-1
            flux=fct_plus(nz,n)*dt/area(nz,n)+flux_eps
            fct_plus(nz,n)=min(1.0_WP,fct_ttf_max(nz,n)/flux)
            flux=fct_minus(nz,n)*dt/area(nz,n)-flux_eps
            fct_minus(nz,n)=min(1.0_WP,fct_ttf_min(nz,n)/flux)
        end do
    end do 
    
    ! fct_minus and fct_plus must be known to neighbouring PE
    call exchange_nod(fct_plus, fct_minus)
    */
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
    /*
    ! b3. Limiting   
        !Vertical
        do n=1, myDim_nod2D
            nz=1
            ae=1.0_WP
            flux=fct_adf_v(nz,n)
            if(flux>=0.0_WP) then 
                ae=min(ae,fct_plus(nz,n))
            else
                ae=min(ae,fct_minus(nz,n))
            end if
            fct_adf_v(nz,n)=ae*fct_adf_v(nz,n) 
            
            do nz=2,nlevels_nod2D(n)-1
                ae=1.0_WP
                flux=fct_adf_v(nz,n)
                if(flux>=0._WP) then 
                    ae=min(ae,fct_minus(nz-1,n))
                    ae=min(ae,fct_plus(nz,n))
                else
                    ae=min(ae,fct_plus(nz-1,n))
                    ae=min(ae,fct_minus(nz,n))
                end if
                
                if (iter_yn) then
                    fct_adf_v2(nz,n)=(1.0_WP-ae)*fct_adf_v(nz,n)
                end if
                fct_adf_v(nz,n)=ae*fct_adf_v(nz,n)
            end do
        ! the bottom flux is always zero 
        end do

        call exchange_nod_end  ! fct_plus, fct_minus
        
        !Horizontal
        do edge=1, myDim_edge2D
            enodes(1:2)=edges(:,edge)
            el=edge_tri(:,edge)
            nl1=nlevels(el(1))-1
            nl2=0
            if(el(2)>0) then
                nl2=nlevels(el(2))-1
            end if  
            do nz=1, max(nl1,nl2)
                ae=1.0_WP
                flux=fct_adf_h(nz,edge)
                
                if(flux>=0._WP) then
                    ae=min(ae,fct_plus(nz,enodes(1)))
                    ae=min(ae,fct_minus(nz,enodes(2)))
                else
                    ae=min(ae,fct_minus(nz,enodes(1)))
                    ae=min(ae,fct_plus(nz,enodes(2)))
                endif
                
                if (iter_yn) then
                    fct_adf_h2(nz,edge)=(1.0_WP-ae)*fct_adf_h(nz,edge)
                end if
                fct_adf_h(nz,edge)=ae*fct_adf_h(nz,edge)
            end do
        end do
    */
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
    /*
    if (iter_yn) then
        !___________________________________________________________________________
        ! c. Update the LO
        ! Vertical
        do n=1, myDim_nod2d
            do nz=1,nlevels_nod2D(n)-1  
                fct_LO(nz,n)=fct_LO(nz,n)+(fct_adf_v(nz,n)-fct_adf_v(nz+1,n))*dt/area(nz,n)/hnode_new(nz,n)
            end do
        end do
        
        ! Horizontal
        do edge=1, myDim_edge2D
            enodes(1:2)=edges(:,edge)
            el=edge_tri(:,edge)
            nl1=nlevels(el(1))-1
            nl2=0
            if (el(2)>0) nl2=nlevels(el(2))-1
            do nz=1, max(nl1,nl2)
                fct_LO(nz,enodes(1))=fct_LO(nz,enodes(1))+fct_adf_h(nz,edge)*dt/area(nz,enodes(1))/hnode_new(nz,enodes(1))
                fct_LO(nz,enodes(2))=fct_LO(nz,enodes(2))-fct_adf_h(nz,edge)*dt/area(nz,enodes(2))/hnode_new(nz,enodes(2))
            end do
        end do
        fct_adf_h=fct_adf_h2
        fct_adf_v=fct_adf_v2
        return !do the next iteration with fct_ale
    end if

    ! c. Update the solution
    ! Vertical
    do n=1, myDim_nod2d
        do nz=1,nlevels_nod2D(n)-1  
            del_ttf_advvert(nz,n)=del_ttf_advvert(nz,n)-ttf(nz,n)*hnode(nz,n)+fct_LO(nz,n)*hnode_new(nz,n) + &
                                    (fct_adf_v(nz,n)-fct_adf_v(nz+1,n))*dt/area(nz,n)
        end do
    end do
    
    ! Horizontal
    do edge=1, myDim_edge2D
        enodes(1:2)=edges(:,edge)
        el=edge_tri(:,edge)
        nl1=nlevels(el(1))-1
        nl2=0
        if(el(2)>0) nl2=nlevels(el(2))-1
        do nz=1, max(nl1,nl2)
            del_ttf_advhoriz(nz,enodes(1))=del_ttf_advhoriz(nz,enodes(1))+fct_adf_h(nz,edge)*dt/area(nz,enodes(1))
            del_ttf_advhoriz(nz,enodes(2))=del_ttf_advhoriz(nz,enodes(2))-fct_adf_h(nz,edge)*dt/area(nz,enodes(2))
        end do
    end do
    */
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

void stress2rhs(unsigned int myDim_nod2D, unsigned int myDim_elem2D, unsigned int elem2D_nodes_size, real_type * U_rhs_ice, real_type * V_rhs_ice, real_type * ice_strength, unsigned int * elem2D_nodes, real_type * elem_area, real_type * sigma11, real_type * sigma12, real_type * sigma22, real_type * gradient_sca, real_type * metric_factor, real_type * inv_areamass, real_type * rhs_a, real_type * rhs_m)
{
    const unsigned int elementNodes = 3;
    const unsigned int elementGradients = 6;
    const real_type one_third = 1.0 / 3.0;

    // Initialization
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