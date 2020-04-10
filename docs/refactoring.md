
# FESOM2 Refactoring

The goal of this document is to provide in-depth analysis of selected parts of our fork of [FESOM2](https://github.com/ESiWACE-S1/fesom2) in order to port them to C++ and CUDA.

Still work in progress, updated while reading the code.

## Subroutine fct_ale

### Source code

```fortran
subroutine fct_ale(ttf, iter_yn, mesh)
    !
    ! 3D Flux Corrected Transport scheme
    ! Limits antidiffusive fluxes==the difference in flux HO-LO
    ! LO ==Low-order  (first-order upwind)
    ! HO ==High-order (3rd/4th order gradient reconstruction method)
    ! Adds limited fluxes to the LO solution   
    use MOD_MESH
    use O_MESH
    use o_ARRAYS
    use o_PARAM
    use g_PARSUP
    use g_CONFIG
    use g_comm_auto
    implicit none
    type(t_mesh), intent(in)  , target :: mesh
    integer                   :: n, nz, k, elem, enodes(3), num, el(2), nl1, nl2, edge
    real(kind=WP)             :: flux, ae,tvert_max(mesh%nl-1),tvert_min(mesh%nl-1) 
    real(kind=WP), intent(in) :: ttf(mesh%nl-1, myDim_nod2D+eDim_nod2D)
    real(kind=WP)             :: flux_eps=1e-16
    real(kind=WP)             :: bignumber=1e3
    integer                   :: vlimit=1
    logical, intent(in)       :: iter_yn !more iterations to be made with fct_ale?

#include "associate_mesh.h"

    ! --------------------------------------------------------------------------
    ! ttf is the tracer field on step n
    ! del_ttf is the increment 
    ! vlimit sets the version of limiting, see below
    ! --------------------------------------------------------------------------
    
    !___________________________________________________________________________
    ! a1. max, min between old solution and updated low-order solution per node
    do n=1,myDim_nod2D + edim_nod2d
        do nz=1, nlevels_nod2D(n)-1 
            fct_ttf_max(nz,n)=max(fct_LO(nz,n), ttf(nz,n))
            fct_ttf_min(nz,n)=min(fct_LO(nz,n), ttf(nz,n))
        end do
    end do       
    
    !___________________________________________________________________________
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
    end do ! --> do elem=1, myDim_elem2D
    
    !___________________________________________________________________________
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
    
    !___________________________________________________________________________
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
    
    !___________________________________________________________________________
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
    
    !___________________________________________________________________________
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
!             fct_plus(nz,n)=fct_plus(nz,n)+ &
!                             (max(0.0_WP,fct_adf_v(nz,n))+max(0.0_WP,-fct_adf_v(nz+1,n))) &
!                             /hnode(nz,n)
!             fct_minus(nz,n)=fct_minus(nz,n)+ &
!                             (min(0.0_WP,fct_adf_v(nz,n))+min(0.0_WP,-fct_adf_v(nz+1,n))) &
!                             /hnode(nz,n)
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
    
    !___________________________________________________________________________
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
    
    !___________________________________________________________________________
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
    
    !___________________________________________________________________________
    ! c. Update the solution
    ! Vertical
    do n=1, myDim_nod2d
        do nz=1,nlevels_nod2D(n)-1  
            del_ttf_advvert(nz,n)=del_ttf_advvert(nz,n)-ttf(nz,n)*hnode(nz,n)+fct_LO(nz,n)*hnode_new(nz,n) + &
                                    (fct_adf_v(nz,n)-fct_adf_v(nz+1,n))*dt/area(nz,n)
!!PS             del_ttf(nz,n)        =del_ttf(nz,n)        -ttf(nz,n)*hnode(nz,n)+fct_LO(nz,n)*hnode_new(nz,n) + &
!!PS                                     (fct_adf_v(nz,n)-fct_adf_v(nz+1,n))*dt/area(nz,n)
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
!!PS             del_ttf(nz,enodes(1))         =del_ttf(nz,enodes(1))         +fct_adf_h(nz,edge)*dt/area(nz,enodes(1))
!!PS             del_ttf(nz,enodes(2))         =del_ttf(nz,enodes(2))         -fct_adf_h(nz,edge)*dt/area(nz,enodes(2))
        end do
    end do
end subroutine fct_ale
```

### Data

All variables defined outside the scope of the subroutine need to be identified.

- myDim_nod2D : *integer*
    - Used in multiple loops, it represents the beginning of some partition.
- eDim_nod2D : *integer*
    - Used in multiple loops, it represent the size of some partition.
- nlevels_nod2D : *integer array*
    - Array containing the size of some local dimension.
- fct_ttf_max : *real array*
    - Two dimensional array, containing the maximum of some previous step.
- fct_ttf_min : *real array*
    - Two dimensional array, containing the minimum of some previous step.
- fct_LO : *real array*
    - Two dimensional array, containing the low order solution of fct.
- ttf : *real array*
    - Two dimensional array, containing the tracer field at step n.
- myDim_elem2D : *integer*
    - Used in multiple loops, it represent the index of some element.
- elem2D_nodes : *integer array*
    - Two dimensional array, not sure about the content yet.
- nlevels : *integer array*
    - Array containing the size of some dimension.
- UV_rhs : *real array*
    - Three dimensional array. The first dimension seem to be limited to the set [1, 2].
- nl : *integer pointer*
    - Points to a single integer value in T_MESH, probably the number of vertical layers.
- vlimit : *integer*
    - Switch to control some different limiting strategies.
- tvert_max : *real array*
    - Array containing the maximum temperature of vertical layers.
- tvert_min : *real array*
    - Array containing the minimum temperature of vertical layers.
- fct_plus : *real array*
    - Two dimensional array, containing the result of some integration.
- fct_minus : *real array*
    - Two dimensional array, containing the result of some integration.
- fct_adf_v : *real array*
    - Two dimensional array, containing some antidif. vertical flux.
- fct_adf_h : *real array*
    - Two dimensional array, containing some antidif. horizontal flux.
- myDim_edge2D : *integer*
    - Represents the number of edges.
- edges : *integer array*
    - Two dimensional array, containing the two vertices of an edge.
- edge_tri : *integer array*
    - Two dimensional array, containing the two vertices of an edge.
