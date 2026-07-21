

#include "macrodef.fh"

! Update outer boundaries with Sommerfeld boundary condition
!
!-----------------------------------------------------------------------------
!5th order interpolation
  subroutine sommerfeld_rout(ex,X, Y, Z,xmin,ymin,zmin,xmax,ymax,zmax,dT,chi0,&
               Lap0,f0,f,SoA,Symmetry,precor)

  implicit none
 
!~~~~~~> Input parameters:
  integer, intent(in):: ex(1:3),Symmetry,precor
  real*8, dimension(ex(1)) :: X
  real*8, dimension(ex(2)) :: Y
  real*8, dimension(ex(3)) :: Z
  real*8,  intent(in):: xmin,ymin,zmin,xmax,ymax,zmax,dT
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in)::chi0,Lap0,f0
  real*8, dimension(ex(1),ex(2),ex(3)),intent(inout)::f
  real*8, dimension(3),intent(in) ::SoA
!~~~~~~> Other variables:
  real*8 :: dX,dY,dZ,r,fac
  integer :: i, j, k,m
  logical :: gont,nouse
  integer,dimension(3) :: cxB,cxT
  integer :: layer(1:6,1:6),gp
! index of layer, first one: i,j,k; second one: front back etc. boundary
  integer,parameter::ordn = 6, CORRECTSTEP=1
  real*8 :: ddy
  real*8,  dimension(1:ordn) :: xa
  real*8,  dimension(1:3) :: cx
  real*8,  dimension(1:ordn,1:ordn,1:ordn) :: ya
  real*8, parameter :: ZEO = 0.d0, ONE = 1.d0, SYM = 1.d0, ANT = -1.d0
  integer, parameter :: NO_SYMM = 0, EQ_SYMM = 1, OCTANT = 2
!~~~~~~> Interface

  interface

  function decide3d(ex,f,fpi,cxB,cxT,SoA,ya,ORDN,Symmetry)  result(gont)
  implicit none

  integer,                                 intent(in) :: ORDN,Symmetry
  integer,dimension(1:3)                 , intent(in) :: ex,cxB,cxT
  real*8, dimension(1:3)                 , intent(in) :: SoA
  real*8, dimension(ex(1),ex(2),ex(3))   , intent(in) :: f,fpi
  real*8, dimension(cxB(1):cxT(1),cxB(2):cxT(2),cxB(3):cxT(3)), intent(out):: ya
  logical::gont
  end function decide3d

  end interface

  dX = X(2) - X(1)
  dY = Y(2) - Y(1)
  dZ = Z(2) - Z(1)

layer(1:3,:) = 1
layer(4:6,:) =-1

if(dabs(X(ex(1))-xmax) < dX)then
   layer(1,1) = ex(1)
   layer(2,1) = 1
   layer(3,1) = 1
   layer(4,1) = ex(1)
   layer(5,1) = ex(2)
   layer(6,1) = ex(3)
endif

if(dabs(Y(ex(2))-ymax) < dY)then 
  layer(1,2) = 1
  layer(2,2) = ex(2)
  layer(3,2) = 1
  layer(4,2) = ex(1)
  layer(5,2) = ex(2)
  layer(6,2) = ex(3)
endif
 

if(dabs(Z(ex(3))-zmax) < dZ)then
  layer(1,3) = 1
  layer(2,3) = 1
  layer(3,3) = ex(3)
  layer(4,3) = ex(1)
  layer(5,3) = ex(2)
  layer(6,3) = ex(3)
endif
! lower boundary         but    not  symmetry boundary
if(dabs(X(1)-xmin) < dX .and. (.not.(Symmetry==OCTANT.and.dabs(xmin)<dX/2)))then
   layer(1,4) = 1
   layer(2,4) = 1
   layer(3,4) = 1
   layer(4,4) = 1
   layer(5,4) = ex(2)
   layer(6,4) = ex(3)
endif 

if(dabs(Y(1)-ymin) < dY .and. (.not.(Symmetry==OCTANT.and.dabs(ymin)<dY/2)))then
   layer(1,5) = 1
   layer(2,5) = 1
   layer(3,5) = 1
   layer(4,5) = ex(1)
   layer(5,5) = 1
   layer(6,5) = ex(3)
endif

if(dabs(Z(1)-zmin) < dZ .and. (.not.(Symmetry>NO_SYMM.and.dabs(zmin)<dZ/2)))then
   layer(1,6) = 1
   layer(2,6) = 1
   layer(3,6) = 1
   layer(4,6) = ex(1)
   layer(5,6) = ex(2)
   layer(6,6) = 1
endif

! so x,y and z are same: 0,1,2,...,(ORDN-1)
  do i = 1, ordn
     xa(i) = dble( i - 1 )
  enddo

!~~~~~~> boundary calculations start...
  if( precor == CORRECTSTEP ) then

   do gp = 1, 6, 1

    gont = any( layer(:,gp) == - 1 )
 
    if( .not. gont ) then

     do k = layer(3,gp), layer(6,gp), 1
      do j = layer(2,gp), layer(5,gp), 1
       do i = layer(1,gp), layer(4,gp), 1

        f(i,j,k) = f0(i,j,k)

       enddo
      enddo
     enddo
    endif
   enddo

  else

  do gp = 1, 6

   gont = any( layer(:,gp) == - 1 )
 
   if( .not. gont ) then

    do k = layer(3,gp), layer(6,gp)
     do j = layer(2,gp), layer(5,gp)
      do i = layer(1,gp), layer(4,gp)
! tc/sc*dT/r
       r = (Lap0(i,j,k) + ONE)*dsqrt(ONE+chi0(i,j,k))*dT/dsqrt(X(i)**2+Y(j)**2+Z(k)**2)
       fac=ONE-r
       cx(1) = r*X(i)/dX
       cx(2) = r*Y(j)/dY
       cx(3) = r*Z(k)/dZ
       if(cx(1)>ZEO)then
         cxB(1) = i-dint(cx(1))-ordn/2
       else
         cxB(1) = i-dint(cx(1))-ordn/2+1
       endif
       if(cx(2)>ZEO)then
         cxB(2) = j-dint(cx(2))-ordn/2
       else
         cxB(2) = j-dint(cx(2))-ordn/2+1
       endif
       if(cx(3)>ZEO)then
         cxB(3) = k-dint(cx(3))-ordn/2
       else
         cxB(3) = k-dint(cx(3))-ordn/2+1
       endif

       where(cx>ZEO)
         cx = dint(cx)-cx+ordn/2
       elsewhere
         cx = dint(cx)-cx+ordn/2-1
       end where

       cxT = cxB+ordn-1

       if(Symmetry==NO_SYMM.and.cxB(3)<1)then
         cx(3)=cx(3)+(cxB(3)-1)
         cxT(3)=cxT(3)-(cxB(3)-1)
         cxB(3)=1
       endif       
       if(Symmetry<OCTANT.and.cxB(2)<1)then
         cx(2)=cx(2)+(cxB(2)-1)
         cxT(2)=cxT(2)-(cxB(2)-1)
         cxB(2)=1
       endif    
       if(Symmetry<OCTANT.and.cxB(1)<1)then
         cx(1)=cx(1)+(cxB(1)-1)
         cxT(1)=cxT(1)-(cxB(1)-1)
         cxB(1)=1
       endif
       do m=1,3
          if(cxT(m)>ex(m))then
             cx(m)=cx(m)+(cxT(m)-ex(m))
             cxB(m)=cxB(m)-(cxT(m)-ex(m))
             cxT(m)=ex(m)
          endif
       enddo

!~~~~~~> Interpolate
       nouse=decide3d(ex,f0,f0,cxB,cxT,SoA,ya,ordn,Symmetry)
       call polin3(xa,xa,xa,ya,cx(1),cx(2),cx(3),r,ddy,ordn)
       f(i,j,k)=r*fac

      enddo
     enddo
    enddo
   endif
  enddo

  endif

  return

  end subroutine sommerfeld_rout
!sommerfeld condition following BAM code
  subroutine sommerfeld_routbam(ex,X, Y, Z,xmin,ymin,zmin,xmax,ymax,zmax,f_rhs,&
                                f0,velocity,SoA,Symmetry)

  implicit none
 
!~~~~~~> Input parameters:
  integer, intent(in):: ex(1:3),Symmetry
  real*8, intent(in) :: velocity
  real*8, dimension(ex(1)) :: X
  real*8, dimension(ex(2)) :: Y
  real*8, dimension(ex(3)) :: Z
  real*8,  intent(in):: xmin,ymin,zmin,xmax,ymax,zmax
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in)::f0
  real*8, dimension(ex(1),ex(2),ex(3)),intent(inout)::f_rhs
  real*8,dimension(3),intent(in) ::SoA
!~~~~~~> Other variables:
  real*8,dimension(0:ex(1),0:ex(2),0:ex(3)) :: fh
  logical :: gont
  real*8 :: dX,dY,dZ,R
  integer :: i, j, k
  real*8 :: d2dx,d2dy,d2dz
  integer :: layer(1:6,1:6),gp
! index of layer, first one: i,j,k; second one: front back etc. boundary
  integer :: imin,jmin,kmin,imax,jmax,kmax
  real*8 :: fx,fy,fz 
  real*8, parameter :: ZEO = 0.d0, ONE = 1.d0, TWO=2.d0
  integer, parameter :: NO_SYMM = 0, EQ_SYMM = 1, OCTANT = 2

  real*8 :: wx,wy,wz
  
  dX = X(2) - X(1)
  dY = Y(2) - Y(1)
  dZ = Z(2) - Z(1)

  d2dx = ONE/TWO/dX
  d2dy = ONE/TWO/dY
  d2dz = ONE/TWO/dZ

  imax = ex(1)
  jmax = ex(2)
  kmax = ex(3)

  imin = 1
  jmin = 1
  kmin = 1
  if(Symmetry > NO_SYMM .and. dabs(Z(1)) < dZ) kmin = 0
  if(Symmetry > EQ_SYMM .and. dabs(X(1)) < dX) imin = 0
  if(Symmetry > EQ_SYMM .and. dabs(Y(1)) < dY) jmin = 0

  call symmetry_bd(1,ex,f0,fh,SoA)

layer(1:3,:) = 1
layer(4:6,:) =-1

if(dabs(X(ex(1))-xmax) < dX)then
   layer(1,1) = ex(1)
   layer(2,1) = 1
   layer(3,1) = 1
   layer(4,1) = ex(1)
   layer(5,1) = ex(2)
   layer(6,1) = ex(3)
endif

if(dabs(Y(ex(2))-ymax) < dY)then 
  layer(1,2) = 1
  layer(2,2) = ex(2)
  layer(3,2) = 1
  layer(4,2) = ex(1)
  layer(5,2) = ex(2)
  layer(6,2) = ex(3)
endif
 
if(dabs(Z(ex(3))-zmax) < dZ)then
  layer(1,3) = 1
  layer(2,3) = 1
  layer(3,3) = ex(3)
  layer(4,3) = ex(1)
  layer(5,3) = ex(2)
  layer(6,3) = ex(3)
endif
! lower boundary         but    not  symmetry boundary
if(dabs(X(1)-xmin) < dX .and. (.not.(Symmetry==OCTANT.and.dabs(xmin)<dX/2)))then
   layer(1,4) = 1
   layer(2,4) = 1
   layer(3,4) = 1
   layer(4,4) = 1
   layer(5,4) = ex(2)
   layer(6,4) = ex(3)
endif 

if(dabs(Y(1)-ymin) < dY .and. (.not.(Symmetry==OCTANT.and.dabs(ymin)<dY/2)))then
   layer(1,5) = 1
   layer(2,5) = 1
   layer(3,5) = 1
   layer(4,5) = ex(1)
   layer(5,5) = 1
   layer(6,5) = ex(3)
endif

if(dabs(Z(1)-zmin) < dZ .and. (.not.(Symmetry>NO_SYMM.and.dabs(zmin)<dZ/2)))then
   layer(1,6) = 1
   layer(2,6) = 1
   layer(3,6) = 1
   layer(4,6) = ex(1)
   layer(5,6) = ex(2)
   layer(6,6) = 1
endif

  do gp = 1, 6

   gont = any( layer(:,gp) == - 1 )
 
   if( .not. gont ) then

    do k = layer(3,gp), layer(6,gp)
     do j = layer(2,gp), layer(5,gp)
      do i = layer(1,gp), layer(4,gp)
#if 0      
!! old code
! x direction   
        if(i+1 <= imax .and. i-1 >= imin)then
      fx=d2dx*(-fh(i-1,j,k)+fh(i+1,j,k))

      elseif(i==imin)then
      fx=(-fh(i,j,k)+fh(i+1,j,k))/dX

      elseif(i==imax)then
      fx=(-fh(i-1,j,k)+fh(i,j,k))/dX
      
      endif
! y direction   
        if(j+1 <= jmax .and. j-1 >= jmin)then
      fy=d2dy*(-fh(i,j-1,k)+fh(i,j+1,k))

      elseif(j==jmin)then
      fy=(-fh(i,j,k)+fh(i,j+1,k))/dY

      elseif(j==jmax)then
      fy=(-fh(i,j-1,k)+fh(i,j,k))/dY
      
      endif
! z direction   
        if(k+1 <= kmax .and. k-1 >= kmin)then
      fz=d2dz*(-fh(i,j,k-1)+fh(i,j,k+1))
      
      elseif(k==kmin)then
      fz=(-fh(i,j,k)+fh(i,j,k+1))/dZ

      elseif(k==kmax)then
      fz=(-fh(i,j,k-1)+fh(i,j,k))/dZ
      
      endif

      R = dsqrt(X(i)**2+Y(j)**2+Z(k)**2)
      f_rhs(i,j,k) = -velocity*(fx*X(i) + fy*Y(j) + fz*Z(k) + f0(i,j,k))/R
#else      
!! new code, 2012dec26, based on bam 
!! we always assume var0 = 0
      R = dsqrt(X(i)**2+Y(j)**2+Z(k)**2)
      wx = velocity*X(i)/R
      wy = velocity*Y(j)/R
      wz = velocity*Z(k)/R
      if(wx > 0)then
         if(i-2>=imin)then
           fx = d2dx*(3*fh(i,j,k)-4*fh(i-1,j,k)+fh(i-2,j,k))
         elseif(i-1>=imin)then
           fx = d2dx*(-fh(i-1,j,k)+fh(i+1,j,k))
         else
           fx = d2dx*(-fh(i+2,j,k)+4*fh(i+1,j,k)-3*fh(i,j,k))
         endif
      elseif(wx < 0)then
         if(i+2<=imax)then
           fx = d2dx*(-fh(i+2,j,k)+4*fh(i+1,j,k)-3*fh(i,j,k))
         elseif(i+1<=imax)then
           fx = d2dx*(-fh(i-1,j,k)+fh(i+1,j,k))
         else
           fx = d2dx*(3*fh(i,j,k)-4*fh(i-1,j,k)+fh(i-2,j,k))
         endif
      endif

      if(wy > 0)then
         if(j-2>=jmin)then
           fy = d2dy*(3*fh(i,j,k)-4*fh(i,j-1,k)+fh(i,j-2,k))
         elseif(j-1>=jmin)then
           fy = d2dy*(-fh(i,j-1,k)+fh(i,j+1,k))
         else
           fy = d2dy*(-fh(i,j+2,k)+4*fh(i,j+1,k)-3*fh(i,j,k))
         endif
      elseif(wy < 0)then
         if(j+2<=jmax)then
           fy = d2dy*(-fh(i,j+2,k)+4*fh(i,j+1,k)-3*fh(i,j,k))
         elseif(j+1<=jmax)then
           fy = d2dy*(-fh(i,j-1,k)+fh(i,j+1,k))
         else
           fy = d2dy*(3*fh(i,j,k)-4*fh(i,j-1,k)+fh(i,j-2,k))
         endif
      endif

      if(wz > 0)then
         if(k-2>=kmin)then
           fz = d2dz*(3*fh(i,j,k)-4*fh(i,j,k-1)+fh(i,j,k-2))
         elseif(k-1>=kmin)then
           fz = d2dz*(-fh(i,j,k-1)+fh(i,j,k+1))
         else
           fz = d2dz*(-fh(i,j,k+2)+4*fh(i,j,k+1)-3*fh(i,j,k))
         endif
      elseif(wz < 0)then
         if(k+2<=kmax)then
           fz = d2dz*(-fh(i,j,k+2)+4*fh(i,j,k+1)-3*fh(i,j,k))
         elseif(k+1<=kmax)then
           fz = d2dz*(-fh(i,j,k-1)+fh(i,j,k+1))
         else
           fz = d2dz*(3*fh(i,j,k)-4*fh(i,j,k-1)+fh(i,j,k-2))
         endif
      endif

      f_rhs(i,j,k) = -velocity*(fx*X(i) + fy*Y(j) + fz*Z(k) + f0(i,j,k))/R
#endif
      enddo
     enddo
    enddo
   endif
  enddo

  return

  end subroutine sommerfeld_routbam
!sommerfeld condition following BAM code for shell
  subroutine sommerfeld_routbam_ss(ex,X, Y, Z,xmin,ymin,zmin,xmax,ymax,zmax,f_rhs,&
                                f0,velocity,SoA,Symmetry)

  implicit none
 
!~~~~~~> Input parameters:
  integer, intent(in):: ex(1:3),Symmetry
  real*8, intent(in) :: velocity
  real*8, dimension(ex(1)) :: X
  real*8, dimension(ex(2)) :: Y
! Z-> R  
  real*8, dimension(ex(3)) :: Z
  real*8,  intent(in):: xmin,ymin,zmin,xmax,ymax,zmax
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in)::f0
  real*8, dimension(ex(1),ex(2),ex(3)),intent(inout)::f_rhs
  real*8,dimension(3),intent(in) ::SoA
!~~~~~~> Other variables:
  logical :: gont
  real*8 :: dZ
  integer :: i, j, k
  real*8 :: d2dz
  integer :: layer(1:6,1:6),gp
! index of layer, first one: i,j,k; second one: front back etc. boundary
  integer :: kmin,kmax
  real*8 :: fz 
  real*8, parameter :: ZEO = 0.d0, ONE = 1.d0, TWO=2.d0
  
  dZ = Z(2) - Z(1)

  d2dz = ONE/TWO/dZ

  kmax = ex(3)

  kmin = 1

layer(1:3,:) = 1
layer(4:6,:) =-1
 
if(dabs(Z(ex(3))-zmax) < dZ)then
  layer(1,3) = 1
  layer(2,3) = 1
  layer(4,3) = ex(1)
  layer(5,3) = ex(2)
#if 1
! do not consider buffer points near boundary
  layer(3,3) = ex(3)
  layer(6,3) = ex(3)
#else  
! consider buffer points near boundary 
  layer(3,3) = ex(3) - ghost_width
  layer(6,3) = ex(3) - ghost_width
#endif
endif

if(dabs(Z(1)-zmin) < dZ)then
   layer(1,6) = 1
   layer(2,6) = 1
   layer(3,6) = 1
   layer(4,6) = ex(1)
   layer(5,6) = ex(2)
   layer(6,6) = 1
endif

! outgoing BD
  gp = 3

  gont = any( layer(:,gp) == - 1 )
 
   if( .not. gont ) then

    do k = layer(3,gp), layer(6,gp)
     do j = layer(2,gp), layer(5,gp)
      do i = layer(1,gp), layer(4,gp)
#if 0      
!! old code
! z direction   
        if(k+1 <= kmax .and. k-1 >= kmin)then
      fz=d2dz*(-f0(i,j,k-1)+f0(i,j,k+1))
      
      elseif(k==kmin)then
      fz=(-f0(i,j,k)+f0(i,j,k+1))/dZ

      elseif(k==kmax)then
      fz=(-f0(i,j,k-1)+f0(i,j,k))/dZ
      
      endif
#else
!! new code, 2012dec16, based on bam
      if(velocity > 0)then
         if(k-2>=kmin)then
           fz = d2dz*(3*f0(i,j,k)-4*f0(i,j,k-1)+f0(i,j,k-2))
         elseif(k-1>=kmin)then
           fz = d2dz*(-f0(i,j,k-1)+f0(i,j,k+1))
         else
           fz = d2dz*(-f0(i,j,k+2)+4*f0(i,j,k+1)-3*f0(i,j,k))
         endif
      elseif(velocity < 0)then
         if(k+2<=kmax)then
           fz = d2dz*(-f0(i,j,k+2)+4*f0(i,j,k+1)-3*f0(i,j,k))
         elseif(k+1<=kmax)then
           fz = d2dz*(-f0(i,j,k-1)+f0(i,j,k+1))
         else
           fz = d2dz*(3*f0(i,j,k)-4*f0(i,j,k-1)+f0(i,j,k-2))
         endif
      endif
#endif      
      f_rhs(i,j,k) = -velocity*(fz+f0(i,j,k)/Z(k))
      enddo
     enddo
    enddo
   endif

! fix BD
  gp = 6

  gont = any( layer(:,gp) == - 1 )
 
   if( .not. gont ) then

    do k = layer(3,gp), layer(6,gp)
     do j = layer(2,gp), layer(5,gp)
      do i = layer(1,gp), layer(4,gp)
! z direction   
        f_rhs(i,j,k) = ZEO
      enddo
     enddo
    enddo
   endif

  return

  end subroutine sommerfeld_routbam_ss
! falloff boundary condition 
  subroutine falloff_ss(ex,X, Y, Z,xmin,ymin,zmin,xmax,ymax,zmax,f,n,SoA,Symmetry)

  implicit none
 
!~~~~~~> Input parameters:
  integer, intent(in):: ex(1:3),Symmetry,n
  real*8, dimension(ex(1)) :: X
  real*8, dimension(ex(2)) :: Y
! Z-> R  
  real*8, dimension(ex(3)) :: Z
  real*8,  intent(in):: xmin,ymin,zmin,xmax,ymax,zmax
  real*8, dimension(ex(1),ex(2),ex(3)),intent(inout)::f
  real*8,dimension(3),intent(in) ::SoA
!~~~~~~> Other variables:
  logical :: gont
  real*8 :: dZ
  integer :: i, j, k
  real*8 :: d2dz
  integer :: layer(1:6,1:6),gp
! index of layer, first one: i,j,k; second one: front back etc. boundary
  integer :: kmin,kmax
  real*8 :: fz 
  real*8, parameter :: ZEO = 0.d0, ONE = 1.d0, TWO=2.d0
  
  dZ = Z(2) - Z(1)

  d2dz = ONE/TWO/dZ

  kmax = ex(3)

  kmin = 1

layer(1:3,:) = 1
layer(4:6,:) =-1
 
if(dabs(Z(ex(3))-zmax) < dZ)then
  layer(1,3) = 1
  layer(2,3) = 1
  layer(4,3) = ex(1)
  layer(5,3) = ex(2)
  layer(3,3) = ex(3)
  layer(6,3) = ex(3)
endif

! falloff BD
  gp = 3

  gont = any( layer(:,gp) == - 1 )
 
   if( .not. gont ) then

    do k = layer(3,gp), layer(6,gp)
     do j = layer(2,gp), layer(5,gp)
      do i = layer(1,gp), layer(4,gp)
! z direction   
        f(i,j,k) = f(i,j,k-1)*((Z(k)+Z(k-1))/n/dZ-1)/((Z(k)+Z(k-1))/n/dZ+1)
      enddo
     enddo
    enddo
   endif

  return

  end subroutine falloff_ss
