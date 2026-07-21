

#include "macrodef.fh"

! we need only distinguish different finite difference order
! Vertex or Cell is distinguished in routine symmetry_bd which locates in
! file "fmisc.f90"

! fourth order code

!-----------------------------------------------------------------------------------------------------------------
!
! General first derivatives of 4_th oder accurate
!
!              f(i-2) - 8 f(i-1) + 8 f(i+1) - f(i+2)
!  fx(i) = ---------------------------------------------
!                             12 dx
!
!-----------------------------------------------------------------------------------------------------------------

  subroutine fderivs(ex,f,fx,fy,fz,X,Y,Z,SYM1,SYM2,SYM3,symmetry,onoff)
  implicit none

  integer,                               intent(in ):: ex(1:3),symmetry,onoff
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(in ):: f
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out):: fx,fy,fz
  real*8,                                intent(in) :: X(ex(1)),Y(ex(2)),Z(ex(3))
  real*8,                                intent(in ):: SYM1,SYM2,SYM3

!~~~~~~ other variables

  real*8 :: dX,dY,dZ
  real*8,dimension(-1:ex(1),-1:ex(2),-1:ex(3))   :: fh
  real*8, dimension(3) :: SoA
  integer :: imin,jmin,kmin,imax,jmax,kmax,i,j,k
  real*8 :: d12dx,d12dy,d12dz,d2dx,d2dy,d2dz
  integer, parameter :: NO_SYMM = 0, EQ_SYMM = 1, OCTANT = 2
  real*8,  parameter :: ZEO=0.d0,ONE=1.d0, F60=6.d1
  real*8,  parameter :: TWO=2.d0,EIT=8.d0
  real*8,  parameter ::  F9=9.d0,F45=4.5d1,F12=1.2d1

  dX = X(2)-X(1)
  dY = Y(2)-Y(1)
  dZ = Z(2)-Z(1)

  imax = ex(1)
  jmax = ex(2)
  kmax = ex(3)

  imin = 1
  jmin = 1
  kmin = 1
  if(Symmetry > NO_SYMM .and. dabs(Z(1)) < dZ) kmin = -1
  if(Symmetry > EQ_SYMM .and. dabs(X(1)) < dX) imin = -1
  if(Symmetry > EQ_SYMM .and. dabs(Y(1)) < dY) jmin = -1

  SoA(1) = SYM1
  SoA(2) = SYM2
  SoA(3) = SYM3

  call symmetry_bd(2,ex,f,fh,SoA)

  d12dx = ONE/F12/dX
  d12dy = ONE/F12/dY
  d12dz = ONE/F12/dZ

  d2dx = ONE/TWO/dX
  d2dy = ONE/TWO/dY
  d2dz = ONE/TWO/dZ

  fx = ZEO
  fy = ZEO
  fz = ZEO

  do k=1,ex(3)-1
  do j=1,ex(2)-1
  do i=1,ex(1)-1
#if 0  
! x direction   
        if(i+2 <= imax .and. i-2 >= imin)then
!
!              f(i-2) - 8 f(i-1) + 8 f(i+1) - f(i+2)
!  fx(i) = ---------------------------------------------
!                             12 dx
      fx(i,j,k)=d12dx*(fh(i-2,j,k)-EIT*fh(i-1,j,k)+EIT*fh(i+1,j,k)-fh(i+2,j,k))

    elseif(i+1 <= imax .and. i-1 >= imin)then
!
!              - f(i-1) + f(i+1)
!  fx(i) = --------------------------------
!                     2 dx
      fx(i,j,k)=d2dx*(-fh(i-1,j,k)+fh(i+1,j,k))

! set imax and imin 0
    endif
! y direction   
        if(j+2 <= jmax .and. j-2 >= jmin)then

      fy(i,j,k)=d12dy*(fh(i,j-2,k)-EIT*fh(i,j-1,k)+EIT*fh(i,j+1,k)-fh(i,j+2,k))

    elseif(j+1 <= jmax .and. j-1 >= jmin)then

     fy(i,j,k)=d2dy*(-fh(i,j-1,k)+fh(i,j+1,k))

! set jmax and jmin 0
    endif
! z direction   
        if(k+2 <= kmax .and. k-2 >= kmin)then

      fz(i,j,k)=d12dz*(fh(i,j,k-2)-EIT*fh(i,j,k-1)+EIT*fh(i,j,k+1)-fh(i,j,k+2))

    elseif(k+1 <= kmax .and. k-1 >= kmin)then

      fz(i,j,k)=d2dz*(-fh(i,j,k-1)+fh(i,j,k+1))

! set kmax and kmin 0
    endif
#elif 0
! x direction   
        if(i+2 <= imax .and. i-2 >= imin)then
!
!              f(i-2) - 8 f(i-1) + 8 f(i+1) - f(i+2)
!  fx(i) = ---------------------------------------------
!                             12 dx
      fx(i,j,k)=d12dx*(fh(i-2,j,k)-EIT*fh(i-1,j,k)+EIT*fh(i+1,j,k)-fh(i+2,j,k))

    elseif(i+3 <= imax .and. i-1 >= imin)then
      fx(i,j,k)=d12dx*(-3.d0*fh(i-1,j,k)-1.d1*fh(i,j,k)+1.8d1*fh(i+1,j,k)-6.d0*fh(i+2,j,k)+fh(i+3,j,k))
    elseif(i+1 <= imax .and. i-3 >= imin)then
      fx(i,j,k)=d12dx*( 3.d0*fh(i+1,j,k)+1.d1*fh(i,j,k)-1.8d1*fh(i-1,j,k)+6.d0*fh(i-2,j,k)-fh(i-3,j,k))
! set imax and imin 0
    endif
! y direction   
        if(j+2 <= jmax .and. j-2 >= jmin)then

      fy(i,j,k)=d12dy*(fh(i,j-2,k)-EIT*fh(i,j-1,k)+EIT*fh(i,j+1,k)-fh(i,j+2,k))

    elseif(j+3 <= jmax .and. j-1 >= jmin)then
      fy(i,j,k)=d12dy*(-3.d0*fh(i,j-1,k)-1.d1*fh(i,j,k)+1.8d1*fh(i,j+1,k)-6.d0*fh(i,j+2,k)+fh(i,j+3,k))
    elseif(j+1 <= jmax .and. j-3 >= jmin)then
      fy(i,j,k)=d12dy*( 3.d0*fh(i,j+1,k)+1.d1*fh(i,j,k)-1.8d1*fh(i,j-1,k)+6.d0*fh(i,j-2,k)-fh(i,j-3,k))

! set jmax and jmin 0
    endif
! z direction   
        if(k+2 <= kmax .and. k-2 >= kmin)then

      fz(i,j,k)=d12dz*(fh(i,j,k-2)-EIT*fh(i,j,k-1)+EIT*fh(i,j,k+1)-fh(i,j,k+2))

    elseif(k+3 <= kmax .and. k-1 >= kmin)then
      fz(i,j,k)=d12dz*(-3.d0*fh(i,j,k-1)-1.d1*fh(i,j,k)+1.8d1*fh(i,j,k+1)-6.d0*fh(i,j,k+2)+fh(i,j,k+3))
    elseif(k+1 <= kmax .and. k-3 >= kmin)then
      fz(i,j,k)=d12dz*( 3.d0*fh(i,j,k+1)+1.d1*fh(i,j,k)-1.8d1*fh(i,j,k-1)+6.d0*fh(i,j,k-2)-fh(i,j,k-3))

! set kmax and kmin 0
    endif
#else
! for bam comparison
   if(i+2 <= imax .and. i-2 >= imin .and. &
      j+2 <= jmax .and. j-2 >= jmin .and. &
      k+2 <= kmax .and. k-2 >= kmin) then
      fx(i,j,k)=d12dx*(fh(i-2,j,k)-EIT*fh(i-1,j,k)+EIT*fh(i+1,j,k)-fh(i+2,j,k))
      fy(i,j,k)=d12dy*(fh(i,j-2,k)-EIT*fh(i,j-1,k)+EIT*fh(i,j+1,k)-fh(i,j+2,k))
      fz(i,j,k)=d12dz*(fh(i,j,k-2)-EIT*fh(i,j,k-1)+EIT*fh(i,j,k+1)-fh(i,j,k+2))
   elseif(i+1 <= imax .and. i-1 >= imin .and. &
          j+1 <= jmax .and. j-1 >= jmin .and. &
          k+1 <= kmax .and. k-1 >= kmin) then
      fx(i,j,k)=d2dx*(-fh(i-1,j,k)+fh(i+1,j,k))
      fy(i,j,k)=d2dy*(-fh(i,j-1,k)+fh(i,j+1,k))
      fz(i,j,k)=d2dz*(-fh(i,j,k-1)+fh(i,j,k+1))
   endif
#endif
  enddo
  enddo
  enddo

  return

  end subroutine fderivs
!-----------------------------------------------------------------------------
!
! single derivatives dx
!
!-----------------------------------------------------------------------------
  subroutine fdx(ex,f,fx,X,SYM1,symmetry,onoff)
  implicit none

  integer,                               intent(in ):: ex(1:3),symmetry,onoff
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(in ):: f
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out):: fx
  real*8,                                intent(in ):: X(ex(1)),SYM1

!~~~~~~ other variables

  real*8 :: dX
  real*8,dimension(-1:ex(1),-1:ex(2),-1:ex(3))   :: fh
  real*8, dimension(3) :: SoA
  integer :: imin,jmin,kmin,imax,jmax,kmax,i,j,k
  real*8 :: d12dx,d2dx
  integer, parameter :: NO_SYMM = 0, EQ_SYMM = 1, OCTANT = 2
  real*8,  parameter :: ZEO=0.d0,ONE=1.d0, F60=6.d1
  real*8,  parameter :: TWO=2.d0,EIT=8.d0
  real*8,  parameter ::  F9=9.d0,F45=4.5d1,F12=1.2d1

  dX = X(2)-X(1)
  
  imax = ex(1)
  jmax = ex(2)
  kmax = ex(3)

  imin = 1
  jmin = 1
  kmin = 1
  if(Symmetry > EQ_SYMM .and. dabs(X(1)) < dX) imin = -1

  SoA(1) = SYM1
! no use  
  SoA(2) = SYM1
  SoA(3) = SYM1

  call symmetry_bd(2,ex,f,fh,SoA)

  d12dx = ONE/F12/dX

  d2dx = ONE/TWO/dX

  fx = ZEO

  do k=1,ex(3)-1
  do j=1,ex(2)-1
  do i=1,ex(1)-1
! x direction   
        if(i+2 <= imax .and. i-2 >= imin)then
!
!              f(i-2) - 8 f(i-1) + 8 f(i+1) - f(i+2)
!  fx(i) = ---------------------------------------------
!                             12 dx
      fx(i,j,k)=d12dx*(fh(i-2,j,k)-EIT*fh(i-1,j,k)+EIT*fh(i+1,j,k)-fh(i+2,j,k))

    elseif(i+1 <= imax .and. i-1 >= imin)then
!
!              - f(i-1) + f(i+1)
!  fx(i) = --------------------------------
!                     2 dx
      fx(i,j,k)=d2dx*(-fh(i-1,j,k)+fh(i+1,j,k))

! set imax and imin 0
    endif

  enddo
  enddo
  enddo

  return

  end subroutine fdx
!-----------------------------------------------------------------------------
!
! single derivatives dy
!
!-----------------------------------------------------------------------------
  subroutine fdy(ex,f,fy,Y,SYM2,symmetry,onoff)
  implicit none

  integer,                               intent(in ):: ex(1:3),symmetry,onoff
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(in ):: f
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out):: fy
  real*8,                                intent(in ):: Y(ex(2)),SYM2

!~~~~~~ other variables

  real*8 :: dY
  real*8,dimension(-1:ex(1),-1:ex(2),-1:ex(3))   :: fh
  real*8, dimension(3) :: SoA
  integer :: imin,jmin,kmin,imax,jmax,kmax,i,j,k
  real*8 :: d12dy,d2dy
  integer, parameter :: NO_SYMM = 0, EQ_SYMM = 1, OCTANT = 2
  real*8,  parameter :: ZEO=0.d0,ONE=1.d0, F60=6.d1
  real*8,  parameter :: TWO=2.d0,EIT=8.d0
  real*8,  parameter ::  F9=9.d0,F45=4.5d1,F12=1.2d1

  dY = Y(2)-Y(1)
  
  imax = ex(1)
  jmax = ex(2)
  kmax = ex(3)

  imin = 1
  jmin = 1
  kmin = 1
  if(Symmetry > EQ_SYMM .and. dabs(Y(1)) < dY) jmin = -1

  SoA(1) = SYM2
  SoA(2) = SYM2
  SoA(3) = SYM2

  call symmetry_bd(2,ex,f,fh,SoA)

  d12dy = ONE/F12/dY

  d2dy = ONE/TWO/dY

  fy = ZEO

  do k=1,ex(3)-1
  do j=1,ex(2)-1
  do i=1,ex(1)-1
! y direction   
        if(j+2 <= jmax .and. j-2 >= jmin)then

      fy(i,j,k)=d12dy*(fh(i,j-2,k)-EIT*fh(i,j-1,k)+EIT*fh(i,j+1,k)-fh(i,j+2,k))

    elseif(j+1 <= jmax .and. j-1 >= jmin)then

     fy(i,j,k)=d2dy*(-fh(i,j-1,k)+fh(i,j+1,k))

! set jmax and jmin 0
    endif

  enddo
  enddo
  enddo

  return

  end subroutine fdy
!-----------------------------------------------------------------------------
!
! single derivatives dz
!
!-----------------------------------------------------------------------------
  subroutine fdz(ex,f,fz,Z,SYM3,symmetry,onoff)
  implicit none

  integer,                               intent(in ):: ex(1:3),symmetry,onoff
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(in ):: f
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out):: fz
  real*8,                                intent(in ):: Z(ex(3)),SYM3

!~~~~~~ other variables
  
  real*8 :: dZ
  real*8,dimension(-1:ex(1),-1:ex(2),-1:ex(3))   :: fh
  real*8, dimension(3) :: SoA
  integer :: imin,jmin,kmin,imax,jmax,kmax,i,j,k
  real*8 :: d12dz,d2dz
  integer, parameter :: NO_SYMM = 0, EQ_SYMM = 1, OCTANT = 2
  real*8,  parameter :: ZEO=0.d0,ONE=1.d0, F60=6.d1
  real*8,  parameter :: TWO=2.d0,EIT=8.d0
  real*8,  parameter ::  F9=9.d0,F45=4.5d1,F12=1.2d1

  dZ = Z(2)-Z(1)
  
  imax = ex(1)
  jmax = ex(2)
  kmax = ex(3)

  imin = 1
  jmin = 1
  kmin = 1
  if(Symmetry > NO_SYMM .and. dabs(Z(1)) < dZ) kmin = -1

  SoA(1) = SYM3
  SoA(2) = SYM3
  SoA(3) = SYM3

  call symmetry_bd(2,ex,f,fh,SoA)

  d12dz = ONE/F12/dZ

  d2dz = ONE/TWO/dZ

  fz = ZEO

  do k=1,ex(3)-1
  do j=1,ex(2)-1
  do i=1,ex(1)-1
! z direction   
        if(k+2 <= kmax .and. k-2 >= kmin)then

      fz(i,j,k)=d12dz*(fh(i,j,k-2)-EIT*fh(i,j,k-1)+EIT*fh(i,j,k+1)-fh(i,j,k+2))

    elseif(k+1 <= kmax .and. k-1 >= kmin)then

      fz(i,j,k)=d2dz*(-fh(i,j,k-1)+fh(i,j,k+1))

! set kmax and kmin 0
    endif

  enddo
  enddo
  enddo

  return

  end subroutine fdz
!-----------------------------------------------------------------------------------------------------------------
!
! General second derivatives of 4_th oder accurate
!
!               - f(i-2) + 16 f(i-1) - 30 f(i) + 16 f(i+1) - f(i+2)
!  fxx(i) = ----------------------------------------------------------
!                                  12 dx^2 
!
!             -   ( - f(i+2,j+2) + 8 f(i+1,j+2) - 8 f(i-1,j+2) + f(i-2,j+2) )
!             + 8 ( - f(i+2,j+1) + 8 f(i+1,j+1) - 8 f(i-1,j+1) + f(i-2,j+1) )
!             - 8 ( - f(i+2,j-1) + 8 f(i+1,j-1) - 8 f(i-1,j-1) + f(i-2,j-1) )
!             +   ( - f(i+2,j-2) + 8 f(i+1,j-2) - 8 f(i-1,j-2) + f(i-2,j-2) )
!  fxy(i,j) = ----------------------------------------------------------------
!                                  144 dx dy
!
!-----------------------------------------------------------------------------------------------------------------
  subroutine fdderivs(ex,f,fxx,fxy,fxz,fyy,fyz,fzz,X,Y,Z, &
                      SYM1,SYM2,SYM3,symmetry,onoff)
  implicit none

  integer,                             intent(in ):: ex(1:3),symmetry,onoff
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ):: f
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out):: fxx,fxy,fxz,fyy,fyz,fzz
  real*8,                              intent(in ):: X(ex(1)),Y(ex(2)),Z(ex(3)),SYM1,SYM2,SYM3

!~~~~~~ other variables
  real*8 :: dX,dY,dZ
  real*8,dimension(-1:ex(1),-1:ex(2),-1:ex(3))   :: fh
  real*8, dimension(3) :: SoA
  integer :: imin,jmin,kmin,imax,jmax,kmax,i,j,k
  real*8  :: Sdxdx,Sdydy,Sdzdz,Fdxdx,Fdydy,Fdzdz
  real*8  :: Sdxdy,Sdxdz,Sdydz,Fdxdy,Fdxdz,Fdydz
  integer, parameter :: NO_SYMM = 0, EQ_SYMM = 1, OCTANT = 2
  real*8, parameter :: ZEO=0.d0, ONE=1.d0, TWO=2.d0, F1o4=2.5d-1, F9=9.d0,  F45=4.5d1
  real*8, parameter :: F8=8.d0, F16=1.6d1, F30=3.d1, F27=2.7d1, F270=2.7d2, F490=4.9d2
  real*8, parameter :: F1o6=ONE/6.d0, F1o12=ONE/1.2d1, F1o144=ONE/1.44d2
  real*8, parameter :: F1o180=ONE/1.8d2,F1o3600=ONE/3.6d3

  dX = X(2)-X(1)
  dY = Y(2)-Y(1)
  dZ = Z(2)-Z(1)

  imax = ex(1)
  jmax = ex(2)
  kmax = ex(3)

  imin = 1
  jmin = 1
  kmin = 1
  if(Symmetry > NO_SYMM .and. dabs(Z(1)) < dZ) kmin = -1
  if(Symmetry > EQ_SYMM .and. dabs(X(1)) < dX) imin = -1
  if(Symmetry > EQ_SYMM .and. dabs(Y(1)) < dY) jmin = -1

  SoA(1) = SYM1
  SoA(2) = SYM2
  SoA(3) = SYM3

  call symmetry_bd(2,ex,f,fh,SoA)

  Sdxdx =  ONE /( dX * dX )
  Sdydy =  ONE /( dY * dY )
  Sdzdz =  ONE /( dZ * dZ )

  Fdxdx = F1o12 /( dX * dX )
  Fdydy = F1o12 /( dY * dY )
  Fdzdz = F1o12 /( dZ * dZ )

  Sdxdy = F1o4 /( dX * dY )
  Sdxdz = F1o4 /( dX * dZ )
  Sdydz = F1o4 /( dY * dZ )

  Fdxdy = F1o144 /( dX * dY )
  Fdxdz = F1o144 /( dX * dZ )
  Fdydz = F1o144 /( dY * dZ )

  fxx = ZEO
  fyy = ZEO
  fzz = ZEO
  fxy = ZEO
  fxz = ZEO
  fyz = ZEO

  do k=1,ex(3)-1
  do j=1,ex(2)-1
  do i=1,ex(1)-1
#if 0  
!~~~~~~ fxx
        if(i+2 <= imax .and. i-2 >= imin)then
!
!               - f(i-2) + 16 f(i-1) - 30 f(i) + 16 f(i+1) - f(i+2)
!  fxx(i) = ----------------------------------------------------------
!                                  12 dx^2 
   fxx(i,j,k) = Fdxdx*(-fh(i-2,j,k)+F16*fh(i-1,j,k)-F30*fh(i,j,k) &
                       -fh(i+2,j,k)+F16*fh(i+1,j,k)              )
   elseif(i+1 <= imax .and. i-1 >= imin)then
!
!               f(i-1) - 2 f(i) + f(i+1)
!  fxx(i) = --------------------------------
!                         dx^2 
   fxx(i,j,k) = Sdxdx*(fh(i-1,j,k)-TWO*fh(i,j,k) &
                      +fh(i+1,j,k)              )
   endif


!~~~~~~ fyy
        if(j+2 <= jmax .and. j-2 >= jmin)then

   fyy(i,j,k) = Fdydy*(-fh(i,j-2,k)+F16*fh(i,j-1,k)-F30*fh(i,j,k) &
                       -fh(i,j+2,k)+F16*fh(i,j+1,k)              )
   elseif(j+1 <= jmax .and. j-1 >= jmin)then

   fyy(i,j,k) = Sdydy*(fh(i,j-1,k)-TWO*fh(i,j,k) &
                      +fh(i,j+1,k)              )
   endif

!~~~~~~ fzz
        if(k+2 <= kmax .and. k-2 >= kmin)then

   fzz(i,j,k) = Fdzdz*(-fh(i,j,k-2)+F16*fh(i,j,k-1)-F30*fh(i,j,k) &
                       -fh(i,j,k+2)+F16*fh(i,j,k+1)              )
   elseif(k+1 <= kmax .and. k-1 >= kmin)then

   fzz(i,j,k) = Sdzdz*(fh(i,j,k-1)-TWO*fh(i,j,k) &
                      +fh(i,j,k+1)              )
   endif
!~~~~~~ fxy
       if(i+2 <= imax .and. i-2 >= imin .and. j+2 <= jmax .and. j-2 >= jmin)then
!
!                 ( f(i-2,j-2) - 8 f(i-1,j-2) + 8 f(i+1,j-2) - f(i+2,j-2) )
!             - 8 ( f(i-2,j-1) - 8 f(i-1,j-1) + 8 f(i+1,j-1) - f(i+2,j-1) )
!             + 8 ( f(i-2,j+1) - 8 f(i-1,j+1) + 8 f(i+1,j+1) - f(i+2,j+1) )
!             -   ( f(i-2,j+2) - 8 f(i-1,j+2) + 8 f(i+1,j+2) - f(i+2,j+2) )
!  fxy(i,j) = ----------------------------------------------------------------
!                                  144 dx dy
   fxy(i,j,k) = Fdxdy*(     (fh(i-2,j-2,k)-F8*fh(i-1,j-2,k)+F8*fh(i+1,j-2,k)-fh(i+2,j-2,k))  &
                       -F8 *(fh(i-2,j-1,k)-F8*fh(i-1,j-1,k)+F8*fh(i+1,j-1,k)-fh(i+2,j-1,k))  &
                       +F8 *(fh(i-2,j+1,k)-F8*fh(i-1,j+1,k)+F8*fh(i+1,j+1,k)-fh(i+2,j+1,k))  &
                       -    (fh(i-2,j+2,k)-F8*fh(i-1,j+2,k)+F8*fh(i+1,j+2,k)-fh(i+2,j+2,k)))

   elseif(i+1 <= imax .and. i-1 >= imin .and. j+1 <= jmax .and. j-1 >= jmin)then
!                 f(i-1,j-1) - f(i+1,j-1) - f(i-1,j+1) + f(i+1,j+1) 
!  fxy(i,j) = -----------------------------------------------------------
!                                      4 dx dy
   fxy(i,j,k) = Sdxdy*(fh(i-1,j-1,k)-fh(i+1,j-1,k)-fh(i-1,j+1,k)+fh(i+1,j+1,k))
   endif
!~~~~~~ fxz
       if(i+2 <= imax .and. i-2 >= imin .and. k+2 <= kmax .and. k-2 >= kmin)then
   fxz(i,j,k) = Fdxdz*(     (fh(i-2,j,k-2)-F8*fh(i-1,j,k-2)+F8*fh(i+1,j,k-2)-fh(i+2,j,k-2))  &
                       -F8 *(fh(i-2,j,k-1)-F8*fh(i-1,j,k-1)+F8*fh(i+1,j,k-1)-fh(i+2,j,k-1))  &
                       +F8 *(fh(i-2,j,k+1)-F8*fh(i-1,j,k+1)+F8*fh(i+1,j,k+1)-fh(i+2,j,k+1))  &
                       -    (fh(i-2,j,k+2)-F8*fh(i-1,j,k+2)+F8*fh(i+1,j,k+2)-fh(i+2,j,k+2)))
   elseif(i+1 <= imax .and. i-1 >= imin .and. k+1 <= kmax .and. k-1 >= kmin)then
   fxz(i,j,k) = Sdxdz*(fh(i-1,j,k-1)-fh(i+1,j,k-1)-fh(i-1,j,k+1)+fh(i+1,j,k+1))
   endif
!~~~~~~ fyz
       if(j+2 <= jmax .and. j-2 >= jmin .and. k+2 <= kmax .and. k-2 >= kmin)then
   fyz(i,j,k) = Fdydz*(     (fh(i,j-2,k-2)-F8*fh(i,j-1,k-2)+F8*fh(i,j+1,k-2)-fh(i,j+2,k-2))  &
                       -F8 *(fh(i,j-2,k-1)-F8*fh(i,j-1,k-1)+F8*fh(i,j+1,k-1)-fh(i,j+2,k-1))  &
                       +F8 *(fh(i,j-2,k+1)-F8*fh(i,j-1,k+1)+F8*fh(i,j+1,k+1)-fh(i,j+2,k+1))  &
                       -    (fh(i,j-2,k+2)-F8*fh(i,j-1,k+2)+F8*fh(i,j+1,k+2)-fh(i,j+2,k+2)))
   elseif(j+1 <= jmax .and. j-1 >= jmin .and. k+1 <= kmax .and. k-1 >= kmin)then
   fyz(i,j,k) = Sdydz*(fh(i,j-1,k-1)-fh(i,j+1,k-1)-fh(i,j-1,k+1)+fh(i,j+1,k+1))
   endif 
#else
! for bam comparison
   if(i+2 <= imax .and. i-2 >= imin .and. &
      j+2 <= jmax .and. j-2 >= jmin .and. &
      k+2 <= kmax .and. k-2 >= kmin) then
   fxx(i,j,k) = Fdxdx*(-fh(i-2,j,k)+F16*fh(i-1,j,k)-F30*fh(i,j,k) &
                       -fh(i+2,j,k)+F16*fh(i+1,j,k)              )
   fyy(i,j,k) = Fdydy*(-fh(i,j-2,k)+F16*fh(i,j-1,k)-F30*fh(i,j,k) &
                       -fh(i,j+2,k)+F16*fh(i,j+1,k)              )
   fzz(i,j,k) = Fdzdz*(-fh(i,j,k-2)+F16*fh(i,j,k-1)-F30*fh(i,j,k) &
                       -fh(i,j,k+2)+F16*fh(i,j,k+1)              )
   fxy(i,j,k) = Fdxdy*(     (fh(i-2,j-2,k)-F8*fh(i-1,j-2,k)+F8*fh(i+1,j-2,k)-fh(i+2,j-2,k))  &
                       -F8 *(fh(i-2,j-1,k)-F8*fh(i-1,j-1,k)+F8*fh(i+1,j-1,k)-fh(i+2,j-1,k))  &
                       +F8 *(fh(i-2,j+1,k)-F8*fh(i-1,j+1,k)+F8*fh(i+1,j+1,k)-fh(i+2,j+1,k))  &
                       -    (fh(i-2,j+2,k)-F8*fh(i-1,j+2,k)+F8*fh(i+1,j+2,k)-fh(i+2,j+2,k)))
   fxz(i,j,k) = Fdxdz*(     (fh(i-2,j,k-2)-F8*fh(i-1,j,k-2)+F8*fh(i+1,j,k-2)-fh(i+2,j,k-2))  &
                       -F8 *(fh(i-2,j,k-1)-F8*fh(i-1,j,k-1)+F8*fh(i+1,j,k-1)-fh(i+2,j,k-1))  &
                       +F8 *(fh(i-2,j,k+1)-F8*fh(i-1,j,k+1)+F8*fh(i+1,j,k+1)-fh(i+2,j,k+1))  &
                       -    (fh(i-2,j,k+2)-F8*fh(i-1,j,k+2)+F8*fh(i+1,j,k+2)-fh(i+2,j,k+2)))
   fyz(i,j,k) = Fdydz*(     (fh(i,j-2,k-2)-F8*fh(i,j-1,k-2)+F8*fh(i,j+1,k-2)-fh(i,j+2,k-2))  &
                       -F8 *(fh(i,j-2,k-1)-F8*fh(i,j-1,k-1)+F8*fh(i,j+1,k-1)-fh(i,j+2,k-1))  &
                       +F8 *(fh(i,j-2,k+1)-F8*fh(i,j-1,k+1)+F8*fh(i,j+1,k+1)-fh(i,j+2,k+1))  &
                       -    (fh(i,j-2,k+2)-F8*fh(i,j-1,k+2)+F8*fh(i,j+1,k+2)-fh(i,j+2,k+2)))
   elseif(i+1 <= imax .and. i-1 >= imin .and. &
          j+1 <= jmax .and. j-1 >= jmin .and. &
          k+1 <= kmax .and. k-1 >= kmin) then
   fxx(i,j,k) = Sdxdx*(fh(i-1,j,k)-TWO*fh(i,j,k) &
                      +fh(i+1,j,k)              )
   fyy(i,j,k) = Sdydy*(fh(i,j-1,k)-TWO*fh(i,j,k) &
                      +fh(i,j+1,k)              )
   fzz(i,j,k) = Sdzdz*(fh(i,j,k-1)-TWO*fh(i,j,k) &
                      +fh(i,j,k+1)              )
   fxy(i,j,k) = Sdxdy*(fh(i-1,j-1,k)-fh(i+1,j-1,k)-fh(i-1,j+1,k)+fh(i+1,j+1,k))
   fxz(i,j,k) = Sdxdz*(fh(i-1,j,k-1)-fh(i+1,j,k-1)-fh(i-1,j,k+1)+fh(i+1,j,k+1))
   fyz(i,j,k) = Sdydz*(fh(i,j-1,k-1)-fh(i,j+1,k-1)-fh(i,j-1,k+1)+fh(i,j+1,k+1))
   endif
#endif
   enddo
   enddo
   enddo

  return

  end subroutine fdderivs
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! only for compute_ricci.f90 usage
!-----------------------------------------------------------------------------
  subroutine fddxx(ex,f,fxx,X,Y,Z,SYM1,SYM2,SYM3,symmetry)
  implicit none

  integer,                             intent(in ):: ex(1:3),symmetry
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ):: f
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out):: fxx
  real*8,                              intent(in ):: X(ex(1)),Y(ex(2)),Z(ex(3)),SYM1,SYM2,SYM3

!~~~~~~ other variables
  real*8 :: dX,dY,dZ
  real*8,dimension(-1:ex(1),-1:ex(2),-1:ex(3))   :: fh
  real*8, dimension(3) :: SoA
  integer :: imin,jmin,kmin,imax,jmax,kmax,i,j,k
  real*8  :: Sdxdx,Fdxdx
  integer, parameter :: NO_SYMM = 0, EQ_SYMM = 1, OCTANT = 2
  real*8, parameter :: ZEO=0.d0, ONE=1.d0, TWO=2.d0, F1o4=2.5d-1, F9=9.d0,  F45=4.5d1
  real*8, parameter :: F8=8.d0, F16=1.6d1, F30=3.d1, F27=2.7d1, F270=2.7d2, F490=4.9d2
  real*8, parameter :: F1o6=ONE/6.d0, F1o12=ONE/1.2d1, F1o144=ONE/1.44d2
  real*8, parameter :: F1o180=ONE/1.8d2,F1o3600=ONE/3.6d3

  dX = X(2)-X(1)
  dY = Y(2)-Y(1)
  dZ = Z(2)-Z(1)

  imax = ex(1)
  jmax = ex(2)
  kmax = ex(3)

  imin = 1
  jmin = 1
  kmin = 1
  if(Symmetry > NO_SYMM .and. dabs(Z(1)) < dZ) kmin = -1
  if(Symmetry > EQ_SYMM .and. dabs(X(1)) < dX) imin = -1
  if(Symmetry > EQ_SYMM .and. dabs(Y(1)) < dY) jmin = -1

  SoA(1) = SYM1
  SoA(2) = SYM2
  SoA(3) = SYM3

  call symmetry_bd(2,ex,f,fh,SoA)

  Sdxdx =  ONE /( dX * dX )

  Fdxdx = F1o12 /( dX * dX )

  fxx = ZEO

  do k=1,ex(3)-1
  do j=1,ex(2)-1
  do i=1,ex(1)-1
!~~~~~~ fxx
        if(i+2 <= imax .and. i-2 >= imin)then
   fxx(i,j,k) = Fdxdx*(-fh(i-2,j,k)+F16*fh(i-1,j,k)-F30*fh(i,j,k) &
                       -fh(i+2,j,k)+F16*fh(i+1,j,k)              )
   elseif(i+1 <= imax .and. i-1 >= imin)then
   fxx(i,j,k) = Sdxdx*(fh(i-1,j,k)-TWO*fh(i,j,k) &
                      +fh(i+1,j,k)              )
   endif

   enddo
   enddo
   enddo

  return

  end subroutine fddxx

  subroutine fddyy(ex,f,fyy,X,Y,Z,SYM1,SYM2,SYM3,symmetry)
  implicit none

  integer,                             intent(in ):: ex(1:3),symmetry
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ):: f
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out):: fyy
  real*8,                              intent(in ):: X(ex(1)),Y(ex(2)),Z(ex(3)),SYM1,SYM2,SYM3

!~~~~~~ other variables
  real*8 :: dX,dY,dZ
  real*8,dimension(-1:ex(1),-1:ex(2),-1:ex(3))   :: fh
  real*8, dimension(3) :: SoA
  integer :: imin,jmin,kmin,imax,jmax,kmax,i,j,k
  real*8  :: Sdydy,Fdydy
  integer, parameter :: NO_SYMM = 0, EQ_SYMM = 1, OCTANT = 2
  real*8, parameter :: ZEO=0.d0, ONE=1.d0, TWO=2.d0, F1o4=2.5d-1, F9=9.d0,  F45=4.5d1
  real*8, parameter :: F8=8.d0, F16=1.6d1, F30=3.d1, F27=2.7d1, F270=2.7d2, F490=4.9d2
  real*8, parameter :: F1o6=ONE/6.d0, F1o12=ONE/1.2d1, F1o144=ONE/1.44d2
  real*8, parameter :: F1o180=ONE/1.8d2,F1o3600=ONE/3.6d3

  dX = X(2)-X(1)
  dY = Y(2)-Y(1)
  dZ = Z(2)-Z(1)

  imax = ex(1)
  jmax = ex(2)
  kmax = ex(3)

  imin = 1
  jmin = 1
  kmin = 1
  if(Symmetry > NO_SYMM .and. dabs(Z(1)) < dZ) kmin = -1
  if(Symmetry > EQ_SYMM .and. dabs(X(1)) < dX) imin = -1
  if(Symmetry > EQ_SYMM .and. dabs(Y(1)) < dY) jmin = -1

  SoA(1) = SYM1
  SoA(2) = SYM2
  SoA(3) = SYM3

  call symmetry_bd(2,ex,f,fh,SoA)

  Sdydy =  ONE /( dY * dY )

  Fdydy = F1o12 /( dY * dY )

  fyy = ZEO

  do k=1,ex(3)-1
  do j=1,ex(2)-1
  do i=1,ex(1)-1
!~~~~~~ fyy
        if(j+2 <= jmax .and. j-2 >= jmin)then

   fyy(i,j,k) = Fdydy*(-fh(i,j-2,k)+F16*fh(i,j-1,k)-F30*fh(i,j,k) &
                       -fh(i,j+2,k)+F16*fh(i,j+1,k)              )
   elseif(j+1 <= jmax .and. j-1 >= jmin)then

   fyy(i,j,k) = Sdydy*(fh(i,j-1,k)-TWO*fh(i,j,k) &
                      +fh(i,j+1,k)              )
   endif

   enddo
   enddo
   enddo

  return

  end subroutine fddyy

  subroutine fddzz(ex,f,fzz,X,Y,Z,SYM1,SYM2,SYM3,symmetry)
  implicit none

  integer,                             intent(in ):: ex(1:3),symmetry
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ):: f
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out):: fzz
  real*8,                              intent(in ):: X(ex(1)),Y(ex(2)),Z(ex(3)),SYM1,SYM2,SYM3

!~~~~~~ other variables
  real*8 :: dX,dY,dZ
  real*8,dimension(-1:ex(1),-1:ex(2),-1:ex(3))   :: fh
  real*8, dimension(3) :: SoA
  integer :: imin,jmin,kmin,imax,jmax,kmax,i,j,k
  real*8  :: Sdzdz,Fdzdz
  integer, parameter :: NO_SYMM = 0, EQ_SYMM = 1, OCTANT = 2
  real*8, parameter :: ZEO=0.d0, ONE=1.d0, TWO=2.d0, F1o4=2.5d-1, F9=9.d0,  F45=4.5d1
  real*8, parameter :: F8=8.d0, F16=1.6d1, F30=3.d1, F27=2.7d1, F270=2.7d2, F490=4.9d2
  real*8, parameter :: F1o6=ONE/6.d0, F1o12=ONE/1.2d1, F1o144=ONE/1.44d2
  real*8, parameter :: F1o180=ONE/1.8d2,F1o3600=ONE/3.6d3

  dX = X(2)-X(1)
  dY = Y(2)-Y(1)
  dZ = Z(2)-Z(1)

  imax = ex(1)
  jmax = ex(2)
  kmax = ex(3)

  imin = 1
  jmin = 1
  kmin = 1
  if(Symmetry > NO_SYMM .and. dabs(Z(1)) < dZ) kmin = -1
  if(Symmetry > EQ_SYMM .and. dabs(X(1)) < dX) imin = -1
  if(Symmetry > EQ_SYMM .and. dabs(Y(1)) < dY) jmin = -1

  SoA(1) = SYM1
  SoA(2) = SYM2
  SoA(3) = SYM3

  call symmetry_bd(2,ex,f,fh,SoA)

  Sdzdz =  ONE /( dZ * dZ )

  Fdzdz = F1o12 /( dZ * dZ )

  fzz = ZEO

  do k=1,ex(3)-1
  do j=1,ex(2)-1
  do i=1,ex(1)-1
!~~~~~~ fzz
        if(k+2 <= kmax .and. k-2 >= kmin)then

   fzz(i,j,k) = Fdzdz*(-fh(i,j,k-2)+F16*fh(i,j,k-1)-F30*fh(i,j,k) &
                       -fh(i,j,k+2)+F16*fh(i,j,k+1)              )
   elseif(k+1 <= kmax .and. k-1 >= kmin)then

   fzz(i,j,k) = Sdzdz*(fh(i,j,k-1)-TWO*fh(i,j,k) &
                      +fh(i,j,k+1)              )
   endif

   enddo
   enddo
   enddo

  return

  end subroutine fddzz

  subroutine fddxy(ex,f,fxy,X,Y,Z,SYM1,SYM2,SYM3,symmetry)
  implicit none

  integer,                             intent(in ):: ex(1:3),symmetry
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ):: f
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out):: fxy
  real*8,                              intent(in ):: X(ex(1)),Y(ex(2)),Z(ex(3)),SYM1,SYM2,SYM3

!~~~~~~ other variables
  real*8 :: dX,dY,dZ
  real*8,dimension(-1:ex(1),-1:ex(2),-1:ex(3))   :: fh
  real*8, dimension(3) :: SoA
  integer :: imin,jmin,kmin,imax,jmax,kmax,i,j,k
  real*8  :: Sdxdy,Fdxdy
  integer, parameter :: NO_SYMM = 0, EQ_SYMM = 1, OCTANT = 2
  real*8, parameter :: ZEO=0.d0, ONE=1.d0, TWO=2.d0, F1o4=2.5d-1, F9=9.d0,  F45=4.5d1
  real*8, parameter :: F8=8.d0, F16=1.6d1, F30=3.d1, F27=2.7d1, F270=2.7d2, F490=4.9d2
  real*8, parameter :: F1o6=ONE/6.d0, F1o12=ONE/1.2d1, F1o144=ONE/1.44d2
  real*8, parameter :: F1o180=ONE/1.8d2,F1o3600=ONE/3.6d3

  dX = X(2)-X(1)
  dY = Y(2)-Y(1)
  dZ = Z(2)-Z(1)

  imax = ex(1)
  jmax = ex(2)
  kmax = ex(3)

  imin = 1
  jmin = 1
  kmin = 1
  if(Symmetry > NO_SYMM .and. dabs(Z(1)) < dZ) kmin = -1
  if(Symmetry > EQ_SYMM .and. dabs(X(1)) < dX) imin = -1
  if(Symmetry > EQ_SYMM .and. dabs(Y(1)) < dY) jmin = -1

  SoA(1) = SYM1
  SoA(2) = SYM2
  SoA(3) = SYM3

  call symmetry_bd(2,ex,f,fh,SoA)

  Sdxdy = F1o4 /( dX * dY )

  Fdxdy = F1o144 /( dX * dY )

  fxy = ZEO

  do k=1,ex(3)-1
  do j=1,ex(2)-1
  do i=1,ex(1)-1
!~~~~~~ fxy
       if(i+2 <= imax .and. i-2 >= imin .and. j+2 <= jmax .and. j-2 >= jmin)then

   fxy(i,j,k) = Fdxdy*(     (fh(i-2,j-2,k)-F8*fh(i-1,j-2,k)+F8*fh(i+1,j-2,k)-fh(i+2,j-2,k))  &
                       -F8 *(fh(i-2,j-1,k)-F8*fh(i-1,j-1,k)+F8*fh(i+1,j-1,k)-fh(i+2,j-1,k))  &
                       +F8 *(fh(i-2,j+1,k)-F8*fh(i-1,j+1,k)+F8*fh(i+1,j+1,k)-fh(i+2,j+1,k))  &
                       -    (fh(i-2,j+2,k)-F8*fh(i-1,j+2,k)+F8*fh(i+1,j+2,k)-fh(i+2,j+2,k)))
   elseif(i+1 <= imax .and. i-1 >= imin .and. j+1 <= jmax .and. j-1 >= jmin)then

   fxy(i,j,k) = Sdxdy*(fh(i-1,j-1,k)-fh(i+1,j-1,k)-fh(i-1,j+1,k)+fh(i+1,j+1,k))
   endif

   enddo
   enddo
   enddo

  return

  end subroutine fddxy

  subroutine fddxz(ex,f,fxz,X,Y,Z,SYM1,SYM2,SYM3,symmetry)
  implicit none

  integer,                             intent(in ):: ex(1:3),symmetry
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ):: f
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out):: fxz
  real*8,                              intent(in ):: X(ex(1)),Y(ex(2)),Z(ex(3)),SYM1,SYM2,SYM3

!~~~~~~ other variables
  real*8 :: dX,dY,dZ
  real*8,dimension(-1:ex(1),-1:ex(2),-1:ex(3))   :: fh
  real*8, dimension(3) :: SoA
  integer :: imin,jmin,kmin,imax,jmax,kmax,i,j,k
  real*8  :: Sdxdz,Fdxdz
  integer, parameter :: NO_SYMM = 0, EQ_SYMM = 1, OCTANT = 2
  real*8, parameter :: ZEO=0.d0, ONE=1.d0, TWO=2.d0, F1o4=2.5d-1, F9=9.d0,  F45=4.5d1
  real*8, parameter :: F8=8.d0, F16=1.6d1, F30=3.d1, F27=2.7d1, F270=2.7d2, F490=4.9d2
  real*8, parameter :: F1o6=ONE/6.d0, F1o12=ONE/1.2d1, F1o144=ONE/1.44d2
  real*8, parameter :: F1o180=ONE/1.8d2,F1o3600=ONE/3.6d3

  dX = X(2)-X(1)
  dY = Y(2)-Y(1)
  dZ = Z(2)-Z(1)

  imax = ex(1)
  jmax = ex(2)
  kmax = ex(3)

  imin = 1
  jmin = 1
  kmin = 1
  if(Symmetry > NO_SYMM .and. dabs(Z(1)) < dZ) kmin = -1
  if(Symmetry > EQ_SYMM .and. dabs(X(1)) < dX) imin = -1
  if(Symmetry > EQ_SYMM .and. dabs(Y(1)) < dY) jmin = -1

  SoA(1) = SYM1
  SoA(2) = SYM2
  SoA(3) = SYM3

  call symmetry_bd(2,ex,f,fh,SoA)

  Sdxdz = F1o4 /( dX * dZ )

  Fdxdz = F1o144 /( dX * dZ )

  fxz = ZEO

  do k=1,ex(3)-1
  do j=1,ex(2)-1
  do i=1,ex(1)-1
!~~~~~~ fxz
       if(i+2 <= imax .and. i-2 >= imin .and. k+2 <= kmax .and. k-2 >= kmin)then
   fxz(i,j,k) = Fdxdz*(     (fh(i-2,j,k-2)-F8*fh(i-1,j,k-2)+F8*fh(i+1,j,k-2)-fh(i+2,j,k-2))  &
                       -F8 *(fh(i-2,j,k-1)-F8*fh(i-1,j,k-1)+F8*fh(i+1,j,k-1)-fh(i+2,j,k-1))  &
                       +F8 *(fh(i-2,j,k+1)-F8*fh(i-1,j,k+1)+F8*fh(i+1,j,k+1)-fh(i+2,j,k+1))  &
                       -    (fh(i-2,j,k+2)-F8*fh(i-1,j,k+2)+F8*fh(i+1,j,k+2)-fh(i+2,j,k+2)))
   elseif(i+1 <= imax .and. i-1 >= imin .and. k+1 <= kmax .and. k-1 >= kmin)then
   fxz(i,j,k) = Sdxdz*(fh(i-1,j,k-1)-fh(i+1,j,k-1)-fh(i-1,j,k+1)+fh(i+1,j,k+1))
   endif

   enddo
   enddo
   enddo

  return

  end subroutine fddxz

  subroutine fddyz(ex,f,fyz,X,Y,Z,SYM1,SYM2,SYM3,symmetry)
  implicit none

  integer,                             intent(in ):: ex(1:3),symmetry
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ):: f
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out):: fyz
  real*8,                              intent(in ):: X(ex(1)),Y(ex(2)),Z(ex(3)),SYM1,SYM2,SYM3

!~~~~~~ other variables
  real*8 :: dX,dY,dZ
  real*8,dimension(-1:ex(1),-1:ex(2),-1:ex(3))   :: fh
  real*8, dimension(3) :: SoA
  integer :: imin,jmin,kmin,imax,jmax,kmax,i,j,k
  real*8  :: Sdydz,Fdydz
  integer, parameter :: NO_SYMM = 0, EQ_SYMM = 1, OCTANT = 2
  real*8, parameter :: ZEO=0.d0, ONE=1.d0, TWO=2.d0, F1o4=2.5d-1, F9=9.d0,  F45=4.5d1
  real*8, parameter :: F8=8.d0, F16=1.6d1, F30=3.d1, F27=2.7d1, F270=2.7d2, F490=4.9d2
  real*8, parameter :: F1o6=ONE/6.d0, F1o12=ONE/1.2d1, F1o144=ONE/1.44d2
  real*8, parameter :: F1o180=ONE/1.8d2,F1o3600=ONE/3.6d3

  dX = X(2)-X(1)
  dY = Y(2)-Y(1)
  dZ = Z(2)-Z(1)

  imax = ex(1)
  jmax = ex(2)
  kmax = ex(3)

  imin = 1
  jmin = 1
  kmin = 1
  if(Symmetry > NO_SYMM .and. dabs(Z(1)) < dZ) kmin = -1
  if(Symmetry > EQ_SYMM .and. dabs(X(1)) < dX) imin = -1
  if(Symmetry > EQ_SYMM .and. dabs(Y(1)) < dY) jmin = -1

  SoA(1) = SYM1
  SoA(2) = SYM2
  SoA(3) = SYM3

  call symmetry_bd(2,ex,f,fh,SoA)

  Sdydz = F1o4 /( dY * dZ )

  Fdydz = F1o144 /( dY * dZ )

  fyz = ZEO

  do k=1,ex(3)-1
  do j=1,ex(2)-1
  do i=1,ex(1)-1
!~~~~~~ fyz
       if(j+2 <= jmax .and. j-2 >= jmin .and. k+2 <= kmax .and. k-2 >= kmin)then
   fyz(i,j,k) = Fdydz*(     (fh(i,j-2,k-2)-F8*fh(i,j-1,k-2)+F8*fh(i,j+1,k-2)-fh(i,j+2,k-2))  &
                       -F8 *(fh(i,j-2,k-1)-F8*fh(i,j-1,k-1)+F8*fh(i,j+1,k-1)-fh(i,j+2,k-1))  &
                       +F8 *(fh(i,j-2,k+1)-F8*fh(i,j-1,k+1)+F8*fh(i,j+1,k+1)-fh(i,j+2,k+1))  &
                       -    (fh(i,j-2,k+2)-F8*fh(i,j-1,k+2)+F8*fh(i,j+1,k+2)-fh(i,j+2,k+2)))
   elseif(j+1 <= jmax .and. j-1 >= jmin .and. k+1 <= kmax .and. k-1 >= kmin)then
   fyz(i,j,k) = Sdydz*(fh(i,j-1,k-1)-fh(i,j+1,k-1)-fh(i,j-1,k+1)+fh(i,j+1,k+1))
   endif 

   enddo
   enddo
   enddo

  return

  end subroutine fddyz

