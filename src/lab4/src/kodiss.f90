

#include "macrodef.fh"

! we need only distinguish different finite difference order
! Vertex or Cell is distinguished in routine symmetry_bd which locates in
! file "fmisc.f90"

! fourth order code

!---------------------------------------------------------------------------------------------
!usual type Kreiss-Oliger type numerical dissipation
!We support cell center only
! Note the notation D_+ and D_- [P240 of B. Gustafsson, H.-O. Kreiss, and J. Oliger, Time
! Dependent Problems and Difference Methods (Wiley, New York, 1995).]
! D_+ = (f(i+1) - f(i))/h
! D_- = (f(i) - f(i-1))/h
! then we have D_+D_- = D_-D_+
!              D_+^3D_-^3 = (D_+D_-)^3 =
!    f(i-3) - 6 f(i-2) + 15 f(i-1) - 20 f(i) + 15 f(i+1) - 6 f(i+2) + f(i+3)
! -----------------------------------------------------------------------------
!                                    dx^6
! this is for 4th order accurate finite difference scheme
!---------------------------------------------------------------------------------------------
subroutine kodis(ex,X,Y,Z,f,f_rhs,SoA,Symmetry,eps)

implicit none
! argument variables
integer,intent(in) :: Symmetry
integer,dimension(3),intent(in)::ex
real*8, dimension(1:3), intent(in) :: SoA
double precision,intent(in),dimension(ex(1))::X
double precision,intent(in),dimension(ex(2))::Y
double precision,intent(in),dimension(ex(3))::Z
double precision,intent(in),dimension(ex(1),ex(2),ex(3))::f
double precision,intent(inout),dimension(ex(1),ex(2),ex(3))::f_rhs
real*8,intent(in) :: eps
! local variables
real*8,dimension(-2:ex(1),-2:ex(2),-2:ex(3))   :: fh
integer :: imin,jmin,kmin,imax,jmax,kmax
integer :: i,j,k
real*8  :: dX,dY,dZ
real*8, parameter :: ONE=1.d0,SIX=6.d0,FIT=1.5d1,TWT=2.d1
real*8,parameter::cof=6.4d1   ! 2^6
integer, parameter :: NO_SYMM=0, OCTANT=2

!rhs_i = rhs_i + eps/dx/cof*(f_i-3 - 6*f_i-2 + 15*f_i-1 - 20*f_i + 15*f_i+1 - 6*f_i+2 + f_i+3)

  dX = X(2)-X(1)
  dY = Y(2)-Y(1)
  dZ = Z(2)-Z(1)
  
  imax = ex(1)
  jmax = ex(2)
  kmax = ex(3)

  imin = 1
  jmin = 1
  kmin = 1

  if(Symmetry > NO_SYMM .and. dabs(Z(1)) < dZ) kmin = -2
  if(Symmetry == OCTANT .and. dabs(X(1)) < dX) imin = -2
  if(Symmetry == OCTANT .and. dabs(Y(1)) < dY) jmin = -2

  call symmetry_bd(3,ex,f,fh,SoA)

  do k=1,ex(3)
  do j=1,ex(2)
  do i=1,ex(1)

  if(i-3 >= imin .and. i+3 <= imax .and. &
     j-3 >= jmin .and. j+3 <= jmax .and. &
     k-3 >= kmin .and. k+3 <= kmax) then
#if 0     
! x direction
   f_rhs(i,j,k)       = f_rhs(i,j,k) + eps/dX/cof * (     &
                              (fh(i-3,j,k)+fh(i+3,j,k)) - &
                          SIX*(fh(i-2,j,k)+fh(i+2,j,k)) + &
                          FIT*(fh(i-1,j,k)+fh(i+1,j,k)) - &
                          TWT* fh(i,j,k)            )
! y direction

   f_rhs(i,j,k)       = f_rhs(i,j,k) + eps/dY/cof * (     &
                              (fh(i,j-3,k)+fh(i,j+3,k)) - &
                          SIX*(fh(i,j-2,k)+fh(i,j+2,k)) + &
                          FIT*(fh(i,j-1,k)+fh(i,j+1,k)) - &
                          TWT* fh(i,j,k)            )
! z direction

   f_rhs(i,j,k)       = f_rhs(i,j,k) + eps/dZ/cof * (     &
                              (fh(i,j,k-3)+fh(i,j,k+3)) - &
                          SIX*(fh(i,j,k-2)+fh(i,j,k+2)) + &
                          FIT*(fh(i,j,k-1)+fh(i,j,k+1)) - &
                          TWT* fh(i,j,k)            )
#else
! calculation order if important ?
   f_rhs(i,j,k)       = f_rhs(i,j,k) + eps/cof *( (     &
                              (fh(i-3,j,k)+fh(i+3,j,k)) - &
                          SIX*(fh(i-2,j,k)+fh(i+2,j,k)) + &
                          FIT*(fh(i-1,j,k)+fh(i+1,j,k)) - &
                          TWT* fh(i,j,k)            )/dX + &
                                                  (     &
                              (fh(i,j-3,k)+fh(i,j+3,k)) - &
                          SIX*(fh(i,j-2,k)+fh(i,j+2,k)) + &
                          FIT*(fh(i,j-1,k)+fh(i,j+1,k)) - &
                          TWT* fh(i,j,k)            )/dY + &
                                                  (     &
                              (fh(i,j,k-3)+fh(i,j,k+3)) - &
                          SIX*(fh(i,j,k-2)+fh(i,j,k+2)) + &
                          FIT*(fh(i,j,k-1)+fh(i,j,k+1)) - &
                          TWT* fh(i,j,k)            )/dZ )
#endif
  endif

  enddo
  enddo
  enddo

  return

  end subroutine kodis