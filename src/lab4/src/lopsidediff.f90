
! Compute advection terms in right hand sides of field equations

#include "macrodef.fh"

! we need only distinguish different finite difference order
! Vertex or Cell is distinguished in routine symmetry_bd which locates in
! file "fmisc.f90"

! fourth order code

!-----------------------------------------------------------------------------
!
! Compute advection terms in right hand sides of field equations
!         v
! D f = ------[ - 3f    - 10f  + 18f    - 6f     + f     ]
!  i     12dx       i-v      i      i+v     i+2v    i+3v
!
! where
!
!        i
!      |B |
! v = -----
!        i
!       B
!
!-----------------------------------------------------------------------------

subroutine lopsided(ex,X,Y,Z,f,f_rhs,Sfx,Sfy,Sfz,Symmetry,SoA)
  implicit none

!~~~~~~> Input parameters:

  integer, intent(in)  :: ex(1:3),Symmetry
  real*8,  intent(in)  :: X(1:ex(1)),Y(1:ex(2)),Z(1:ex(3))
  real*8,dimension(ex(1),ex(2),ex(3)),intent(in)   :: f,Sfx,Sfy,Sfz

  real*8,dimension(ex(1),ex(2),ex(3)),intent(inout):: f_rhs
  real*8,dimension(3),intent(in) ::SoA

!~~~~~~> local variables:
! note index -2,-1,0, so we have 3 extra points
  real*8,dimension(-2:ex(1),-2:ex(2),-2:ex(3))   :: fh
  integer :: imin,jmin,kmin,imax,jmax,kmax,i,j,k
  real*8 :: dX,dY,dZ
  real*8 :: d12dx,d12dy,d12dz,d2dx,d2dy,d2dz
  real*8,  parameter :: ZEO=0.d0,ONE=1.d0, F3=3.d0
  real*8,  parameter :: TWO=2.d0,F6=6.0d0,F18=1.8d1
  real*8,  parameter :: F12=1.2d1, F10=1.d1,EIT=8.d0
  integer, parameter :: NO_SYMM = 0, EQ_SYMM = 1, OCTANT = 2

  dX = X(2)-X(1)
  dY = Y(2)-Y(1)
  dZ = Z(2)-Z(1)

  d12dx = ONE/F12/dX
  d12dy = ONE/F12/dY
  d12dz = ONE/F12/dZ

  d2dx = ONE/TWO/dX
  d2dy = ONE/TWO/dY
  d2dz = ONE/TWO/dZ

  imax = ex(1)
  jmax = ex(2)
  kmax = ex(3)

  imin = 1
  jmin = 1
  kmin = 1
  if(Symmetry > NO_SYMM .and. dabs(Z(1)) < dZ) kmin = -2
  if(Symmetry > EQ_SYMM .and. dabs(X(1)) < dX) imin = -2
  if(Symmetry > EQ_SYMM .and. dabs(Y(1)) < dY) jmin = -2

  call symmetry_bd(3,ex,f,fh,SoA)

! upper bound set ex-1 only for efficiency, 
! the loop body will set ex 0 also
  do k=1,ex(3)-1
  do j=1,ex(2)-1
  do i=1,ex(1)-1
#if 0  
!! old code
! x direction   
    if(Sfx(i,j,k) >= ZEO .and. i+3 <= imax .and. i-1 >= imin)then
!         v
! D f = ------[ - 3f    - 10f  + 18f    - 6f     + f     ]
!  i     12dx       i-v      i      i+v     i+2v    i+3v
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                   &
                  Sfx(i,j,k)*d12dx*(-F3*fh(i-1,j,k)-F10*fh(i,j,k)+F18*fh(i+1,j,k) &
                                    -F6*fh(i+2,j,k)+    fh(i+3,j,k))

    elseif(Sfx(i,j,k) <= ZEO .and. i-3 >= imin .and. i+1 <= imax)then
     f_rhs(i,j,k)=f_rhs(i,j,k)-                                                   &
                  Sfx(i,j,k)*d12dx*(-F3*fh(i+1,j,k)-F10*fh(i,j,k)+F18*fh(i-1,j,k) &
                                    -F6*fh(i-2,j,k)+    fh(i-3,j,k))

     elseif(i+2 <= imax .and. i-2 >= imin)then
!
!              f(i-2) - 8 f(i-1) + 8 f(i+1) - f(i+2)
!  fx(i) = ---------------------------------------------
!                             12 dx
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                           &
                  Sfx(i,j,k)*d12dx*(fh(i-2,j,k)-EIT*fh(i-1,j,k)+EIT*fh(i+1,j,k)-fh(i+2,j,k))

     elseif(i+1 <= imax .and. i-1 >= imin)then
!
!              - f(i-1) + f(i+1)
!  fx(i) = --------------------------------
!                     2 dx
     f_rhs(i,j,k)=f_rhs(i,j,k) + Sfx(i,j,k)*d2dx*(-fh(i-1,j,k)+fh(i+1,j,k))

! set imax and imin 0
    endif

! y direction   
    if(Sfy(i,j,k) >= ZEO .and. j+3 <= jmax .and. j-1 >= jmin)then
!         v
! D f = ------[ - 3f    - 10f  + 18f    - 6f     + f     ]
!  i     12dx       i-v      i      i+v     i+2v    i+3v
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                   &
                  Sfy(i,j,k)*d12dy*(-F3*fh(i,j-1,k)-F10*fh(i,j,k)+F18*fh(i,j+1,k) &
                                    -F6*fh(i,j+2,k)+    fh(i,j+3,k))

    elseif(Sfy(i,j,k) <= ZEO .and. j-3 >= jmin .and. j+1 <= jmax)then
     f_rhs(i,j,k)=f_rhs(i,j,k)-                                                   &
                  Sfy(i,j,k)*d12dy*(-F3*fh(i,j+1,k)-F10*fh(i,j,k)+F18*fh(i,j-1,k) &
                                    -F6*fh(i,j-2,k)+    fh(i,j-3,k))

     elseif(j+2 <= jmax .and. j-2 >= jmin)then

     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                            &
                  Sfy(i,j,k)*d12dy*(fh(i,j-2,k)-EIT*fh(i,j-1,k)+EIT*fh(i,j+1,k)-fh(i,j+2,k))

     elseif(j+1 <= jmax .and. j-1 >= jmin)then

     f_rhs(i,j,k)=f_rhs(i,j,k) + Sfy(i,j,k)*d2dy*(-fh(i,j-1,k)+fh(i,j+1,k))
! set jmin and jmax 0
     endif
!! z direction   
    if(Sfz(i,j,k) >= ZEO .and. k+3 <= kmax .and. k-1 >= kmin)then
!         v
! D f = ------[ - 3f    - 10f  + 18f    - 6f     + f     ]
!  i     12dx       i-v      i      i+v     i+2v    i+3v
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                   &
                  Sfz(i,j,k)*d12dz*(-F3*fh(i,j,k-1)-F10*fh(i,j,k)+F18*fh(i,j,k+1) &
                                    -F6*fh(i,j,k+2)+    fh(i,j,k+3))

    elseif(Sfz(i,j,k) <= ZEO .and. k-3 >= kmin .and. k+1 <= kmax)then
     f_rhs(i,j,k)=f_rhs(i,j,k)-                                                   &
                  Sfz(i,j,k)*d12dz*(-F3*fh(i,j,k+1)-F10*fh(i,j,k)+F18*fh(i,j,k-1) &
                                    -F6*fh(i,j,k-2)+    fh(i,j,k-3))

     elseif(k+2 <= kmax .and. k-2 >= kmin)then

     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                            &
                  Sfz(i,j,k)*d12dz*(fh(i,j,k-2)-EIT*fh(i,j,k-1)+EIT*fh(i,j,k+1)-fh(i,j,k+2))

     elseif(k+1 <= kmax .and. k-1 >= kmin)then

     f_rhs(i,j,k)=f_rhs(i,j,k)+Sfz(i,j,k)*d2dz*(-fh(i,j,k-1)+fh(i,j,k+1))
! set kmin and kmax 0
     endif
#else
!! new code, 2012dec27, based on bam
! x direction   
    if(Sfx(i,j,k) > ZEO)then
      if(i+3 <= imax)then
!         v
! D f = ------[ - 3f    - 10f  + 18f    - 6f     + f     ]
!  i     12dx       i-v      i      i+v     i+2v    i+3v
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                   &
                  Sfx(i,j,k)*d12dx*(-F3*fh(i-1,j,k)-F10*fh(i,j,k)+F18*fh(i+1,j,k) &
                                    -F6*fh(i+2,j,k)+    fh(i+3,j,k))
     elseif(i+2 <= imax)then
!
!              f(i-2) - 8 f(i-1) + 8 f(i+1) - f(i+2)
!  fx(i) = ---------------------------------------------
!                             12 dx
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                           &
                  Sfx(i,j,k)*d12dx*(fh(i-2,j,k)-EIT*fh(i-1,j,k)+EIT*fh(i+1,j,k)-fh(i+2,j,k))

     elseif(i+1 <= imax)then
!         v
! D f = ------[   3f    + 10f  - 18f    + 6f     - f     ]
!  i     12dx       i+v      i      i-v     i-2v    i-3v
     f_rhs(i,j,k)=f_rhs(i,j,k)-                                                   &
                  Sfx(i,j,k)*d12dx*(-F3*fh(i+1,j,k)-F10*fh(i,j,k)+F18*fh(i-1,j,k) &
                                    -F6*fh(i-2,j,k)+    fh(i-3,j,k))
! set imax and imin 0
     endif
   elseif(Sfx(i,j,k) < ZEO)then
      if(i-3 >= imin)then
!         v
! D f = ------[ - 3f    - 10f  + 18f    - 6f     + f     ]
!  i     12dx       i-v      i      i+v     i+2v    i+3v
     f_rhs(i,j,k)=f_rhs(i,j,k)-                                                   &
                  Sfx(i,j,k)*d12dx*(-F3*fh(i+1,j,k)-F10*fh(i,j,k)+F18*fh(i-1,j,k) &
                                    -F6*fh(i-2,j,k)+    fh(i-3,j,k))
     elseif(i-2 >= imin)then
!
!              f(i-2) - 8 f(i-1) + 8 f(i+1) - f(i+2)
!  fx(i) = ---------------------------------------------
!                             12 dx
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                           &
                  Sfx(i,j,k)*d12dx*(fh(i-2,j,k)-EIT*fh(i-1,j,k)+EIT*fh(i+1,j,k)-fh(i+2,j,k))

     elseif(i-1 >= imin)then
!         v
! D f = ------[   3f    + 10f  - 18f    + 6f     - f     ]
!  i     12dx       i+v      i      i-v     i-2v    i-3v
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                   &
                  Sfx(i,j,k)*d12dx*(-F3*fh(i-1,j,k)-F10*fh(i,j,k)+F18*fh(i+1,j,k) &
                                    -F6*fh(i+2,j,k)+    fh(i+3,j,k))
! set imax and imin 0
     endif
   endif

! y direction   
    if(Sfy(i,j,k) > ZEO)then
      if(j+3 <= jmax)then
!         v
! D f = ------[ - 3f    - 10f  + 18f    - 6f     + f     ]
!  i     12dx       i-v      i      i+v     i+2v    i+3v
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                   &
                  Sfy(i,j,k)*d12dy*(-F3*fh(i,j-1,k)-F10*fh(i,j,k)+F18*fh(i,j+1,k) &
                                    -F6*fh(i,j+2,k)+    fh(i,j+3,k))
     elseif(j+2 <= jmax)then
!
!              f(i-2) - 8 f(i-1) + 8 f(i+1) - f(i+2)
!  fx(i) = ---------------------------------------------
!                             12 dx
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                           &
                  Sfy(i,j,k)*d12dy*(fh(i,j-2,k)-EIT*fh(i,j-1,k)+EIT*fh(i,j+1,k)-fh(i,j+2,k))

     elseif(j+1 <= jmax)then
!         v
! D f = ------[   3f    + 10f  - 18f    + 6f     - f     ]
!  i     12dx       i+v      i      i-v     i-2v    i-3v
     f_rhs(i,j,k)=f_rhs(i,j,k)-                                                   &
                  Sfy(i,j,k)*d12dy*(-F3*fh(i,j+1,k)-F10*fh(i,j,k)+F18*fh(i,j-1,k) &
                                    -F6*fh(i,j-2,k)+    fh(i,j-3,k))
! set imax and imin 0
     endif
   elseif(Sfy(i,j,k) < ZEO)then
      if(j-3 >= jmin)then
!         v
! D f = ------[ - 3f    - 10f  + 18f    - 6f     + f     ]
!  i     12dx       i-v      i      i+v     i+2v    i+3v
     f_rhs(i,j,k)=f_rhs(i,j,k)-                                                   &
                  Sfy(i,j,k)*d12dy*(-F3*fh(i,j+1,k)-F10*fh(i,j,k)+F18*fh(i,j-1,k) &
                                    -F6*fh(i,j-2,k)+    fh(i,j-3,k))
     elseif(j-2 >= jmin)then
!
!              f(i-2) - 8 f(i-1) + 8 f(i+1) - f(i+2)
!  fx(i) = ---------------------------------------------
!                             12 dx
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                           &
                  Sfy(i,j,k)*d12dy*(fh(i,j-2,k)-EIT*fh(i,j-1,k)+EIT*fh(i,j+1,k)-fh(i,j+2,k))

     elseif(j-1 >= jmin)then
!         v
! D f = ------[   3f    + 10f  - 18f    + 6f     - f     ]
!  i     12dx       i+v      i      i-v     i-2v    i-3v
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                   &
                  Sfy(i,j,k)*d12dy*(-F3*fh(i,j-1,k)-F10*fh(i,j,k)+F18*fh(i,j+1,k) &
                                    -F6*fh(i,j+2,k)+    fh(i,j+3,k))
! set jmax and jmin 0
     endif
   endif

! z direction   
    if(Sfz(i,j,k) > ZEO)then
      if(k+3 <= kmax)then
!         v
! D f = ------[ - 3f    - 10f  + 18f    - 6f     + f     ]
!  i     12dx       i-v      i      i+v     i+2v    i+3v
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                   &
                  Sfz(i,j,k)*d12dz*(-F3*fh(i,j,k-1)-F10*fh(i,j,k)+F18*fh(i,j,k+1) &
                                    -F6*fh(i,j,k+2)+    fh(i,j,k+3))
     elseif(k+2 <= kmax)then
!
!              f(i-2) - 8 f(i-1) + 8 f(i+1) - f(i+2)
!  fx(i) = ---------------------------------------------
!                             12 dx
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                           &
                  Sfz(i,j,k)*d12dz*(fh(i,j,k-2)-EIT*fh(i,j,k-1)+EIT*fh(i,j,k+1)-fh(i,j,k+2))

     elseif(k+1 <= kmax)then
!         v
! D f = ------[   3f    + 10f  - 18f    + 6f     - f     ]
!  i     12dx       i+v      i      i-v     i-2v    i-3v
     f_rhs(i,j,k)=f_rhs(i,j,k)-                                                   &
                  Sfz(i,j,k)*d12dz*(-F3*fh(i,j,k+1)-F10*fh(i,j,k)+F18*fh(i,j,k-1) &
                                    -F6*fh(i,j,k-2)+    fh(i,j,k-3))
! set imax and imin 0
     endif
   elseif(Sfz(i,j,k) < ZEO)then
      if(k-3 >= kmin)then
!         v
! D f = ------[ - 3f    - 10f  + 18f    - 6f     + f     ]
!  i     12dx       i-v      i      i+v     i+2v    i+3v
     f_rhs(i,j,k)=f_rhs(i,j,k)-                                                   &
                  Sfz(i,j,k)*d12dz*(-F3*fh(i,j,k+1)-F10*fh(i,j,k)+F18*fh(i,j,k-1) &
                                    -F6*fh(i,j,k-2)+    fh(i,j,k-3))
     elseif(k-2 >= kmin)then
!
!              f(i-2) - 8 f(i-1) + 8 f(i+1) - f(i+2)
!  fx(i) = ---------------------------------------------
!                             12 dx
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                           &
                  Sfz(i,j,k)*d12dz*(fh(i,j,k-2)-EIT*fh(i,j,k-1)+EIT*fh(i,j,k+1)-fh(i,j,k+2))

     elseif(k-1 >= kmin)then
!         v
! D f = ------[   3f    + 10f  - 18f    + 6f     - f     ]
!  i     12dx       i+v      i      i-v     i-2v    i-3v
     f_rhs(i,j,k)=f_rhs(i,j,k)+                                                   &
                  Sfz(i,j,k)*d12dz*(-F3*fh(i,j,k-1)-F10*fh(i,j,k)+F18*fh(i,j,k+1) &
                                    -F6*fh(i,j,k+2)+    fh(i,j,k+3))
! set kmax and kmin 0
     endif
   endif
#endif
  enddo
  enddo
  enddo

  return

  end subroutine lopsided

