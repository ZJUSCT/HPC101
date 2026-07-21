!-----------------------------------------------------------------------------
! $Id: rungekutta4_rout.f90,v 1.6 2012/12/26 11:47:43 zjcao Exp $
! Carry out 4th-order Runge-Kutta method
!-----------------------------------------------------------------------------
! rk4 for scalar
  subroutine rungekutta4_scalar(dT,f0,f1,f_rhs,RK4)

  implicit none

!~~~~~~% Input parameters:

  integer ,intent(in):: RK4
  real*8  ,intent(in):: dT,f0
  real*8  ,intent(inout):: f1,f_rhs

!~~~~~~% Local parameter

  real*8, parameter :: F1o6=1.d0/6.d0, HLF=5.d-1, TWO=2.d0

  if( RK4 == 0 ) then

   f1 = f0 + HLF * dT * f_rhs

  elseif(RK4 == 1 ) then

   f_rhs = f_rhs + TWO * f1
   f1 = f0 + HLF * dT * f1

  elseif(RK4 == 2 ) then

   f_rhs = f_rhs + TWO * f1
   f1 = f0 +       dT * f1

  elseif( RK4 == 3 ) then
 
   f1 = f0 +F1o6 * dT *(f1 + f_rhs)

  else

   write(*,*) "rungekutta4_scalar: something is wrong in RK4 counting!!"
   stop

  endif

  return   

  end subroutine rungekutta4_scalar
!~~~~~~~~~~~~~~~~~~  
! rk4 for complex scalar
  subroutine rungekutta4_cplxscalar(dT,f0,f1,f_rhs,RK4)

  implicit none

!~~~~~~% Input parameters:

  integer ,intent(in):: RK4
  real*8  ,intent(in):: dT
  double complex  ,intent(in):: f0
  double complex  ,intent(inout):: f1,f_rhs

!~~~~~~% Local parameter

  real*8, parameter :: F1o6=1.d0/6.d0, HLF=5.d-1, TWO=2.d0

  if( RK4 == 0 ) then

   f1 = f0 + HLF * dT * f_rhs

  elseif(RK4 == 1 ) then

   f_rhs = f_rhs + TWO * f1
   f1 = f0 + HLF * dT * f1

  elseif(RK4 == 2 ) then

   f_rhs = f_rhs + TWO * f1
   f1 = f0 +       dT * f1

  elseif( RK4 == 3 ) then
 
   f1 = f0 +F1o6 * dT *(f1 + f_rhs)

  else

   write(*,*) "rungekutta4_cplxscalar: something is wrong in RK4 counting!!"
   stop

  endif

  return   

  end subroutine rungekutta4_cplxscalar
!~~~~~~~~~~~~~~~~~~  
  subroutine rungekutta4_rout(ex,dT,f0,f1,f_rhs,RK4)

  implicit none

!~~~~~~% Input parameters:

  integer ,intent(in):: ex(1:3),RK4
  real*8  ,intent(in):: dT
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in) ::f0
  real*8, dimension(ex(1),ex(2),ex(3)),intent(inout) ::f_rhs
  real*8, dimension(ex(1),ex(2),ex(3)),intent(inout) ::f1

!~~~~~~% Local parameter

  real*8, parameter :: F1o6=1.d0/6.d0, HLF=5.d-1, TWO=2.d0

  if( RK4 == 0 ) then

   f1 = f0 + HLF * dT * f_rhs

  elseif(RK4 == 1 ) then

   f_rhs = f_rhs + TWO * f1

   f1 = f0 + HLF * dT * f1

  elseif(RK4 == 2 ) then

   f_rhs = f_rhs + TWO * f1

   f1 = f0 +       dT * f1

  elseif( RK4 == 3 ) then
 
   f1 = f0 +F1o6 * dT *(f1 + f_rhs)

  else

   write(*,*) "rungekutta4_rout: something is wrong in RK4 counting!!"
   stop

  endif

  return   

  end subroutine rungekutta4_rout
!-----------------------------------------------------------------------------
! icn for scalar
  subroutine icn_scalar(dT,f0,f1,f_rhs,RK4)

  implicit none

!~~~~~~% Input parameters:

  integer ,intent(in):: RK4
  real*8  ,intent(in):: dT,f0
  real*8  ,intent(inout):: f1,f_rhs

!~~~~~~% Local parameter

  real*8, parameter :: HLF=5.d-1

  if( RK4 == 0 ) then

   f1 = f0 + dT * f_rhs

  else

   f1 = f0 + HLF * dT * (f1+f_rhs)

  endif

  return   

  end subroutine icn_scalar
!~~~~~~~~~~~~~~~~~~  
! icn for complex scalar
  subroutine icn_cplxscalar(dT,f0,f1,f_rhs,RK4)

  implicit none

!~~~~~~% Input parameters:

  integer ,intent(in):: RK4
  real*8  ,intent(in):: dT
  double complex  ,intent(in):: f0
  double complex  ,intent(inout):: f1,f_rhs

!~~~~~~% Local parameter

  real*8, parameter :: HLF=5.d-1

  if( RK4 == 0 ) then

   f1 = f0 + dT * f_rhs

  else

   f1 = f0 + HLF * dT * (f1+f_rhs)

  endif

  return   

  end subroutine icn_cplxscalar
!~~~~~~~~~~~~~~~~~~  
  subroutine icn_rout(ex,dT,f0,f1,f_rhs,RK4)

  implicit none

!~~~~~~% Input parameters:

  integer ,intent(in):: ex(1:3),RK4
  real*8  ,intent(in):: dT
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in) ::f0
  real*8, dimension(ex(1),ex(2),ex(3)),intent(inout) ::f_rhs
  real*8, dimension(ex(1),ex(2),ex(3)),intent(inout) ::f1

!~~~~~~% Local parameter

  real*8, parameter :: HLF=5.d-1

  if( RK4 == 0 ) then

   f1 = f0 + dT * f_rhs

  else

   f1 = f0 + HLF * dT * (f1+f_rhs)

  endif

  return   

  end subroutine icn_rout
!~~~~~~~~~~~~~~~~~~  
  subroutine euler_rout(ex,dT,f0,f1,f_rhs)

  implicit none

!~~~~~~% Input parameters:

  integer ,intent(in):: ex(1:3)
  real*8  ,intent(in):: dT
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in) ::f0
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in) ::f_rhs
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) ::f1

   f1 = f0 + dT * f_rhs

  return   

  end subroutine euler_rout
