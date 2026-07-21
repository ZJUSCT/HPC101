
!-----------------------------------------------------------------------------
! ADM quantites for surface intergral
!-----------------------------------------------------------------------------
  subroutine admmass_bssn(ex, X, Y, Z,                            &
               chi    ,   trK, &
               dxx    ,   gxy    ,   gxz    ,   dyy    ,   gyz    ,   dzz    , &
               Axx    ,   Axy    ,   Axz    ,   Ayy    ,   Ayz    ,   Azz    , &
               Gamx   ,  Gamy   ,  Gamz   ,  &
               massx,massy,massz, symmetry)

  implicit none
 !~~~~~~= Input parameters:
 
  integer,intent(in) :: ex(1:3),symmetry
  real*8, intent(in ):: X(1:ex(1)),Y(1:ex(2)),Z(1:ex(3))
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ) :: dxx,gxy,gxz,dyy,gyz,dzz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ) :: Axx,Axy,Axz,Ayy,Ayz,Azz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ) :: chi,trK
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ) :: Gamx,Gamy,Gamz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: massx,massy,massz
! local variables
  real*8, dimension(ex(1),ex(2),ex(3)) :: gxx,gyy,gzz
!  inverse metric
  real*8, dimension(ex(1),ex(2),ex(3)) :: gupxx,gupxy,gupxz
  real*8, dimension(ex(1),ex(2),ex(3)) :: gupyy,gupyz,gupzz
!  partial derivative of chi, chi_i
  real*8, dimension(ex(1),ex(2),ex(3)) :: chix,chiy,chiz
  real*8, dimension(ex(1),ex(2),ex(3)) :: f
  real*8 :: PI, F1o2pi
  real*8, parameter :: ONE = 1.d0, F1o8 = 1.d0/8.d0
  real*8, parameter :: SYM = 1.D0, ANTI= - 1.D0
  real*8            :: dX, dY, dZ

  dX = X(2) - X(1)
  dY = Y(2) - Y(1)
  dZ = Z(2) - Z(1)

       PI = dacos( - ONE )
  F1o2pi = ONE / ( 2.d0 * PI )

  gxx = dxx + ONE
  gyy = dyy + ONE
  gzz = dzz + ONE

  gupzz =  gxx * gyy * gzz + gxy * gyz * gxz + gxz * gxy * gyz - &
           gxz * gyy * gxz - gxy * gxy * gzz - gxx * gyz * gyz
  gupxx =   ( gyy * gzz - gyz * gyz ) / gupzz
  gupxy = - ( gxy * gzz - gyz * gxz ) / gupzz
  gupxz =   ( gxy * gyz - gyy * gxz ) / gupzz
  gupyy =   ( gxx * gzz - gxz * gxz ) / gupzz
  gupyz = - ( gxx * gyz - gxy * gxz ) / gupzz
  gupzz =   ( gxx * gyy - gxy * gxy ) / gupzz

  call fderivs(ex,chi,chix,chiy,chiz,X,Y,Z,SYM,SYM,SYM,Symmetry,0)

  f=1/4.d0/(chi+ONE)**1.25d0
! mass_i = (Gami/8 + gupij*phi_j/(4*chi^1.25))/(2*Pi)
  massx = (F1o8*Gamx + f*(gupxx*chix+gupxy*chiy+gupxz*chiz))*F1o2pi
  massy = (F1o8*Gamy + f*(gupxy*chix+gupyy*chiy+gupyz*chiz))*F1o2pi
  massz = (F1o8*Gamz + f*(gupxz*chix+gupyz*chiy+gupzz*chiz))*F1o2pi

  return

  end subroutine admmass_bssn
!-----------------------------------------------------------------------------------------------
! P^i = int r^j p_ji
!-----------------------------------------------------------------------------------------------
  subroutine admmomentum_bssn(ex,                            &
               chi, trK, &
               dxx    ,   gxy    ,   gxz    ,   dyy    ,   gyz    ,   dzz    , &
               Axx    ,   Axy    ,   Axz    ,   Ayy    ,   Ayz    ,   Azz    , &
               Gamx   ,  Gamy   ,  Gamz   ,  &
               pxx,pxy,pxz,pyy,pyz,pzz)

  implicit none
 !~~~~~~= Input parameters:
 
  integer,intent(in) :: ex(1:3)
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ) :: dxx,gxy,gxz,dyy,gyz,dzz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ) :: Axx,Axy,Axz,Ayy,Ayz,Azz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ) :: chi,trK
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ) :: Gamx,Gamy,Gamz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: pxx,pxy,pxz,pyy,pyz,pzz
! local variables
  real*8, dimension(ex(1),ex(2),ex(3)) :: Kxx,Kxy,Kxz,Kyy,Kyz,Kzz
  real*8, dimension(ex(1),ex(2),ex(3)) :: gxx,gyy,gzz,chim4
  real*8 :: PI, F1o8pi
  real*8, parameter :: ONE = 1.d0, F1o3 = 1.d0/3.d0

       PI = acos( - ONE )
  F1o8pi = ONE / ( 8.d0 * PI )

  gxx = dxx + ONE
  gyy = dyy + ONE
  gzz = dzz + ONE

  chim4=1.d0/(chi+ONE)**4
  Kxx = chim4*(Axx+F1o3*gxx*trK)
  Kxy = chim4*(Axy+F1o3*gxy*trK)
  Kxz = chim4*(Axz+F1o3*gxz*trK)
  Kyy = chim4*(Ayy+F1o3*gyy*trK)
  Kyz = chim4*(Ayz+F1o3*gyz*trK)
  Kzz = chim4*(Azz+F1o3*gzz*trK)

  pxx = (Kxx-trK)*F1o8pi
  pxy = (Kxy    )*F1o8pi
  pxz = (Kxz    )*F1o8pi
  pyy = (Kyy-trK)*F1o8pi
  pyz = (Kyz    )*F1o8pi
  pzz = (Kzz-trK)*F1o8pi

  return

  end subroutine admmomentum_bssn
