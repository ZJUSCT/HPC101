
!-------------------------------------------------------------  
! kerrschild for schwarzschild   
!-------------------------------------------------------------
subroutine get_initial_kerrschild(ex,XX,YY,ZZ,&
            chi,trK,&
            dxx,gxy,gxz,dyy,gyz,dzz,&
            Axx,Axy,Axz,Ayy,Ayz,Azz,&
            Gmx,Gmy,Gmz,&
            Lap,Sfx,Sfy,Sfz,dtSfx,dtSfy,dtSfz)
implicit none
! argument variables
integer, intent(in ):: ex(1:3)
real*8, intent(in ):: XX(1:ex(1)),YY(1:ex(2)),ZZ(1:ex(3))
real*8,dimension(ex(1),ex(2),ex(3)),intent(out)::chi,trK,dxx,gxy,gxz,dyy,gyz,dzz
real*8,dimension(ex(1),ex(2),ex(3)),intent(out)::Axx,Axy,Axz,Ayy,Ayz,Azz
real*8,dimension(ex(1),ex(2),ex(3)),intent(out)::Gmx,Gmy,Gmz,Lap,Sfx,Sfy,Sfz,dtSfx,dtSfy,dtSfz

integer :: i,j,k
real*8 ::x,y,z
real*8,parameter :: M = 1.d0,ZEO=0.d0,mF1o3=-1.d0/3.d0

do i=1,ex(1)
  x = XX(i)
do j=1,ex(2)
  y = YY(j)
do k=1,ex(3)
  z = ZZ(k)
  chi(i,j,k) = ((sqrt(x**2+y**2+z**2)+2*M)/sqrt(x**2+y**2+z**2))**mF1o3 - 1.d0

  trK(i,j,k) = 2*(sqrt(x**2+y**2+z**2)+3*M)*M/(x**2+y**2+z**2)/(sqrt(x**2+y**2+z**2)+2*M)&
              /sqrt((sqrt(x**2+y**2+z**2)+2*M)/sqrt(x**2+y**2+z**2))

  dxx(i,j,k) = (sqrt(x**2+y**2+z**2)*x**2+sqrt(x**2+y**2+z**2)*y**2+sqrt(x**2+y**2+z**2)*z**2&
               +2*x**2*M)/((sqrt(x**2+y**2+z**2)+2*M)/sqrt(x**2+y**2+z**2))**(1.D0/3.D0)/&
               sqrt(x**2+y**2+z**2)**3 - 1.0
  gxy(i,j,k) = 2/((sqrt(x**2+y**2+z**2)+2*M)/sqrt(x**2+y**2+z**2))**(1.D0/3.D0)*M*x*y/&
               sqrt(x**2+y**2+z**2)**3
  gxz(i,j,k) = 2/((sqrt(x**2+y**2+z**2)+2*M)/sqrt(x**2+y**2+z**2))**(1.D0/3.D0)*M*x*z/&
               sqrt(x**2+y**2+z**2)**3
  dyy(i,j,k) = (sqrt(x**2+y**2+z**2)*x**2+sqrt(x**2+y**2+z**2)*y**2+sqrt(x**2+y**2+z**2)&
               *z**2+2*M*y**2)/((sqrt(x**2+y**2+z**2)+2*M)/sqrt(x**2+y**2+z**2))**(1.D0/3.D0)&
               /sqrt(x**2+y**2+z**2)**3 - 1.0
  gyz(i,j,k) = 2/((sqrt(x**2+y**2+z**2)+2*M)/sqrt(x**2+y**2+z**2))**(1.D0/3.D0)*M*y*z/&
               sqrt(x**2+y**2+z**2)**3
  dzz(i,j,k) = (sqrt(x**2+y**2+z**2)*x**2+sqrt(x**2+y**2+z**2)*y**2+sqrt(x**2+y**2+z**2)*&
               z**2+2*M*z**2)/((sqrt(x**2+y**2+z**2)+2*M)/sqrt(x**2+y**2+z**2))**(1.D0/3.D0)&
               /sqrt(x**2+y**2+z**2)**3 - 1.0
  Axx(i,j,k) = -2.D0/3.D0*M*(4*x**4+2*x**2*y**2+12*x**2*M**2+2*x**2*z**2+14*x**2*sqrt(x**2+y**2+z**2)&
               *M-4*y**2*z**2-2*z**4-2*y**4-3*sqrt(x**2+y**2+z**2)*M*y**2-3*sqrt(x**2+y**2+z**2)*M*z**2)&
               /sqrt(x**2+y**2+z**2)**5/(sqrt(x**2+y**2+z**2)+2*M)/((sqrt(x**2+y**2+z**2)+2*M)/&
               sqrt(x**2+y**2+z**2))**(5.D0/6.D0)
  Axy(i,j,k) = -2.D0/3.D0*M*x*y*(6*x**2+12*M**2+6*z**2+6*y**2+17*sqrt(x**2+y**2+z**2)*M)/&
               sqrt(x**2+y**2+z**2)**5/(sqrt(x**2+y**2+z**2)+2*M)/((sqrt(x**2+y**2+z**2)+2*M)/&
               sqrt(x**2+y**2+z**2))**(5.D0/6.D0)
  Axz(i,j,k) = -2.D0/3.D0*z*M*x*(6*x**2+12*M**2+6*z**2+6*y**2+17*M*sqrt(x**2+y**2+z**2))/&
               sqrt(x**2+y**2+z**2)**5/(sqrt(x**2+y**2+z**2)+2*M)/((sqrt(x**2+y**2+z**2)+2*M)/&
               sqrt(x**2+y**2+z**2))**(5.D0/6.D0)
  Ayy(i,j,k) = 2.D0/3.D0*M*(2*x**4+3*x**2*sqrt(x**2+y**2+z**2)*M-2*x**2*y**2+4*x**2*z**2-2*y**2*z**2&
               +2*z**4-4*y**4+3*sqrt(x**2+y**2+z**2)*M*z**2-14*sqrt(x**2+y**2+z**2)*M*y**2-12*M**2*y**2)&
               /sqrt(x**2+y**2+z**2)**5/(sqrt(x**2+y**2+z**2)+2*M)/((sqrt(x**2+y**2+z**2)+2*M)/&
               sqrt(x**2+y**2+z**2))**(5.D0/6.D0)
  Ayz(i,j,k) = -2.D0/3.D0*z*y*M*(6*x**2+6*z**2+17*sqrt(x**2+y**2+z**2)*M+12*M**2+6*y**2)/&
               sqrt(x**2+y**2+z**2)**5/(sqrt(x**2+y**2+z**2)+2*M)/((sqrt(x**2+y**2+z**2)+2*M)/&
               sqrt(x**2+y**2+z**2))**(5.D0/6.D0)
  Azz(i,j,k) = 2.D0/3.D0*M*(2*x**4-2*x**2*z**2+4*x**2*y**2+3*x**2*sqrt(x**2+y**2+z**2)*M- &
               2*y**2*z**2-4*z**4+2*y**4-12*M**2*z**2-14*sqrt(x**2+y**2+z**2)*M*z**2+3*&
               sqrt(x**2+y**2+z**2)*M*y**2)/sqrt(x**2+y**2+z**2)**5/(sqrt(x**2+y**2+z**2)+2*M)/&
               ((sqrt(x**2+y**2+z**2)+2*M)/sqrt(x**2+y**2+z**2))**(5.D0/6.D0)
  Gmx(i,j,k) = 8.D0/3.D0*x*M*(x**2+6*M**2+z**2+5*M*sqrt(x**2+y**2+z**2)+y**2)/(sqrt(x**2+y**2+z**2) &
               +2*M)**2/sqrt(x**2+y**2+z**2)**3/((sqrt(x**2+y**2+z**2)+2*M)/sqrt(x**2+y**2+z**2))**(2.D0/3.D0)
  Gmy(i,j,k) = 8.D0/3.D0*M*y*(x**2+6*M**2+z**2+y**2+5*M*sqrt(x**2+y**2+z**2))/(sqrt(x**2+y**2+z**2)+2*M)**2/&
               sqrt(x**2+y**2+z**2)**3/((sqrt(x**2+y**2+z**2)+2*M)/sqrt(x**2+y**2+z**2))**(2.D0/3.D0)
  Gmz(i,j,k) = 8.D0/3.D0*M*z*(x**2+z**2+5*M*sqrt(x**2+y**2+z**2)+y**2+6*M**2)/(sqrt(x**2+y**2+z**2)+2*M)**2/&
               sqrt(x**2+y**2+z**2)**3/((sqrt(x**2+y**2+z**2)+2*M)/sqrt(x**2+y**2+z**2))**(2.D0/3.D0)
  Lap(i,j,k) = sqrt(sqrt(x**2+y**2+z**2)/(sqrt(x**2+y**2+z**2)+2*M)) - 1.0
  Sfx(i,j,k) = 2/sqrt(x**2+y**2+z**2)*x*M/(sqrt(x**2+y**2+z**2)+2*M)
  Sfy(i,j,k) = 2/sqrt(x**2+y**2+z**2)*M*y/(sqrt(x**2+y**2+z**2)+2*M)
  Sfz(i,j,k) = 2/sqrt(x**2+y**2+z**2)*z*M/(sqrt(x**2+y**2+z**2)+2*M)

enddo
enddo
enddo
dtSfx = ZEO
dtSfy = ZEO
dtSfz = ZEO

return

end subroutine get_initial_kerrschild
!for shell
subroutine get_initial_kerrschild_ss(ex,XX,YY,ZZ,&
            chi,trK,&
            dxx,gxy,gxz,dyy,gyz,dzz,&
            Axx,Axy,Axz,Ayy,Ayz,Azz,&
            Gmx,Gmy,Gmz,&
            Lap,Sfx,Sfy,Sfz,dtSfx,dtSfy,dtSfz)
implicit none
! argument variables
integer, intent(in ):: ex(1:3)
real*8,dimension(ex(1),ex(2),ex(3)),intent(in ):: XX,YY,ZZ
real*8,dimension(ex(1),ex(2),ex(3)),intent(out)::chi,trK,dxx,gxy,gxz,dyy,gyz,dzz
real*8,dimension(ex(1),ex(2),ex(3)),intent(out)::Axx,Axy,Axz,Ayy,Ayz,Azz
real*8,dimension(ex(1),ex(2),ex(3)),intent(out)::Gmx,Gmy,Gmz,Lap,Sfx,Sfy,Sfz,dtSfx,dtSfy,dtSfz

integer :: i,j,k
real*8 ::x,y,z
real*8,parameter :: M = 1.d0,ZEO=0.d0,mF1o3=-1.d0/3.d0

do i=1,ex(1)
do j=1,ex(2)
do k=1,ex(3)
  x = XX(i,j,k)
  y = YY(i,j,k)
  z = ZZ(i,j,k)
  chi(i,j,k) = ((sqrt(x**2+y**2+z**2)+2*M)/sqrt(x**2+y**2+z**2))**mF1o3 - 1.d0

  trK(i,j,k) = 2*(sqrt(x**2+y**2+z**2)+3*M)*M/(x**2+y**2+z**2)/(sqrt(x**2+y**2+z**2)+2*M)&
              /sqrt((sqrt(x**2+y**2+z**2)+2*M)/sqrt(x**2+y**2+z**2))

  dxx(i,j,k) = (sqrt(x**2+y**2+z**2)*x**2+sqrt(x**2+y**2+z**2)*y**2+sqrt(x**2+y**2+z**2)*z**2&
               +2*x**2*M)/((sqrt(x**2+y**2+z**2)+2*M)/sqrt(x**2+y**2+z**2))**(1.D0/3.D0)/&
               sqrt(x**2+y**2+z**2)**3 - 1.0
  gxy(i,j,k) = 2/((sqrt(x**2+y**2+z**2)+2*M)/sqrt(x**2+y**2+z**2))**(1.D0/3.D0)*M*x*y/&
               sqrt(x**2+y**2+z**2)**3
  gxz(i,j,k) = 2/((sqrt(x**2+y**2+z**2)+2*M)/sqrt(x**2+y**2+z**2))**(1.D0/3.D0)*M*x*z/&
               sqrt(x**2+y**2+z**2)**3
  dyy(i,j,k) = (sqrt(x**2+y**2+z**2)*x**2+sqrt(x**2+y**2+z**2)*y**2+sqrt(x**2+y**2+z**2)&
               *z**2+2*M*y**2)/((sqrt(x**2+y**2+z**2)+2*M)/sqrt(x**2+y**2+z**2))**(1.D0/3.D0)&
               /sqrt(x**2+y**2+z**2)**3 - 1.0
  gyz(i,j,k) = 2/((sqrt(x**2+y**2+z**2)+2*M)/sqrt(x**2+y**2+z**2))**(1.D0/3.D0)*M*y*z/&
               sqrt(x**2+y**2+z**2)**3
  dzz(i,j,k) = (sqrt(x**2+y**2+z**2)*x**2+sqrt(x**2+y**2+z**2)*y**2+sqrt(x**2+y**2+z**2)*&
               z**2+2*M*z**2)/((sqrt(x**2+y**2+z**2)+2*M)/sqrt(x**2+y**2+z**2))**(1.D0/3.D0)&
               /sqrt(x**2+y**2+z**2)**3 - 1.0
  Axx(i,j,k) = -2.D0/3.D0*M*(4*x**4+2*x**2*y**2+12*x**2*M**2+2*x**2*z**2+14*x**2*sqrt(x**2+y**2+z**2)&
               *M-4*y**2*z**2-2*z**4-2*y**4-3*sqrt(x**2+y**2+z**2)*M*y**2-3*sqrt(x**2+y**2+z**2)*M*z**2)&
               /sqrt(x**2+y**2+z**2)**5/(sqrt(x**2+y**2+z**2)+2*M)/((sqrt(x**2+y**2+z**2)+2*M)/&
               sqrt(x**2+y**2+z**2))**(5.D0/6.D0)
  Axy(i,j,k) = -2.D0/3.D0*M*x*y*(6*x**2+12*M**2+6*z**2+6*y**2+17*sqrt(x**2+y**2+z**2)*M)/&
               sqrt(x**2+y**2+z**2)**5/(sqrt(x**2+y**2+z**2)+2*M)/((sqrt(x**2+y**2+z**2)+2*M)/&
               sqrt(x**2+y**2+z**2))**(5.D0/6.D0)
  Axz(i,j,k) = -2.D0/3.D0*z*M*x*(6*x**2+12*M**2+6*z**2+6*y**2+17*M*sqrt(x**2+y**2+z**2))/&
               sqrt(x**2+y**2+z**2)**5/(sqrt(x**2+y**2+z**2)+2*M)/((sqrt(x**2+y**2+z**2)+2*M)/&
               sqrt(x**2+y**2+z**2))**(5.D0/6.D0)
  Ayy(i,j,k) = 2.D0/3.D0*M*(2*x**4+3*x**2*sqrt(x**2+y**2+z**2)*M-2*x**2*y**2+4*x**2*z**2-2*y**2*z**2&
               +2*z**4-4*y**4+3*sqrt(x**2+y**2+z**2)*M*z**2-14*sqrt(x**2+y**2+z**2)*M*y**2-12*M**2*y**2)&
               /sqrt(x**2+y**2+z**2)**5/(sqrt(x**2+y**2+z**2)+2*M)/((sqrt(x**2+y**2+z**2)+2*M)/&
               sqrt(x**2+y**2+z**2))**(5.D0/6.D0)
  Ayz(i,j,k) = -2.D0/3.D0*z*y*M*(6*x**2+6*z**2+17*sqrt(x**2+y**2+z**2)*M+12*M**2+6*y**2)/&
               sqrt(x**2+y**2+z**2)**5/(sqrt(x**2+y**2+z**2)+2*M)/((sqrt(x**2+y**2+z**2)+2*M)/&
               sqrt(x**2+y**2+z**2))**(5.D0/6.D0)
  Azz(i,j,k) = 2.D0/3.D0*M*(2*x**4-2*x**2*z**2+4*x**2*y**2+3*x**2*sqrt(x**2+y**2+z**2)*M- &
               2*y**2*z**2-4*z**4+2*y**4-12*M**2*z**2-14*sqrt(x**2+y**2+z**2)*M*z**2+3*&
               sqrt(x**2+y**2+z**2)*M*y**2)/sqrt(x**2+y**2+z**2)**5/(sqrt(x**2+y**2+z**2)+2*M)/&
               ((sqrt(x**2+y**2+z**2)+2*M)/sqrt(x**2+y**2+z**2))**(5.D0/6.D0)
  Gmx(i,j,k) = 8.D0/3.D0*x*M*(x**2+6*M**2+z**2+5*M*sqrt(x**2+y**2+z**2)+y**2)/(sqrt(x**2+y**2+z**2) &
               +2*M)**2/sqrt(x**2+y**2+z**2)**3/((sqrt(x**2+y**2+z**2)+2*M)/sqrt(x**2+y**2+z**2))**(2.D0/3.D0)
  Gmy(i,j,k) = 8.D0/3.D0*M*y*(x**2+6*M**2+z**2+y**2+5*M*sqrt(x**2+y**2+z**2))/(sqrt(x**2+y**2+z**2)+2*M)**2/&
               sqrt(x**2+y**2+z**2)**3/((sqrt(x**2+y**2+z**2)+2*M)/sqrt(x**2+y**2+z**2))**(2.D0/3.D0)
  Gmz(i,j,k) = 8.D0/3.D0*M*z*(x**2+z**2+5*M*sqrt(x**2+y**2+z**2)+y**2+6*M**2)/(sqrt(x**2+y**2+z**2)+2*M)**2/&
               sqrt(x**2+y**2+z**2)**3/((sqrt(x**2+y**2+z**2)+2*M)/sqrt(x**2+y**2+z**2))**(2.D0/3.D0)
  Lap(i,j,k) = sqrt(sqrt(x**2+y**2+z**2)/(sqrt(x**2+y**2+z**2)+2*M)) - 1.0
  Sfx(i,j,k) = 2/sqrt(x**2+y**2+z**2)*x*M/(sqrt(x**2+y**2+z**2)+2*M)
  Sfy(i,j,k) = 2/sqrt(x**2+y**2+z**2)*M*y/(sqrt(x**2+y**2+z**2)+2*M)
  Sfz(i,j,k) = 2/sqrt(x**2+y**2+z**2)*z*M/(sqrt(x**2+y**2+z**2)+2*M)

enddo
enddo
enddo
dtSfx = ZEO
dtSfy = ZEO
dtSfz = ZEO

return

end subroutine get_initial_kerrschild_ss
!-----------------------------------------------------------------------------------
!
!Set up approximate puncture initial data for single black hole with small P and
!S, my own formula
! 
!-----------------------------------------------------------------------------------

  subroutine get_initial_bssn3(ex,X,Y,Z, &
                         chi,  trK, &
                         gxx,  gxy,  gxz,  gyy,  gyz,  gzz,&
                         Axx,  Axy,  Axz,  Ayy,  Ayz,  Azz,&
                         Gmx,  Gmy,  Gmz,  &
                         Lap,  Sfx,  Sfy,  Sfz,&
                         dtSfx,dtSfy,dtSfz,M,Porg,Pmom,Spin)

  implicit none

!------= input arguments
 
  integer, dimension(3),  intent(in) :: ex
  real*8,  dimension(ex(1)), intent(in) :: X
  real*8,  dimension(ex(2)), intent(in) :: Y
  real*8,  dimension(ex(3)), intent(in) :: Z
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: chi
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: gxx,gxy,gxz,gyy,gyz,gzz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: Axx,Axy,Axz,Ayy,Ayz,Azz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: trK,Lap,Sfx,Sfy,Sfz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: Gmx,Gmy,Gmz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: dtSfx,dtSfy,dtSfz
  real*8,  intent(in) :: M,Porg(3),Pmom(3),Spin(3)

!------= local variables
  real*8,dimension(ex(1),ex(2),ex(3))::psi
  integer :: i,j,k
  real*8 :: Px,Py,Pz,PP,Sx,Sy,Sz,SS
  real*8 :: nx,ny,nz,rr,tmp
  real*8 :: u,u1,u2,u3,u4
  real*8 :: mup,mus,b,ell
  real*8,  parameter :: HLF = 5.d-1, ZEO = 0.d0, ONE = 1.d0, THR = 3.d0

  Px = Pmom(1)
  Py = Pmom(2)
  Pz = Pmom(3)

  Sx = Spin(1)
  Sy = Spin(2)
  Sz = Spin(3)

  PP = dsqrt(Px**2 + Py**2 + Pz**2)
  SS = dsqrt(Sx**2 + Sy**2 + Sz**2)

  do k = 1,ex(3)
   do j = 1,ex(2)
    do i = 1,ex(1)

      nx = X(i)-Porg(1)
      ny = Y(j)-Porg(2)
      nz = Z(k)-Porg(3)
      rr = dsqrt(nx**2+ny**2+nz**2)
      b = 2.d0*rr/M
      ell = 1.d0/(1.d0+b)

      nx = nx / rr
      ny = ny / rr
      nz = nz / rr
      if(PP .gt. 0.d0) then
        mup = (Px*nx+Py*ny+Pz*nz)/PP
      else
        mup = 0.0
      endif
      if(SS .gt. 0.d0) then
        mus = (Sx*nx+Sy*ny+Sz*nz)/SS
      else
        mus = 0.0
      endif

     u1 = 5.d0/8.d0*ell*(1.d0-2.d0*ell+2.d0*ell**2-ell**3+ell**4/5.d0)       
     u2 = (1.5d1+1.17d2*ell-7.9d1*ell**2+4.3d1*ell**3-1.4d1*ell**4+2.d0*ell**5 &
         +8.4d1*dlog(ell)/b)/4.d1/b**2
     u3 = ell/2.d1*(1.d0+ell+ell**2-4.d0*ell**3+2.d0*ell**4)
     u4 = ell**2/1.d1*(1.d1-2.5d1*ell+2.1d1*ell**2-6.d0*ell**3)

     tmp = (Py*Sz-Pz*Sy)*nx + (Pz*Sx-Px*Sz)*ny + (Px*Sy-Py*Sx)*nz

     u = PP**2/M**2*(u1 + u2*(3.d0*mup**2-ONE)) + &
         6.d0*u3/M**4*SS**2*(1.d0+mus**2) + u4/M**3*tmp

     psi(i,j,k) = ONE + u + HLF*M/rr

     tmp = Px * nx + Py * ny + Pz * nz

     Axx(i,j,k) = (HLF *( Px * nx + nx * Px - ( ONE - nx * nx )* tmp ) + &
                   ( nx * Sy * nz - nx * Sz * ny + nx * Sy * nz - nx * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Ayy(i,j,k) = (HLF *( Py * ny + ny * Py - ( ONE - ny * ny )* tmp ) + &
                   ( ny * Sz * nx - ny * Sx * nz + ny * Sz * nx - ny * Sx * nz ) / rr ) * &
                   THR / ( rr * rr )

     Azz(i,j,k) = (HLF *( Pz * nz + nz * Pz - ( ONE - nz * nz )* tmp ) + &
                   ( nz * Sx * ny - nz * Sy * nx + nz * Sx * ny - nz * Sy * nx ) / rr ) * &
                   THR / ( rr * rr )

     Axy(i,j,k) = (HLF *( Px * ny + nx * Py + nx * ny * tmp ) + &
                   ( nx * Sz * nx - nx * Sx * nz + ny * Sy * nz - ny * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Axz(i,j,k) = (HLF *( Px * nz + nx * Pz + nx * nz * tmp ) + &
                   ( nx * Sx * ny - nx * Sy * nx + nz * Sy * nz - nz * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Ayz(i,j,k) = (HLF *( Py * nz + ny * Pz + ny * nz * tmp ) + &
                   ( ny * Sx * ny - ny * Sy * nx + nz * Sz * nx - nz * Sx * nz ) / rr ) * &
                   THR / ( rr * rr )
    enddo
   enddo
  enddo

  chi = ONE / psi **4 - ONE

  Lap = ONE / psi **2 - ONE

!~~~~~~ tilde Aij = Aij / Psi^6
  psi = psi * psi * psi * psi * psi * psi

  Axx = Axx / psi
  Ayy = Ayy / psi
  Azz = Azz / psi
  Axy = Axy / psi
  Axz = Axz / psi
  Ayz = Ayz / psi

  gxx = ZEO
  gyy = ZEO
  gzz = ZEO
  gxy = ZEO
  gxz = ZEO
  gyz = ZEO

  trK = ZEO

  Gmx = ZEO
  Gmy = ZEO
  Gmz = ZEO

  Sfx = ZEO
  Sfy = ZEO
  Sfz = ZEO

  dtSfx = ZEO
  dtSfy = ZEO
  dtSfz = ZEO
  
  return

  end subroutine get_initial_bssn3
!-----------------------------------------------------------------------------------
!
!Set up approximate puncture initial data for inspiral binary
!
!-----------------------------------------------------------------------------------

  subroutine get_initial_bssn6(ex,X,Y,Z, &
                         chi,  trK, &
                         gxx,  gxy,  gxz,  gyy,  gyz,  gzz,&
                         Axx,  Axy,  Axz,  Ayy,  Ayz,  Azz,&
                         Gmx,  Gmy,  Gmz,  &
                         Lap,  Sfx,  Sfy,  Sfz,&
                         dtSfx,dtSfy,dtSfz,Mass,Porg,Pmom,Spin)

  implicit none

!------= input arguments
 
  integer, dimension(3),  intent(in) :: ex
  real*8,  dimension(ex(1)), intent(in) :: X
  real*8,  dimension(ex(2)), intent(in) :: Y
  real*8,  dimension(ex(3)), intent(in) :: Z
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: chi
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: gxx,gxy,gxz,gyy,gyz,gzz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: Axx,Axy,Axz,Ayy,Ayz,Azz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: trK,Lap,Sfx,Sfy,Sfz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: Gmx,Gmy,Gmz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: dtSfx,dtSfy,dtSfz
  real*8,  dimension(2), intent(in) :: Mass
  real*8,  dimension(6), intent(in) :: Porg,Pmom,Spin

!------= local variables
  real*8,dimension(ex(1),ex(2),ex(3))::psi
  integer :: i,j,k
  real*8 :: M,Px,Py,Pz,PP,Sx,Sy,Sz,SS
  real*8 :: nx,ny,nz,rr,tmp
  real*8 :: u,u1,u2,u3,u4
  real*8 :: mup,mus,b,ell
  real*8,  parameter :: HLF = 5.d-1, ZEO = 0.d0, ONE = 1.d0, THR = 3.d0

  do k = 1,ex(3)
   do j = 1,ex(2)
    do i = 1,ex(1)
! black hole 1
      M = mass(1)
      nx = x(i) - Porg(1)
      ny = y(j) - Porg(2)
      nz = z(k) - Porg(3)
      Px = Pmom(1)
      Py = Pmom(2)
      Pz = Pmom(3)
      Sx = Spin(1)
      Sy = Spin(2)
      Sz = Spin(3)

      rr = dsqrt(nx*nx+ny*ny+nz*nz)
      nx = nx / rr
      ny = ny / rr
      nz = nz / rr
      PP = dsqrt(Px**2 + Py**2 + Pz**2)
      if(PP .gt. 0.d0) then
        mup = (Px*nx+Py*ny+Pz*nz)/PP
      else
        mup = 0.0
      endif
      SS = dsqrt(Sx**2 + Sy**2 + Sz**2)
      if(SS .gt. 0.d0) then
        mus = (Sx*nx+Sy*ny+Sz*nz)/SS
      else
        mus = 0.0
      endif
      b = 2.d0*rr/M
      ell = 1.d0/(1.d0+b)

     u1 = 5.d0/8.d0*ell*(1.d0-2.d0*ell+2.d0*ell**2-ell**3+ell**4/5.d0)       
     u2 = (1.5d1+1.17d2*ell-7.9d1*ell**2+4.3d1*ell**3-1.4d1*ell**4+2.d0*ell**5 &
         +8.4d1*dlog(ell)/b)/4.d1/b**2
     u3 = ell/2.d1*(1.d0+ell+ell**2-4.d0*ell**3+2.d0*ell**4)
     u4 = ell**2/1.d1*(1.d1-2.5d1*ell+2.1d1*ell**2-6.d0*ell**3)

     tmp = (Py*Sz-Pz*Sy)*nx + (Pz*Sx-Px*Sz)*ny + (Px*Sy-Py*Sx)*nz

     u = PP**2/M**2*(u1 + u2*(3.d0*mup**2-ONE)) + &
         6.d0*u3/M**4*SS**2*(1.d0+mus**2) + u4/M**3*tmp

     psi(i,j,k) = ONE + u + HLF*M/rr

     tmp = Px * nx + Py * ny + Pz * nz

     Axx(i,j,k) = (HLF *( Px * nx + nx * Px - ( ONE - nx * nx )* tmp ) + &
                   ( nx * Sy * nz - nx * Sz * ny + nx * Sy * nz - nx * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Ayy(i,j,k) = (HLF *( Py * ny + ny * Py - ( ONE - ny * ny )* tmp ) + &
                   ( ny * Sz * nx - ny * Sx * nz + ny * Sz * nx - ny * Sx * nz ) / rr ) * &
                   THR / ( rr * rr )

     Azz(i,j,k) = (HLF *( Pz * nz + nz * Pz - ( ONE - nz * nz )* tmp ) + &
                   ( nz * Sx * ny - nz * Sy * nx + nz * Sx * ny - nz * Sy * nx ) / rr ) * &
                   THR / ( rr * rr )

     Axy(i,j,k) = (HLF *( Px * ny + nx * Py + nx * ny * tmp ) + &
                   ( nx * Sz * nx - nx * Sx * nz + ny * Sy * nz - ny * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Axz(i,j,k) = (HLF *( Px * nz + nx * Pz + nx * nz * tmp ) + &
                   ( nx * Sx * ny - nx * Sy * nx + nz * Sy * nz - nz * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Ayz(i,j,k) = (HLF *( Py * nz + ny * Pz + ny * nz * tmp ) + &
                   ( ny * Sx * ny - ny * Sy * nx + nz * Sz * nx - nz * Sx * nz ) / rr ) * &
                   THR / ( rr * rr )
! black hole 2
      M = Mass(2)
      nx = x(i) - Porg(4)
      ny = y(j) - Porg(5)
      nz = z(k) - Porg(6)
      Px = Pmom(4)
      Py = Pmom(5)
      Pz = Pmom(6)
      Sx = Spin(4)
      Sy = Spin(5)
      Sz = Spin(6)

      rr = dsqrt(nx*nx+ny*ny+nz*nz)
      nx = nx / rr
      ny = ny / rr
      nz = nz / rr
      PP = dsqrt(Px**2 + Py**2 + Pz**2)
      if(PP .gt. 0.d0) then
        mup = (Px*nx+Py*ny+Pz*nz)/PP
      else
        mup = 0.0
      endif
      SS = dsqrt(Sx**2 + Sy**2 + Sz**2)
      if(SS .gt. 0.d0) then
        mus = (Sx*nx+Sy*ny+Sz*nz)/SS
      else
        mus = 0.0
      endif
      b = 2.d0*rr/M
      ell = 1.d0/(1.d0+b)

     u1 = 5.d0/8.d0*ell*(1.d0-2.d0*ell+2.d0*ell**2-ell**3+ell**4/5.d0)       
     u2 = (1.5d1+1.17d2*ell-7.9d1*ell**2+4.3d1*ell**3-1.4d1*ell**4+2.d0*ell**5 &
         +8.4d1*dlog(ell)/b)/4.d1/b**2
     u3 = ell/2.d1*(1.d0+ell+ell**2-4.d0*ell**3+2.d0*ell**4)
     u4 = ell**2/1.d1*(1.d1-2.5d1*ell+2.1d1*ell**2-6.d0*ell**3)

     tmp = (Py*Sz-Pz*Sy)*nx + (Pz*Sx-Px*Sz)*ny + (Px*Sy-Py*Sx)*nz

     u = PP**2/M**2*(u1 + u2*(3.d0*mup**2-ONE)) + &
         6.d0*u3/M**4*SS**2*(1.d0+mus**2) + u4/M**3*tmp

     psi(i,j,k) = psi(i,j,k) + u + HLF*M/rr

     tmp = Px * nx + Py * ny + Pz * nz

     Axx(i,j,k) = Axx(i,j,k) + &
                  (HLF *( Px * nx + nx * Px - ( ONE - nx * nx )* tmp ) + &
                  ( nx * Sy * nz - nx * Sz * ny + nx * Sy * nz - nx * Sz * ny ) / rr ) * &
                  THR / ( rr * rr )

     Ayy(i,j,k) = Ayy(i,j,k) + &
                  (HLF *( Py * ny + ny * Py - ( ONE - ny * ny )* tmp ) + &
                  ( ny * Sz * nx - ny * Sx * nz + ny * Sz * nx - ny * Sx * nz ) / rr ) * &
                  THR / ( rr * rr )

     Azz(i,j,k) = Azz(i,j,k) + &
                  (HLF *( Pz * nz + nz * Pz - ( ONE - nz * nz )* tmp ) + &
                  ( nz * Sx * ny - nz * Sy * nx + nz * Sx * ny - nz * Sy * nx ) / rr ) * &
                  THR / ( rr * rr )

     Axy(i,j,k) = Axy(i,j,k) + &
                  (HLF *( Px * ny + nx * Py + nx * ny * tmp ) + &
                  ( nx * Sz * nx - nx * Sx * nz + ny * Sy * nz - ny * Sz * ny ) / rr ) * &
                  THR / ( rr * rr )

     Axz(i,j,k) = Axz(i,j,k) + &
                  (HLF *( Px * nz + nx * Pz + nx * nz * tmp ) + &
                  ( nx * Sx * ny - nx * Sy * nx + nz * Sy * nz - nz * Sz * ny ) / rr ) * &
                  THR / ( rr * rr )

     Ayz(i,j,k) = Ayz(i,j,k) + &
                  (HLF *( Py * nz + ny * Pz + ny * nz * tmp ) + &
                  ( ny * Sx * ny - ny * Sy * nx + nz * Sz * nx - nz * Sx * nz ) / rr ) * &
                  THR / ( rr * rr )
    enddo
   enddo
  enddo

  chi = ONE / psi **4 - ONE

  Lap = ONE / ( psi * psi ) - ONE

!~~~~~~ tilde Aij = Aij / Psi^6
  psi = psi * psi * psi * psi * psi * psi

  Axx = Axx / psi
  Ayy = Ayy / psi
  Azz = Azz / psi
  Axy = Axy / psi
  Axz = Axz / psi
  Ayz = Ayz / psi

  gxx = ZEO
  gyy = ZEO
  gzz = ZEO
  gxy = ZEO
  gxz = ZEO
  gyz = ZEO

  trK = ZEO

  Gmx = ZEO
  Gmy = ZEO
  Gmz = ZEO

  Sfx = ZEO
  Sfy = ZEO
  Sfz = ZEO

  dtSfx = ZEO
  dtSfy = ZEO
  dtSfz = ZEO
  
  return

  end subroutine get_initial_bssn6
!-----------------------------------------------------------------------------------
!
!post deal the initial data after reading from file
!
!-----------------------------------------------------------------------------------
  subroutine get_initial_postdeal(ex,X,Y,Z, &
                         chi,  trK, &
                         dxx,  gxy,  gxz,  dyy,  gyz,  dzz,&
                         Axx,  Axy,  Axz,  Ayy,  Ayz,  Azz,&
                         Gmx,  Gmy,  Gmz,  &
                         Lap,  Sfx,  Sfy,  Sfz,&
                         dtSfx,dtSfy,dtSfz)

  implicit none

!------= input arguments
! for chi: input phi, output chi
  integer, dimension(3),  intent(in) :: ex
  real*8,  dimension(ex(1)), intent(in) :: X
  real*8,  dimension(ex(2)), intent(in) :: Y
  real*8,  dimension(ex(3)), intent(in) :: Z
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(inout) :: chi,trK
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(inout) :: dxx,gxy,gxz,dyy,gyz,dzz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(inout) :: Axx,Axy,Axz,Ayy,Ayz,Azz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(inout) :: Lap,Sfx,Sfy,Sfz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(inout) :: Gmx,Gmy,Gmz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(inout) :: dtSfx,dtSfy,dtSfz

!------= local variables

  real*8,parameter :: ZEO = 0.d0, ONE = 1.d0

! psi=exp(phi)  
  chi = dexp( chi )
! Lap=exp(-2*phi)  
  Lap = ONE / ( chi * chi ) - ONE
! chi=exp(-4*phi)  
  chi = ONE / chi **4 - ONE

  dxx = ZEO
  dyy = ZEO
  dzz = ZEO
  gxy = ZEO
  gxz = ZEO
  gyz = ZEO

  trK = ZEO

  Gmx = ZEO
  Gmy = ZEO
  Gmz = ZEO

  Sfx = ZEO
  Sfy = ZEO
  Sfz = ZEO

  dtSfx = ZEO
  dtSfy = ZEO
  dtSfz = ZEO
  
  return

  end subroutine get_initial_postdeal
!-----------------------------------------------------------------------------------
!
!Set up puncture initial data for single black hole with the given solution u by
!Ansorg
! 
!-----------------------------------------------------------------------------------

  subroutine get_ansorg_single(ex,X,Y,Z, &
                         chi,  trK, &
                         gxx,  gxy,  gxz,  gyy,  gyz,  gzz,&
                         Axx,  Axy,  Axz,  Ayy,  Ayz,  Azz,&
                         Gmx,  Gmy,  Gmz,  &
                         Lap,  Sfx,  Sfy,  Sfz,&
                         dtSfx,dtSfy,dtSfz,M,Porg,Pmom,Spin)

  implicit none

!------= input arguments
 
  integer, dimension(3),  intent(in) :: ex
  real*8,  dimension(ex(1)), intent(in) :: X
  real*8,  dimension(ex(2)), intent(in) :: Y
  real*8,  dimension(ex(3)), intent(in) :: Z
! in u, out chi
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(inout) :: chi
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: gxx,gxy,gxz,gyy,gyz,gzz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: Axx,Axy,Axz,Ayy,Ayz,Azz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: trK,Lap,Sfx,Sfy,Sfz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: Gmx,Gmy,Gmz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: dtSfx,dtSfy,dtSfz
  real*8,  intent(in) :: M,Porg(3),Pmom(3),Spin(3)

!------= local variables
  real*8,dimension(ex(1),ex(2),ex(3))::psi
  integer :: i,j,k
  real*8 :: Px,Py,Pz,Sx,Sy,Sz
  real*8 :: nx,ny,nz,rr,tmp
  real*8,  parameter :: HLF = 5.d-1, ZEO = 0.d0, ONE = 1.d0, THR = 3.d0
  real*8,parameter::TINYRR=1.d-14

  Px = Pmom(1)
  Py = Pmom(2)
  Pz = Pmom(3)

  Sx = Spin(1)
  Sy = Spin(2)
  Sz = Spin(3)

  do k = 1,ex(3)
   do j = 1,ex(2)
    do i = 1,ex(1)

      nx = X(i)-Porg(1)
      ny = Y(j)-Porg(2)
      nz = Z(k)-Porg(3)
      rr = dsqrt(nx**2+ny**2+nz**2)
      if(rr.lt.TINYRR) rr=(X(2)-X(1))/2.d0

      nx = nx / rr
      ny = ny / rr
      nz = nz / rr

     psi(i,j,k) = ONE + chi(i,j,k) + HLF*M/rr

     tmp = Px * nx + Py * ny + Pz * nz

     Axx(i,j,k) = (HLF *( Px * nx + nx * Px - ( ONE - nx * nx )* tmp ) + &
                   ( nx * Sy * nz - nx * Sz * ny + nx * Sy * nz - nx * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Ayy(i,j,k) = (HLF *( Py * ny + ny * Py - ( ONE - ny * ny )* tmp ) + &
                   ( ny * Sz * nx - ny * Sx * nz + ny * Sz * nx - ny * Sx * nz ) / rr ) * &
                   THR / ( rr * rr )

     Azz(i,j,k) = (HLF *( Pz * nz + nz * Pz - ( ONE - nz * nz )* tmp ) + &
                   ( nz * Sx * ny - nz * Sy * nx + nz * Sx * ny - nz * Sy * nx ) / rr ) * &
                   THR / ( rr * rr )

     Axy(i,j,k) = (HLF *( Px * ny + nx * Py + nx * ny * tmp ) + &
                   ( nx * Sz * nx - nx * Sx * nz + ny * Sy * nz - ny * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Axz(i,j,k) = (HLF *( Px * nz + nx * Pz + nx * nz * tmp ) + &
                   ( nx * Sx * ny - nx * Sy * nx + nz * Sy * nz - nz * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Ayz(i,j,k) = (HLF *( Py * nz + ny * Pz + ny * nz * tmp ) + &
                   ( ny * Sx * ny - ny * Sy * nx + nz * Sz * nx - nz * Sx * nz ) / rr ) * &
                   THR / ( rr * rr )
    enddo
   enddo
  enddo

  chi = ONE / psi **4 - ONE

  Lap = ONE / ( psi * psi ) - ONE

!~~~~~~ tilde Aij = Aij / Psi^6
  psi = psi * psi * psi * psi * psi * psi

  Axx = Axx / psi
  Ayy = Ayy / psi
  Azz = Azz / psi
  Axy = Axy / psi
  Axz = Axz / psi
  Ayz = Ayz / psi

  gxx = ZEO
  gyy = ZEO
  gzz = ZEO
  gxy = ZEO
  gxz = ZEO
  gyz = ZEO

  trK = ZEO

  Gmx = ZEO
  Gmy = ZEO
  Gmz = ZEO

  Sfx = ZEO
  Sfy = ZEO
  Sfz = ZEO

  dtSfx = ZEO
  dtSfy = ZEO
  dtSfz = ZEO
  
  return

  end subroutine get_ansorg_single
!-----------------------------------------------------------------------------------
!
!Set up puncture initial data for inspiral binary with the given solution u by
!Ansorg
!
!-----------------------------------------------------------------------------------

  subroutine get_ansorg_binary(ex,X,Y,Z, &
                         chi,  trK, &
                         gxx,  gxy,  gxz,  gyy,  gyz,  gzz,&
                         Axx,  Axy,  Axz,  Ayy,  Ayz,  Azz,&
                         Gmx,  Gmy,  Gmz,  &
                         Lap,  Sfx,  Sfy,  Sfz,&
                         dtSfx,dtSfy,dtSfz,Mass,Porg,Pmom,Spin)

  implicit none

!------= input arguments
 
  integer, dimension(3),  intent(in) :: ex
  real*8,  dimension(ex(1)), intent(in) :: X
  real*8,  dimension(ex(2)), intent(in) :: Y
  real*8,  dimension(ex(3)), intent(in) :: Z
! in u, out chi  
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(inout) :: chi
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: gxx,gxy,gxz,gyy,gyz,gzz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: Axx,Axy,Axz,Ayy,Ayz,Azz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: trK,Lap,Sfx,Sfy,Sfz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: Gmx,Gmy,Gmz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: dtSfx,dtSfy,dtSfz
  real*8,  dimension(2), intent(in) :: Mass
  real*8,  dimension(6), intent(in) :: Porg,Pmom,Spin

!------= local variables
  real*8,dimension(ex(1),ex(2),ex(3))::psi
  integer :: i,j,k
  real*8 :: M,Px,Py,Pz,Sx,Sy,Sz
  real*8 :: nx,ny,nz,rr,tmp
  real*8,  parameter :: HLF = 5.d-1, ZEO = 0.d0, ONE = 1.d0, THR = 3.d0
  real*8,parameter::TINYRR=1.d-14

  do k = 1,ex(3)
   do j = 1,ex(2)
    do i = 1,ex(1)
! black hole 1
      M = mass(1)
      nx = x(i) - Porg(1)
      ny = y(j) - Porg(2)
      nz = z(k) - Porg(3)
      Px = Pmom(1)
      Py = Pmom(2)
      Pz = Pmom(3)
      Sx = Spin(1)
      Sy = Spin(2)
      Sz = Spin(3)

      rr = dsqrt(nx*nx+ny*ny+nz*nz)
      if(rr.lt.TINYRR) rr=(X(2)-X(1))/2.d0
      nx = nx / rr
      ny = ny / rr
      nz = nz / rr

     psi(i,j,k) = ONE + chi(i,j,k) + HLF*M/rr

     tmp = Px * nx + Py * ny + Pz * nz

     Axx(i,j,k) = (HLF *( Px * nx + nx * Px - ( ONE - nx * nx )* tmp ) + &
                   ( nx * Sy * nz - nx * Sz * ny + nx * Sy * nz - nx * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Ayy(i,j,k) = (HLF *( Py * ny + ny * Py - ( ONE - ny * ny )* tmp ) + &
                   ( ny * Sz * nx - ny * Sx * nz + ny * Sz * nx - ny * Sx * nz ) / rr ) * &
                   THR / ( rr * rr )

     Azz(i,j,k) = (HLF *( Pz * nz + nz * Pz - ( ONE - nz * nz )* tmp ) + &
                   ( nz * Sx * ny - nz * Sy * nx + nz * Sx * ny - nz * Sy * nx ) / rr ) * &
                   THR / ( rr * rr )

     Axy(i,j,k) = (HLF *( Px * ny + nx * Py + nx * ny * tmp ) + &
                   ( nx * Sz * nx - nx * Sx * nz + ny * Sy * nz - ny * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Axz(i,j,k) = (HLF *( Px * nz + nx * Pz + nx * nz * tmp ) + &
                   ( nx * Sx * ny - nx * Sy * nx + nz * Sy * nz - nz * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Ayz(i,j,k) = (HLF *( Py * nz + ny * Pz + ny * nz * tmp ) + &
                   ( ny * Sx * ny - ny * Sy * nx + nz * Sz * nx - nz * Sx * nz ) / rr ) * &
                   THR / ( rr * rr )
! black hole 2
      M = Mass(2)
      nx = x(i) - Porg(4)
      ny = y(j) - Porg(5)
      nz = z(k) - Porg(6)
      Px = Pmom(4)
      Py = Pmom(5)
      Pz = Pmom(6)
      Sx = Spin(4)
      Sy = Spin(5)
      Sz = Spin(6)

      rr = dsqrt(nx*nx+ny*ny+nz*nz)
      if(rr.lt.TINYRR) rr=(X(2)-X(1))/2.d0
      nx = nx / rr
      ny = ny / rr
      nz = nz / rr

     psi(i,j,k) = psi(i,j,k) + HLF*M/rr

     tmp = Px * nx + Py * ny + Pz * nz

     Axx(i,j,k) = Axx(i,j,k) + &
                  (HLF *( Px * nx + nx * Px - ( ONE - nx * nx )* tmp ) + &
                  ( nx * Sy * nz - nx * Sz * ny + nx * Sy * nz - nx * Sz * ny ) / rr ) * &
                  THR / ( rr * rr )

     Ayy(i,j,k) = Ayy(i,j,k) + &
                  (HLF *( Py * ny + ny * Py - ( ONE - ny * ny )* tmp ) + &
                  ( ny * Sz * nx - ny * Sx * nz + ny * Sz * nx - ny * Sx * nz ) / rr ) * &
                  THR / ( rr * rr )

     Azz(i,j,k) = Azz(i,j,k) + &
                  (HLF *( Pz * nz + nz * Pz - ( ONE - nz * nz )* tmp ) + &
                  ( nz * Sx * ny - nz * Sy * nx + nz * Sx * ny - nz * Sy * nx ) / rr ) * &
                  THR / ( rr * rr )

     Axy(i,j,k) = Axy(i,j,k) + &
                  (HLF *( Px * ny + nx * Py + nx * ny * tmp ) + &
                  ( nx * Sz * nx - nx * Sx * nz + ny * Sy * nz - ny * Sz * ny ) / rr ) * &
                  THR / ( rr * rr )

     Axz(i,j,k) = Axz(i,j,k) + &
                  (HLF *( Px * nz + nx * Pz + nx * nz * tmp ) + &
                  ( nx * Sx * ny - nx * Sy * nx + nz * Sy * nz - nz * Sz * ny ) / rr ) * &
                  THR / ( rr * rr )

     Ayz(i,j,k) = Ayz(i,j,k) + &
                  (HLF *( Py * nz + ny * Pz + ny * nz * tmp ) + &
                  ( ny * Sx * ny - ny * Sy * nx + nz * Sz * nx - nz * Sx * nz ) / rr ) * &
                  THR / ( rr * rr )
    enddo
   enddo
  enddo

  chi = ONE / psi **4 - ONE

  Lap = ONE / ( psi * psi ) - ONE

!~~~~~~ tilde Aij = Aij / Psi^6
  psi = psi * psi * psi * psi * psi * psi

  Axx = Axx / psi
  Ayy = Ayy / psi
  Azz = Azz / psi
  Axy = Axy / psi
  Axz = Axz / psi
  Ayz = Ayz / psi

  gxx = ZEO
  gyy = ZEO
  gzz = ZEO
  gxy = ZEO
  gxz = ZEO
  gyz = ZEO

  trK = ZEO

  Gmx = ZEO
  Gmy = ZEO
  Gmz = ZEO

  Sfx = ZEO
  Sfy = ZEO
  Sfz = ZEO

  dtSfx = ZEO
  dtSfy = ZEO
  dtSfz = ZEO
  
  return

  end subroutine get_ansorg_binary

!-----------------------------------------------------------------------------------
!
!Set up puncture initial data for black hole system with the given solution u by
!Ansorg
! 
!-----------------------------------------------------------------------------------
  subroutine get_ansorg_nbhs(ex,X,Y,Z, &
                         chi,  trK, &
                         gxx,  gxy,  gxz,  gyy,  gyz,  gzz,&
                         Axx,  Axy,  Axz,  Ayy,  Ayz,  Azz,&
                         Gmx,  Gmy,  Gmz,  &
                         Lap,  Sfx,  Sfy,  Sfz,&
                         dtSfx,dtSfy,dtSfz,    &
                         Mass,Porg,Pmom,Spin,N)

  implicit none

!------= input arguments
 
  integer,intent(in) :: N
  integer, dimension(3),  intent(in) :: ex
  real*8,  dimension(ex(1)), intent(in) :: X
  real*8,  dimension(ex(2)), intent(in) :: Y
  real*8,  dimension(ex(3)), intent(in) :: Z
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(inout) :: chi
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: gxx,gxy,gxz,gyy,gyz,gzz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: Axx,Axy,Axz,Ayy,Ayz,Azz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: trK,Lap,Sfx,Sfy,Sfz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: Gmx,Gmy,Gmz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: dtSfx,dtSfy,dtSfz
  real*8,  dimension(N), intent(in) :: Mass
  real*8,  dimension(3*N), intent(in) :: Porg,Pmom,Spin

!------= local variables
  real*8,dimension(ex(1),ex(2),ex(3))::psi
  integer :: i,j,k,bhi
  real*8 :: M,Px,Py,Pz,Sx,Sy,Sz
  real*8 :: nx,ny,nz,rr,tmp
  real*8,  parameter :: HLF = 5.d-1, ZEO = 0.d0, ONE = 1.d0, THR = 3.d0
  real*8,parameter :: TINYRR=1.d-14
  real*8,parameter :: phi0=1.d0,r0=120.d0,sigma=8.d0

  do k = 1,ex(3)
   do j = 1,ex(2)
    do i = 1,ex(1)
! black hole 1
      M = mass(1)
      nx = x(i) - Porg(1)
      ny = y(j) - Porg(2)
      nz = z(k) - Porg(3)
      Px = Pmom(1)
      Py = Pmom(2)
      Pz = Pmom(3)
      Sx = Spin(1)
      Sy = Spin(2)
      Sz = Spin(3)

      rr = dsqrt(nx*nx+ny*ny+nz*nz)
      if(rr.lt.TINYRR) rr=(X(2)-X(1))/2.d0
      nx = nx / rr
      ny = ny / rr
      nz = nz / rr

     psi(i,j,k) = ONE + HLF*M/rr

     tmp = Px * nx + Py * ny + Pz * nz

     Axx(i,j,k) = (HLF *( Px * nx + nx * Px - ( ONE - nx * nx )* tmp ) + &
                   ( nx * Sy * nz - nx * Sz * ny + nx * Sy * nz - nx * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Ayy(i,j,k) = (HLF *( Py * ny + ny * Py - ( ONE - ny * ny )* tmp ) + &
                   ( ny * Sz * nx - ny * Sx * nz + ny * Sz * nx - ny * Sx * nz ) / rr ) * &
                   THR / ( rr * rr )

     Azz(i,j,k) = (HLF *( Pz * nz + nz * Pz - ( ONE - nz * nz )* tmp ) + &
                   ( nz * Sx * ny - nz * Sy * nx + nz * Sx * ny - nz * Sy * nx ) / rr ) * &
                   THR / ( rr * rr )

     Axy(i,j,k) = (HLF *( Px * ny + nx * Py + nx * ny * tmp ) + &
                   ( nx * Sz * nx - nx * Sx * nz + ny * Sy * nz - ny * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Axz(i,j,k) = (HLF *( Px * nz + nx * Pz + nx * nz * tmp ) + &
                   ( nx * Sx * ny - nx * Sy * nx + nz * Sy * nz - nz * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Ayz(i,j,k) = (HLF *( Py * nz + ny * Pz + ny * nz * tmp ) + &
                   ( ny * Sx * ny - ny * Sy * nx + nz * Sz * nx - nz * Sx * nz ) / rr ) * &
                   THR / ( rr * rr )
! black hole 2 and 3, ...
     do bhi=2,N
      M = Mass(bhi)
      nx = x(i) - Porg(3*(bhi-1)+1)
      ny = y(j) - Porg(3*(bhi-1)+2)
      nz = z(k) - Porg(3*(bhi-1)+3)
      Px = Pmom(3*(bhi-1)+1)
      Py = Pmom(3*(bhi-1)+2)
      Pz = Pmom(3*(bhi-1)+3)
      Sx = Spin(3*(bhi-1)+1)
      Sy = Spin(3*(bhi-1)+2)
      Sz = Spin(3*(bhi-1)+3)

      rr = dsqrt(nx*nx+ny*ny+nz*nz)
      if(rr.lt.TINYRR) rr=(X(2)-X(1))/2.d0
      nx = nx / rr
      ny = ny / rr
      nz = nz / rr

     psi(i,j,k) = psi(i,j,k) + HLF*M/rr

     tmp = Px * nx + Py * ny + Pz * nz

     Axx(i,j,k) = Axx(i,j,k) + &
                  (HLF *( Px * nx + nx * Px - ( ONE - nx * nx )* tmp ) + &
                  ( nx * Sy * nz - nx * Sz * ny + nx * Sy * nz - nx * Sz * ny ) / rr ) * &
                  THR / ( rr * rr )

     Ayy(i,j,k) = Ayy(i,j,k) + &
                  (HLF *( Py * ny + ny * Py - ( ONE - ny * ny )* tmp ) + &
                  ( ny * Sz * nx - ny * Sx * nz + ny * Sz * nx - ny * Sx * nz ) / rr ) * &
                  THR / ( rr * rr )

     Azz(i,j,k) = Azz(i,j,k) + &
                  (HLF *( Pz * nz + nz * Pz - ( ONE - nz * nz )* tmp ) + &
                  ( nz * Sx * ny - nz * Sy * nx + nz * Sx * ny - nz * Sy * nx ) / rr ) * &
                  THR / ( rr * rr )

     Axy(i,j,k) = Axy(i,j,k) + &
                  (HLF *( Px * ny + nx * Py + nx * ny * tmp ) + &
                  ( nx * Sz * nx - nx * Sx * nz + ny * Sy * nz - ny * Sz * ny ) / rr ) * &
                  THR / ( rr * rr )

     Axz(i,j,k) = Axz(i,j,k) + &
                  (HLF *( Px * nz + nx * Pz + nx * nz * tmp ) + &
                  ( nx * Sx * ny - nx * Sy * nx + nz * Sy * nz - nz * Sz * ny ) / rr ) * &
                  THR / ( rr * rr )

     Ayz(i,j,k) = Ayz(i,j,k) + &
                  (HLF *( Py * nz + ny * Pz + ny * nz * tmp ) + &
                  ( ny * Sx * ny - ny * Sy * nx + nz * Sz * nx - nz * Sx * nz ) / rr ) * &
                  THR / ( rr * rr )
      enddo
    enddo
   enddo
  enddo

  psi = chi + psi

  chi = ONE / psi **4 - ONE

  Lap = ONE / ( psi * psi ) - ONE

!~~~~~~ tilde Aij = Aij / Psi^6
  psi = psi * psi * psi * psi * psi * psi

  Axx = Axx / psi
  Ayy = Ayy / psi
  Azz = Azz / psi
  Axy = Axy / psi
  Axz = Axz / psi
  Ayz = Ayz / psi

  gxx = ZEO
  gyy = ZEO
  gzz = ZEO
  gxy = ZEO
  gxz = ZEO
  gyz = ZEO

  trK = ZEO

  Gmx = ZEO
  Gmy = ZEO
  Gmz = ZEO

  Sfx = ZEO
  Sfy = ZEO
  Sfz = ZEO

  dtSfx = ZEO
  dtSfy = ZEO
  dtSfz = ZEO
  
  return

  end subroutine get_ansorg_nbhs
!-----------------------------------------------------------------------------------
!
!Set up puncture initial data for black hole system with the given solution u by
!Ansorg
! for shell part
!-----------------------------------------------------------------------------------
  subroutine get_ansorg_nbhs_ss(ex,X,Y,Z, &
                         chi,  trK, &
                         gxx,  gxy,  gxz,  gyy,  gyz,  gzz,&
                         Axx,  Axy,  Axz,  Ayy,  Ayz,  Azz,&
                         Gmx,  Gmy,  Gmz,  &
                         Lap,  Sfx,  Sfy,  Sfz,&
                         dtSfx,dtSfy,dtSfz,    &
                         Mass,Porg,Pmom,Spin,N)

  implicit none

!------= input arguments
 
  integer,intent(in) :: N
  integer, dimension(3),  intent(in) :: ex
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(in) :: X,Y,Z
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(inout) :: chi
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: gxx,gxy,gxz,gyy,gyz,gzz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: Axx,Axy,Axz,Ayy,Ayz,Azz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: trK,Lap,Sfx,Sfy,Sfz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: Gmx,Gmy,Gmz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: dtSfx,dtSfy,dtSfz
  real*8,  dimension(N), intent(in) :: Mass
  real*8,  dimension(3*N), intent(in) :: Porg,Pmom,Spin

!------= local variables
  real*8,dimension(ex(1),ex(2),ex(3))::psi
  integer :: i,j,k,bhi
  real*8 :: M,Px,Py,Pz,Sx,Sy,Sz
  real*8 :: nx,ny,nz,rr,tmp
  real*8,  parameter :: HLF = 5.d-1, ZEO = 0.d0, ONE = 1.d0, THR = 3.d0
  real*8,parameter :: TINYRR=1.d-14

  do k = 1,ex(3)
   do j = 1,ex(2)
    do i = 1,ex(1)
! black hole 1
      M = mass(1)
      nx = x(i,j,k) - Porg(1)
      ny = y(i,j,k) - Porg(2)
      nz = z(i,j,k) - Porg(3)
      Px = Pmom(1)
      Py = Pmom(2)
      Pz = Pmom(3)
      Sx = Spin(1)
      Sy = Spin(2)
      Sz = Spin(3)

      rr = dsqrt(nx*nx+ny*ny+nz*nz)
      if(rr.lt.TINYRR) rr=TINYRR
      nx = nx / rr
      ny = ny / rr
      nz = nz / rr

     psi(i,j,k) = ONE + HLF*M/rr

     tmp = Px * nx + Py * ny + Pz * nz

     Axx(i,j,k) = (HLF *( Px * nx + nx * Px - ( ONE - nx * nx )* tmp ) + &
                   ( nx * Sy * nz - nx * Sz * ny + nx * Sy * nz - nx * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Ayy(i,j,k) = (HLF *( Py * ny + ny * Py - ( ONE - ny * ny )* tmp ) + &
                   ( ny * Sz * nx - ny * Sx * nz + ny * Sz * nx - ny * Sx * nz ) / rr ) * &
                   THR / ( rr * rr )

     Azz(i,j,k) = (HLF *( Pz * nz + nz * Pz - ( ONE - nz * nz )* tmp ) + &
                   ( nz * Sx * ny - nz * Sy * nx + nz * Sx * ny - nz * Sy * nx ) / rr ) * &
                   THR / ( rr * rr )

     Axy(i,j,k) = (HLF *( Px * ny + nx * Py + nx * ny * tmp ) + &
                   ( nx * Sz * nx - nx * Sx * nz + ny * Sy * nz - ny * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Axz(i,j,k) = (HLF *( Px * nz + nx * Pz + nx * nz * tmp ) + &
                   ( nx * Sx * ny - nx * Sy * nx + nz * Sy * nz - nz * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Ayz(i,j,k) = (HLF *( Py * nz + ny * Pz + ny * nz * tmp ) + &
                   ( ny * Sx * ny - ny * Sy * nx + nz * Sz * nx - nz * Sx * nz ) / rr ) * &
                   THR / ( rr * rr )
! black hole 2 and 3, ...
     do bhi=2,N
      M = Mass(bhi)
      nx = x(i,j,k) - Porg(3*(bhi-1)+1)
      ny = y(i,j,k) - Porg(3*(bhi-1)+2)
      nz = z(i,j,k) - Porg(3*(bhi-1)+3)
      Px = Pmom(3*(bhi-1)+1)
      Py = Pmom(3*(bhi-1)+2)
      Pz = Pmom(3*(bhi-1)+3)
      Sx = Spin(3*(bhi-1)+1)
      Sy = Spin(3*(bhi-1)+2)
      Sz = Spin(3*(bhi-1)+3)

      rr = dsqrt(nx*nx+ny*ny+nz*nz)
      if(rr.lt.TINYRR) rr=TINYRR
      nx = nx / rr
      ny = ny / rr
      nz = nz / rr

     psi(i,j,k) = psi(i,j,k) + HLF*M/rr

     tmp = Px * nx + Py * ny + Pz * nz

     Axx(i,j,k) = Axx(i,j,k) + &
                  (HLF *( Px * nx + nx * Px - ( ONE - nx * nx )* tmp ) + &
                  ( nx * Sy * nz - nx * Sz * ny + nx * Sy * nz - nx * Sz * ny ) / rr ) * &
                  THR / ( rr * rr )

     Ayy(i,j,k) = Ayy(i,j,k) + &
                  (HLF *( Py * ny + ny * Py - ( ONE - ny * ny )* tmp ) + &
                  ( ny * Sz * nx - ny * Sx * nz + ny * Sz * nx - ny * Sx * nz ) / rr ) * &
                  THR / ( rr * rr )

     Azz(i,j,k) = Azz(i,j,k) + &
                  (HLF *( Pz * nz + nz * Pz - ( ONE - nz * nz )* tmp ) + &
                  ( nz * Sx * ny - nz * Sy * nx + nz * Sx * ny - nz * Sy * nx ) / rr ) * &
                  THR / ( rr * rr )

     Axy(i,j,k) = Axy(i,j,k) + &
                  (HLF *( Px * ny + nx * Py + nx * ny * tmp ) + &
                  ( nx * Sz * nx - nx * Sx * nz + ny * Sy * nz - ny * Sz * ny ) / rr ) * &
                  THR / ( rr * rr )

     Axz(i,j,k) = Axz(i,j,k) + &
                  (HLF *( Px * nz + nx * Pz + nx * nz * tmp ) + &
                  ( nx * Sx * ny - nx * Sy * nx + nz * Sy * nz - nz * Sz * ny ) / rr ) * &
                  THR / ( rr * rr )

     Ayz(i,j,k) = Ayz(i,j,k) + &
                  (HLF *( Py * nz + ny * Pz + ny * nz * tmp ) + &
                  ( ny * Sx * ny - ny * Sy * nx + nz * Sz * nx - nz * Sx * nz ) / rr ) * &
                  THR / ( rr * rr )
      enddo
    enddo
   enddo
  enddo

  psi = chi + psi

  chi = ONE / psi **4 - ONE

  Lap = ONE / ( psi * psi ) - ONE

!~~~~~~ tilde Aij = Aij / Psi^6
  psi = psi * psi * psi * psi * psi * psi

  Axx = Axx / psi
  Ayy = Ayy / psi
  Azz = Azz / psi
  Axy = Axy / psi
  Axz = Axz / psi
  Ayz = Ayz / psi

  gxx = ZEO
  gyy = ZEO
  gzz = ZEO
  gxy = ZEO
  gxz = ZEO
  gyz = ZEO

  trK = ZEO

  Gmx = ZEO
  Gmy = ZEO
  Gmz = ZEO

  Sfx = ZEO
  Sfy = ZEO
  Sfz = ZEO

  dtSfx = ZEO
  dtSfy = ZEO
  dtSfz = ZEO
  
  return

  end subroutine get_ansorg_nbhs_ss
!-----------------------------------------------------------------------------------
!
!Set up approximate puncture initial data for n black holes
!
!-----------------------------------------------------------------------------------

  subroutine get_initial_nbhs(ex,X,Y,Z, &
                         chi,  trK, &
                         gxx,  gxy,  gxz,  gyy,  gyz,  gzz,&
                         Axx,  Axy,  Axz,  Ayy,  Ayz,  Azz,&
                         Gmx,  Gmy,  Gmz,  &
                         Lap,  Sfx,  Sfy,  Sfz,&
                         dtSfx,dtSfy,dtSfz,Mass,Porg,Pmom,Spin,N)

  implicit none

!------= input arguments
 
  integer,intent(in) :: N
  integer, dimension(3),  intent(in) :: ex
  real*8,  dimension(ex(1)), intent(in) :: X
  real*8,  dimension(ex(2)), intent(in) :: Y
  real*8,  dimension(ex(3)), intent(in) :: Z
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: chi
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: gxx,gxy,gxz,gyy,gyz,gzz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: Axx,Axy,Axz,Ayy,Ayz,Azz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: trK,Lap,Sfx,Sfy,Sfz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: Gmx,Gmy,Gmz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: dtSfx,dtSfy,dtSfz
  real*8,  dimension(N), intent(in) :: Mass
  real*8,  dimension(3*N), intent(in) :: Porg,Pmom,Spin

!------= local variables
  real*8,dimension(ex(1),ex(2),ex(3))::psi
  integer :: i,j,k,bhi
  real*8 :: M,Px,Py,Pz,PP,Sx,Sy,Sz,SS
  real*8 :: nx,ny,nz,rr,tmp
  real*8 :: u,u1,u2,u3,u4
  real*8 :: mup,mus,b,ell
  real*8,  parameter :: HLF = 5.d-1, ZEO = 0.d0, ONE = 1.d0, THR = 3.d0
  real*8,parameter::TINYRR=1.d-14

  do k = 1,ex(3)
   do j = 1,ex(2)
    do i = 1,ex(1)
! black hole 1
      M = mass(1)
      nx = x(i) - Porg(1)
      ny = y(j) - Porg(2)
      nz = z(k) - Porg(3)
      Px = Pmom(1)
      Py = Pmom(2)
      Pz = Pmom(3)
      Sx = Spin(1)
      Sy = Spin(2)
      Sz = Spin(3)

      rr = dsqrt(nx*nx+ny*ny+nz*nz)
      if(rr.lt.TINYRR) rr=(X(2)-X(1))/2.d0
      nx = nx / rr
      ny = ny / rr
      nz = nz / rr
      PP = dsqrt(Px**2 + Py**2 + Pz**2)
      if(PP .gt. 0.d0) then
        mup = (Px*nx+Py*ny+Pz*nz)/PP
      else
        mup = 0.0
      endif
      SS = dsqrt(Sx**2 + Sy**2 + Sz**2)
      if(SS .gt. 0.d0) then
        mus = (Sx*nx+Sy*ny+Sz*nz)/SS
      else
        mus = 0.0
      endif
      b = 2.d0*rr/M
      ell = 1.d0/(1.d0+b)

     u1 = 5.d0/8.d0*ell*(1.d0-2.d0*ell+2.d0*ell**2-ell**3+ell**4/5.d0)       
     u2 = (1.5d1+1.17d2*ell-7.9d1*ell**2+4.3d1*ell**3-1.4d1*ell**4+2.d0*ell**5 &
         +8.4d1*dlog(ell)/b)/4.d1/b**2
     u3 = ell/2.d1*(1.d0+ell+ell**2-4.d0*ell**3+2.d0*ell**4)
     u4 = ell**2/1.d1*(1.d1-2.5d1*ell+2.1d1*ell**2-6.d0*ell**3)

     tmp = (Py*Sz-Pz*Sy)*nx + (Pz*Sx-Px*Sz)*ny + (Px*Sy-Py*Sx)*nz

     u = PP**2/M**2*(u1 + u2*(3.d0*mup**2-ONE)) + &
         6.d0*u3/M**4*SS**2*(1.d0+mus**2) + u4/M**3*tmp

     psi(i,j,k) = ONE + u + HLF*M/rr

     tmp = Px * nx + Py * ny + Pz * nz

     Axx(i,j,k) = (HLF *( Px * nx + nx * Px - ( ONE - nx * nx )* tmp ) + &
                   ( nx * Sy * nz - nx * Sz * ny + nx * Sy * nz - nx * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Ayy(i,j,k) = (HLF *( Py * ny + ny * Py - ( ONE - ny * ny )* tmp ) + &
                   ( ny * Sz * nx - ny * Sx * nz + ny * Sz * nx - ny * Sx * nz ) / rr ) * &
                   THR / ( rr * rr )

     Azz(i,j,k) = (HLF *( Pz * nz + nz * Pz - ( ONE - nz * nz )* tmp ) + &
                   ( nz * Sx * ny - nz * Sy * nx + nz * Sx * ny - nz * Sy * nx ) / rr ) * &
                   THR / ( rr * rr )

     Axy(i,j,k) = (HLF *( Px * ny + nx * Py + nx * ny * tmp ) + &
                   ( nx * Sz * nx - nx * Sx * nz + ny * Sy * nz - ny * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Axz(i,j,k) = (HLF *( Px * nz + nx * Pz + nx * nz * tmp ) + &
                   ( nx * Sx * ny - nx * Sy * nx + nz * Sy * nz - nz * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Ayz(i,j,k) = (HLF *( Py * nz + ny * Pz + ny * nz * tmp ) + &
                   ( ny * Sx * ny - ny * Sy * nx + nz * Sz * nx - nz * Sx * nz ) / rr ) * &
                   THR / ( rr * rr )
! black hole 2 and 3, ...
     do bhi=2,N
      M = Mass(bhi)
      nx = x(i) - Porg(3*(bhi-1)+1)
      ny = y(j) - Porg(3*(bhi-1)+2)
      nz = z(k) - Porg(3*(bhi-1)+3)
      Px = Pmom(3*(bhi-1)+1)
      Py = Pmom(3*(bhi-1)+2)
      Pz = Pmom(3*(bhi-1)+3)
      Sx = Spin(3*(bhi-1)+1)
      Sy = Spin(3*(bhi-1)+2)
      Sz = Spin(3*(bhi-1)+3)

      rr = dsqrt(nx*nx+ny*ny+nz*nz)
      if(rr.lt.TINYRR) rr=(X(2)-X(1))/2.d0
      nx = nx / rr
      ny = ny / rr
      nz = nz / rr
      PP = dsqrt(Px**2 + Py**2 + Pz**2)
      if(PP .gt. 0.d0) then
        mup = (Px*nx+Py*ny+Pz*nz)/PP
      else
        mup = 0.0
      endif
      SS = dsqrt(Sx**2 + Sy**2 + Sz**2)
      if(SS .gt. 0.d0) then
        mus = (Sx*nx+Sy*ny+Sz*nz)/SS
      else
        mus = 0.0
      endif
      b = 2.d0*rr/M
      ell = 1.d0/(1.d0+b)

     u1 = 5.d0/8.d0*ell*(1.d0-2.d0*ell+2.d0*ell**2-ell**3+ell**4/5.d0)       
     u2 = (1.5d1+1.17d2*ell-7.9d1*ell**2+4.3d1*ell**3-1.4d1*ell**4+2.d0*ell**5 &
         +8.4d1*dlog(ell)/b)/4.d1/b**2
     u3 = ell/2.d1*(1.d0+ell+ell**2-4.d0*ell**3+2.d0*ell**4)
     u4 = ell**2/1.d1*(1.d1-2.5d1*ell+2.1d1*ell**2-6.d0*ell**3)

     tmp = (Py*Sz-Pz*Sy)*nx + (Pz*Sx-Px*Sz)*ny + (Px*Sy-Py*Sx)*nz

     u = PP**2/M**2*(u1 + u2*(3.d0*mup**2-ONE)) + &
         6.d0*u3/M**4*SS**2*(1.d0+mus**2) + u4/M**3*tmp

     psi(i,j,k) = psi(i,j,k) + u + HLF*M/rr

     tmp = Px * nx + Py * ny + Pz * nz

     Axx(i,j,k) = Axx(i,j,k) + &
                  (HLF *( Px * nx + nx * Px - ( ONE - nx * nx )* tmp ) + &
                  ( nx * Sy * nz - nx * Sz * ny + nx * Sy * nz - nx * Sz * ny ) / rr ) * &
                  THR / ( rr * rr )

     Ayy(i,j,k) = Ayy(i,j,k) + &
                  (HLF *( Py * ny + ny * Py - ( ONE - ny * ny )* tmp ) + &
                  ( ny * Sz * nx - ny * Sx * nz + ny * Sz * nx - ny * Sx * nz ) / rr ) * &
                  THR / ( rr * rr )

     Azz(i,j,k) = Azz(i,j,k) + &
                  (HLF *( Pz * nz + nz * Pz - ( ONE - nz * nz )* tmp ) + &
                  ( nz * Sx * ny - nz * Sy * nx + nz * Sx * ny - nz * Sy * nx ) / rr ) * &
                  THR / ( rr * rr )

     Axy(i,j,k) = Axy(i,j,k) + &
                  (HLF *( Px * ny + nx * Py + nx * ny * tmp ) + &
                  ( nx * Sz * nx - nx * Sx * nz + ny * Sy * nz - ny * Sz * ny ) / rr ) * &
                  THR / ( rr * rr )

     Axz(i,j,k) = Axz(i,j,k) + &
                  (HLF *( Px * nz + nx * Pz + nx * nz * tmp ) + &
                  ( nx * Sx * ny - nx * Sy * nx + nz * Sy * nz - nz * Sz * ny ) / rr ) * &
                  THR / ( rr * rr )

     Ayz(i,j,k) = Ayz(i,j,k) + &
                  (HLF *( Py * nz + ny * Pz + ny * nz * tmp ) + &
                  ( ny * Sx * ny - ny * Sy * nx + nz * Sz * nx - nz * Sx * nz ) / rr ) * &
                  THR / ( rr * rr )
      enddo
    enddo
   enddo
  enddo

  chi = ONE / psi **4 - ONE

  Lap = ONE / ( psi * psi ) - ONE

!~~~~~~ tilde Aij = Aij / Psi^6
  psi = psi * psi * psi * psi * psi * psi

  Axx = Axx / psi
  Ayy = Ayy / psi
  Azz = Azz / psi
  Axy = Axy / psi
  Axz = Axz / psi
  Ayz = Ayz / psi

  gxx = ZEO
  gyy = ZEO
  gzz = ZEO
  gxy = ZEO
  gxz = ZEO
  gyz = ZEO

  trK = ZEO

  Gmx = ZEO
  Gmy = ZEO
  Gmz = ZEO

  Sfx = ZEO
  Sfy = ZEO
  Sfz = ZEO

  dtSfx = ZEO
  dtSfy = ZEO
  dtSfz = ZEO
  
  return

  end subroutine get_initial_nbhs
!-----------------------------------------------------------------------------------
!
!Set up approximate puncture initial data for n black holes
!
!-----------------------------------------------------------------------------------

  subroutine get_initial_nbhs_ss(ex,X,Y,Z, &
                         chi,  trK, &
                         gxx,  gxy,  gxz,  gyy,  gyz,  gzz,&
                         Axx,  Axy,  Axz,  Ayy,  Ayz,  Azz,&
                         Gmx,  Gmy,  Gmz,  &
                         Lap,  Sfx,  Sfy,  Sfz,&
                         dtSfx,dtSfy,dtSfz,Mass,Porg,Pmom,Spin,N)

  implicit none

!------= input arguments
 
  integer,intent(in) :: N
  integer, dimension(3),  intent(in) :: ex
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(in) :: X,Y,Z
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: chi
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: gxx,gxy,gxz,gyy,gyz,gzz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: Axx,Axy,Axz,Ayy,Ayz,Azz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: trK,Lap,Sfx,Sfy,Sfz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: Gmx,Gmy,Gmz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: dtSfx,dtSfy,dtSfz
  real*8,  dimension(N), intent(in) :: Mass
  real*8,  dimension(3*N), intent(in) :: Porg,Pmom,Spin

!------= local variables
  real*8,dimension(ex(1),ex(2),ex(3))::psi
  integer :: i,j,k,bhi
  real*8 :: M,Px,Py,Pz,PP,Sx,Sy,Sz,SS
  real*8 :: nx,ny,nz,rr,tmp
  real*8 :: u,u1,u2,u3,u4
  real*8 :: mup,mus,b,ell
  real*8,  parameter :: HLF = 5.d-1, ZEO = 0.d0, ONE = 1.d0, THR = 3.d0
  real*8,parameter::TINYRR=1.d-14

  do k = 1,ex(3)
   do j = 1,ex(2)
    do i = 1,ex(1)
! black hole 1
      M = mass(1)
      nx = x(i,j,k) - Porg(1)
      ny = y(i,j,k) - Porg(2)
      nz = z(i,j,k) - Porg(3)
      Px = Pmom(1)
      Py = Pmom(2)
      Pz = Pmom(3)
      Sx = Spin(1)
      Sy = Spin(2)
      Sz = Spin(3)

      rr = dsqrt(nx*nx+ny*ny+nz*nz)
      if(rr.lt.TINYRR) rr=TINYRR
      nx = nx / rr
      ny = ny / rr
      nz = nz / rr
      PP = dsqrt(Px**2 + Py**2 + Pz**2)
      if(PP .gt. 0.d0) then
        mup = (Px*nx+Py*ny+Pz*nz)/PP
      else
        mup = 0.0
      endif
      SS = dsqrt(Sx**2 + Sy**2 + Sz**2)
      if(SS .gt. 0.d0) then
        mus = (Sx*nx+Sy*ny+Sz*nz)/SS
      else
        mus = 0.0
      endif
      b = 2.d0*rr/M
      ell = 1.d0/(1.d0+b)

     u1 = 5.d0/8.d0*ell*(1.d0-2.d0*ell+2.d0*ell**2-ell**3+ell**4/5.d0)       
     u2 = (1.5d1+1.17d2*ell-7.9d1*ell**2+4.3d1*ell**3-1.4d1*ell**4+2.d0*ell**5 &
         +8.4d1*dlog(ell)/b)/4.d1/b**2
     u3 = ell/2.d1*(1.d0+ell+ell**2-4.d0*ell**3+2.d0*ell**4)
     u4 = ell**2/1.d1*(1.d1-2.5d1*ell+2.1d1*ell**2-6.d0*ell**3)

     tmp = (Py*Sz-Pz*Sy)*nx + (Pz*Sx-Px*Sz)*ny + (Px*Sy-Py*Sx)*nz

     u = PP**2/M**2*(u1 + u2*(3.d0*mup**2-ONE)) + &
         6.d0*u3/M**4*SS**2*(1.d0+mus**2) + u4/M**3*tmp

     psi(i,j,k) = ONE + u + HLF*M/rr

     tmp = Px * nx + Py * ny + Pz * nz

     Axx(i,j,k) = (HLF *( Px * nx + nx * Px - ( ONE - nx * nx )* tmp ) + &
                   ( nx * Sy * nz - nx * Sz * ny + nx * Sy * nz - nx * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Ayy(i,j,k) = (HLF *( Py * ny + ny * Py - ( ONE - ny * ny )* tmp ) + &
                   ( ny * Sz * nx - ny * Sx * nz + ny * Sz * nx - ny * Sx * nz ) / rr ) * &
                   THR / ( rr * rr )

     Azz(i,j,k) = (HLF *( Pz * nz + nz * Pz - ( ONE - nz * nz )* tmp ) + &
                   ( nz * Sx * ny - nz * Sy * nx + nz * Sx * ny - nz * Sy * nx ) / rr ) * &
                   THR / ( rr * rr )

     Axy(i,j,k) = (HLF *( Px * ny + nx * Py + nx * ny * tmp ) + &
                   ( nx * Sz * nx - nx * Sx * nz + ny * Sy * nz - ny * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Axz(i,j,k) = (HLF *( Px * nz + nx * Pz + nx * nz * tmp ) + &
                   ( nx * Sx * ny - nx * Sy * nx + nz * Sy * nz - nz * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Ayz(i,j,k) = (HLF *( Py * nz + ny * Pz + ny * nz * tmp ) + &
                   ( ny * Sx * ny - ny * Sy * nx + nz * Sz * nx - nz * Sx * nz ) / rr ) * &
                   THR / ( rr * rr )
! black hole 2 and 3, ...
     do bhi=2,N
      M = Mass(bhi)
      nx = x(i,j,k) - Porg(3*(bhi-1)+1)
      ny = y(i,j,k) - Porg(3*(bhi-1)+2)
      nz = z(i,j,k) - Porg(3*(bhi-1)+3)
      Px = Pmom(3*(bhi-1)+1)
      Py = Pmom(3*(bhi-1)+2)
      Pz = Pmom(3*(bhi-1)+3)
      Sx = Spin(3*(bhi-1)+1)
      Sy = Spin(3*(bhi-1)+2)
      Sz = Spin(3*(bhi-1)+3)

      rr = dsqrt(nx*nx+ny*ny+nz*nz)
      if(rr.lt.TINYRR) rr=TINYRR
      nx = nx / rr
      ny = ny / rr
      nz = nz / rr
      PP = dsqrt(Px**2 + Py**2 + Pz**2)
      if(PP .gt. 0.d0) then
        mup = (Px*nx+Py*ny+Pz*nz)/PP
      else
        mup = 0.0
      endif
      SS = dsqrt(Sx**2 + Sy**2 + Sz**2)
      if(SS .gt. 0.d0) then
        mus = (Sx*nx+Sy*ny+Sz*nz)/SS
      else
        mus = 0.0
      endif
      b = 2.d0*rr/M
      ell = 1.d0/(1.d0+b)

     u1 = 5.d0/8.d0*ell*(1.d0-2.d0*ell+2.d0*ell**2-ell**3+ell**4/5.d0)       
     u2 = (1.5d1+1.17d2*ell-7.9d1*ell**2+4.3d1*ell**3-1.4d1*ell**4+2.d0*ell**5 &
         +8.4d1*dlog(ell)/b)/4.d1/b**2
     u3 = ell/2.d1*(1.d0+ell+ell**2-4.d0*ell**3+2.d0*ell**4)
     u4 = ell**2/1.d1*(1.d1-2.5d1*ell+2.1d1*ell**2-6.d0*ell**3)

     tmp = (Py*Sz-Pz*Sy)*nx + (Pz*Sx-Px*Sz)*ny + (Px*Sy-Py*Sx)*nz

     u = PP**2/M**2*(u1 + u2*(3.d0*mup**2-ONE)) + &
         6.d0*u3/M**4*SS**2*(1.d0+mus**2) + u4/M**3*tmp

     psi(i,j,k) = psi(i,j,k) + u + HLF*M/rr

     tmp = Px * nx + Py * ny + Pz * nz

     Axx(i,j,k) = Axx(i,j,k) + &
                  (HLF *( Px * nx + nx * Px - ( ONE - nx * nx )* tmp ) + &
                  ( nx * Sy * nz - nx * Sz * ny + nx * Sy * nz - nx * Sz * ny ) / rr ) * &
                  THR / ( rr * rr )

     Ayy(i,j,k) = Ayy(i,j,k) + &
                  (HLF *( Py * ny + ny * Py - ( ONE - ny * ny )* tmp ) + &
                  ( ny * Sz * nx - ny * Sx * nz + ny * Sz * nx - ny * Sx * nz ) / rr ) * &
                  THR / ( rr * rr )

     Azz(i,j,k) = Azz(i,j,k) + &
                  (HLF *( Pz * nz + nz * Pz - ( ONE - nz * nz )* tmp ) + &
                  ( nz * Sx * ny - nz * Sy * nx + nz * Sx * ny - nz * Sy * nx ) / rr ) * &
                  THR / ( rr * rr )

     Axy(i,j,k) = Axy(i,j,k) + &
                  (HLF *( Px * ny + nx * Py + nx * ny * tmp ) + &
                  ( nx * Sz * nx - nx * Sx * nz + ny * Sy * nz - ny * Sz * ny ) / rr ) * &
                  THR / ( rr * rr )

     Axz(i,j,k) = Axz(i,j,k) + &
                  (HLF *( Px * nz + nx * Pz + nx * nz * tmp ) + &
                  ( nx * Sx * ny - nx * Sy * nx + nz * Sy * nz - nz * Sz * ny ) / rr ) * &
                  THR / ( rr * rr )

     Ayz(i,j,k) = Ayz(i,j,k) + &
                  (HLF *( Py * nz + ny * Pz + ny * nz * tmp ) + &
                  ( ny * Sx * ny - ny * Sy * nx + nz * Sz * nx - nz * Sx * nz ) / rr ) * &
                  THR / ( rr * rr )
      enddo
    enddo
   enddo
  enddo

  chi = ONE / psi **4 - ONE

  Lap = ONE / ( psi * psi ) - ONE

!~~~~~~ tilde Aij = Aij / Psi^6
  psi = psi * psi * psi * psi * psi * psi

  Axx = Axx / psi
  Ayy = Ayy / psi
  Azz = Azz / psi
  Axy = Axy / psi
  Axz = Axz / psi
  Ayz = Ayz / psi

  gxx = ZEO
  gyy = ZEO
  gzz = ZEO
  gxy = ZEO
  gxz = ZEO
  gyz = ZEO

  trK = ZEO

  Gmx = ZEO
  Gmy = ZEO
  Gmz = ZEO

  Sfx = ZEO
  Sfy = ZEO
  Sfz = ZEO

  dtSfx = ZEO
  dtSfy = ZEO
  dtSfz = ZEO
  
  return

  end subroutine get_initial_nbhs_ss
!-----------------------------------------------------------------------------------
!
!Set up puncture initial data for inspiral binary with the given solution u 
!
!-----------------------------------------------------------------------------------

  subroutine get_pablo_nbhs(ex,X,Y,Z, &
                         chi,  trK, &
                         gxx,  gxy,  gxz,  gyy,  gyz,  gzz,&
                         Axx,  Axy,  Axz,  Ayy,  Ayz,  Azz,&
                         Gmx,  Gmy,  Gmz,  &
                         Lap,  Sfx,  Sfy,  Sfz,&
                         dtSfx,dtSfy,dtSfz,Mass,Porg,Pmom,Spin,N)

  implicit none

!------= input arguments

  integer,intent(in) :: N 
  integer, dimension(3),  intent(in) :: ex
  real*8,  dimension(ex(1)), intent(in) :: X
  real*8,  dimension(ex(2)), intent(in) :: Y
  real*8,  dimension(ex(3)), intent(in) :: Z
! in u, out chi  
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(inout) :: chi
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: gxx,gxy,gxz,gyy,gyz,gzz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: Axx,Axy,Axz,Ayy,Ayz,Azz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: trK,Lap,Sfx,Sfy,Sfz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: Gmx,Gmy,Gmz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: dtSfx,dtSfy,dtSfz
  real*8,  dimension(N), intent(in) :: Mass
  real*8,  dimension(3*N), intent(in) :: Porg,Pmom,Spin

!------= local variables
  real*8,dimension(ex(1),ex(2),ex(3))::psi
  integer :: i,j,k,bhi
  real*8 :: M,Px,Py,Pz,Sx,Sy,Sz
  real*8 :: nx,ny,nz,rr,tmp
  real*8,  parameter :: HLF = 5.d-1, ZEO = 0.d0, ONE = 1.d0, THR = 3.d0
  real*8,parameter::TINYRR=1.d-14

  do k = 1,ex(3)
   do j = 1,ex(2)
    do i = 1,ex(1)
! black hole 1
      M = mass(1)
      nx = x(i) - Porg(1)
      ny = y(j) - Porg(2)
      nz = z(k) - Porg(3)
      Px = Pmom(1)
      Py = Pmom(2)
      Pz = Pmom(3)
      Sx = Spin(1)
      Sy = Spin(2)
      Sz = Spin(3)

      rr = dsqrt(nx*nx+ny*ny+nz*nz)
      if(rr.lt.TINYRR) rr=(X(2)-X(1))/2.d0
      nx = nx / rr
      ny = ny / rr
      nz = nz / rr

     psi(i,j,k) = ONE + chi(i,j,k) + HLF*M/rr

     tmp = Px * nx + Py * ny + Pz * nz

     Axx(i,j,k) = (HLF *( Px * nx + nx * Px - ( ONE - nx * nx )* tmp ) + &
                   ( nx * Sy * nz - nx * Sz * ny + nx * Sy * nz - nx * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Ayy(i,j,k) = (HLF *( Py * ny + ny * Py - ( ONE - ny * ny )* tmp ) + &
                   ( ny * Sz * nx - ny * Sx * nz + ny * Sz * nx - ny * Sx * nz ) / rr ) * &
                   THR / ( rr * rr )

     Azz(i,j,k) = (HLF *( Pz * nz + nz * Pz - ( ONE - nz * nz )* tmp ) + &
                   ( nz * Sx * ny - nz * Sy * nx + nz * Sx * ny - nz * Sy * nx ) / rr ) * &
                   THR / ( rr * rr )

     Axy(i,j,k) = (HLF *( Px * ny + nx * Py + nx * ny * tmp ) + &
                   ( nx * Sz * nx - nx * Sx * nz + ny * Sy * nz - ny * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Axz(i,j,k) = (HLF *( Px * nz + nx * Pz + nx * nz * tmp ) + &
                   ( nx * Sx * ny - nx * Sy * nx + nz * Sy * nz - nz * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Ayz(i,j,k) = (HLF *( Py * nz + ny * Pz + ny * nz * tmp ) + &
                   ( ny * Sx * ny - ny * Sy * nx + nz * Sz * nx - nz * Sx * nz ) / rr ) * &
                   THR / ( rr * rr )
! black hole 2 and 3, ...
     do bhi=2,N
      M = Mass(bhi)
      nx = x(i) - Porg(3*(bhi-1)+1)
      ny = y(j) - Porg(3*(bhi-1)+2)
      nz = z(k) - Porg(3*(bhi-1)+3)
      Px = Pmom(3*(bhi-1)+1)
      Py = Pmom(3*(bhi-1)+2)
      Pz = Pmom(3*(bhi-1)+3)
      Sx = Spin(3*(bhi-1)+1)
      Sy = Spin(3*(bhi-1)+2)
      Sz = Spin(3*(bhi-1)+3)

      rr = dsqrt(nx*nx+ny*ny+nz*nz)
      if(rr.lt.TINYRR) rr=(X(2)-X(1))/2.d0
      nx = nx / rr
      ny = ny / rr
      nz = nz / rr

     psi(i,j,k) = psi(i,j,k) + HLF*M/rr

     tmp = Px * nx + Py * ny + Pz * nz

     Axx(i,j,k) = Axx(i,j,k) + &
                  (HLF *( Px * nx + nx * Px - ( ONE - nx * nx )* tmp ) + &
                  ( nx * Sy * nz - nx * Sz * ny + nx * Sy * nz - nx * Sz * ny ) / rr ) * &
                  THR / ( rr * rr )

     Ayy(i,j,k) = Ayy(i,j,k) + &
                  (HLF *( Py * ny + ny * Py - ( ONE - ny * ny )* tmp ) + &
                  ( ny * Sz * nx - ny * Sx * nz + ny * Sz * nx - ny * Sx * nz ) / rr ) * &
                  THR / ( rr * rr )

     Azz(i,j,k) = Azz(i,j,k) + &
                  (HLF *( Pz * nz + nz * Pz - ( ONE - nz * nz )* tmp ) + &
                  ( nz * Sx * ny - nz * Sy * nx + nz * Sx * ny - nz * Sy * nx ) / rr ) * &
                  THR / ( rr * rr )

     Axy(i,j,k) = Axy(i,j,k) + &
                  (HLF *( Px * ny + nx * Py + nx * ny * tmp ) + &
                  ( nx * Sz * nx - nx * Sx * nz + ny * Sy * nz - ny * Sz * ny ) / rr ) * &
                  THR / ( rr * rr )

     Axz(i,j,k) = Axz(i,j,k) + &
                  (HLF *( Px * nz + nx * Pz + nx * nz * tmp ) + &
                  ( nx * Sx * ny - nx * Sy * nx + nz * Sy * nz - nz * Sz * ny ) / rr ) * &
                  THR / ( rr * rr )

     Ayz(i,j,k) = Ayz(i,j,k) + &
                  (HLF *( Py * nz + ny * Pz + ny * nz * tmp ) + &
                  ( ny * Sx * ny - ny * Sy * nx + nz * Sz * nx - nz * Sx * nz ) / rr ) * &
                  THR / ( rr * rr )
     enddo
    enddo
   enddo
  enddo

  chi = ONE / psi **4 - ONE

  Lap = ONE / ( psi * psi ) - ONE

!~~~~~~ tilde Aij = Aij / Psi^6
  psi = psi * psi * psi * psi * psi * psi

  Axx = Axx / psi
  Ayy = Ayy / psi
  Azz = Azz / psi
  Axy = Axy / psi
  Axz = Axz / psi
  Ayz = Ayz / psi

  gxx = ZEO
  gyy = ZEO
  gzz = ZEO
  gxy = ZEO
  gxz = ZEO
  gyz = ZEO

  trK = ZEO

  Gmx = ZEO
  Gmy = ZEO
  Gmz = ZEO

  Sfx = ZEO
  Sfy = ZEO
  Sfz = ZEO

  dtSfx = ZEO
  dtSfy = ZEO
  dtSfz = ZEO
  
  return

  end subroutine get_pablo_nbhs
!-----------------------------------------------------------------------------------
!
!Set up approximate puncture initial data for n black holes with lousto's
!formula PRD 77, 024034 (2008)
!
!-----------------------------------------------------------------------------------

  subroutine get_lousto_nbhs(ex,X,Y,Z, &
                         chi,  trK, &
                         gxx,  gxy,  gxz,  gyy,  gyz,  gzz,&
                         Axx,  Axy,  Axz,  Ayy,  Ayz,  Azz,&
                         Gmx,  Gmy,  Gmz,  &
                         Lap,  Sfx,  Sfy,  Sfz,&
                         dtSfx,dtSfy,dtSfz,Mass,Porg,Pmom,Spin,N)

  implicit none

!------= input arguments
 
  integer,intent(in) :: N
  integer, dimension(3),  intent(in) :: ex
  real*8,  dimension(ex(1)), intent(in) :: X
  real*8,  dimension(ex(2)), intent(in) :: Y
  real*8,  dimension(ex(3)), intent(in) :: Z
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: chi
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: gxx,gxy,gxz,gyy,gyz,gzz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: Axx,Axy,Axz,Ayy,Ayz,Azz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: trK,Lap,Sfx,Sfy,Sfz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: Gmx,Gmy,Gmz
  real*8,  dimension(ex(1),ex(2),ex(3)), intent(out) :: dtSfx,dtSfy,dtSfz
  real*8,  dimension(N), intent(in) :: Mass
  real*8,  dimension(3*N), intent(in) :: Porg,Pmom,Spin

!------= local variables
  real*8,dimension(ex(1),ex(2),ex(3))::psi
  integer :: i,j,k,bhi
  real*8 :: M,Px,Py,Pz,PP,Sx,Sy,Sz,SS
  real*8 :: nx,ny,nz,rr,tmp
  real*8 :: u,u1,u2,u3,u4,u5
  real*8 :: mup,mus,b,ell
  real*8,  parameter :: HLF = 5.d-1, ZEO = 0.d0, ONE = 1.d0, THR = 3.d0
  real*8,parameter::TINYRR=1.d-14

  do k = 1,ex(3)
   do j = 1,ex(2)
    do i = 1,ex(1)
! black hole 1
      M = mass(1)
      nx = x(i) - Porg(1)
      ny = y(j) - Porg(2)
      nz = z(k) - Porg(3)
      Px = Pmom(1)
      Py = Pmom(2)
      Pz = Pmom(3)
      Sx = Spin(1)
      Sy = Spin(2)
      Sz = Spin(3)

      rr = dsqrt(nx*nx+ny*ny+nz*nz)
      if(rr.lt.TINYRR) rr=(X(2)-X(1))/2.d0
      nx = nx / rr
      ny = ny / rr
      nz = nz / rr
      PP = dsqrt(Px**2 + Py**2 + Pz**2)
      if(PP .gt. 0.d0) then
        mup = (Px*nx+Py*ny+Pz*nz)/PP
      else
        mup = 0.0
      endif
      SS = dsqrt(Sx**2 + Sy**2 + Sz**2)
      if(SS .gt. 0.d0) then
        mus = (Sx*nx+Sy*ny+Sz*nz)/SS
      else
        mus = 0.0
      endif
      b = 2.d0*rr/M
      ell = 1.d0/(1.d0+b)

      u1 = 5.d0/8.d0*ell*(1.d0-2.d0*ell+2.d0*ell**2-ell**3+ell**4/5.d0)       
      u2 = (1.5d1+1.17d2*ell-7.9d1*ell**2+4.3d1*ell**3-1.4d1*ell**4+2.d0*ell**5 &
          +8.4d1*dlog(ell)/b)/4.d1/b**2
      u3 = ell+ell**2+ell**3-4.d0*ell**4+2.d0*ell**5
      u4 = -b**2*ell**5
      u5 = b*(1.d0+5.d0*b+1.d1*b**2)*ell**5/8.d1

      tmp = (Py*Sz-Pz*Sy)*nx + (Pz*Sx-Px*Sz)*ny + (Px*Sy-Py*Sx)*nz

      u = PP**2/M**2*(u1 + u2*(3.d0*mup**2-ONE)) + &
          2.d0*SS**2/5.d0/M**4*(u3 + u4*(3.d0*mus**2-ONE)) + &
          u5*tmp

     psi(i,j,k) = ONE + u + HLF*M/rr

     tmp = Px * nx + Py * ny + Pz * nz

     Axx(i,j,k) = (HLF *( Px * nx + nx * Px - ( ONE - nx * nx )* tmp ) + &
                   ( nx * Sy * nz - nx * Sz * ny + nx * Sy * nz - nx * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Ayy(i,j,k) = (HLF *( Py * ny + ny * Py - ( ONE - ny * ny )* tmp ) + &
                   ( ny * Sz * nx - ny * Sx * nz + ny * Sz * nx - ny * Sx * nz ) / rr ) * &
                   THR / ( rr * rr )

     Azz(i,j,k) = (HLF *( Pz * nz + nz * Pz - ( ONE - nz * nz )* tmp ) + &
                   ( nz * Sx * ny - nz * Sy * nx + nz * Sx * ny - nz * Sy * nx ) / rr ) * &
                   THR / ( rr * rr )

     Axy(i,j,k) = (HLF *( Px * ny + nx * Py + nx * ny * tmp ) + &
                   ( nx * Sz * nx - nx * Sx * nz + ny * Sy * nz - ny * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Axz(i,j,k) = (HLF *( Px * nz + nx * Pz + nx * nz * tmp ) + &
                   ( nx * Sx * ny - nx * Sy * nx + nz * Sy * nz - nz * Sz * ny ) / rr ) * &
                   THR / ( rr * rr )

     Ayz(i,j,k) = (HLF *( Py * nz + ny * Pz + ny * nz * tmp ) + &
                   ( ny * Sx * ny - ny * Sy * nx + nz * Sz * nx - nz * Sx * nz ) / rr ) * &
                   THR / ( rr * rr )
! black hole 2 and 3, ...
     do bhi=2,N
      M = Mass(bhi)
      nx = x(i) - Porg(3*(bhi-1)+1)
      ny = y(j) - Porg(3*(bhi-1)+2)
      nz = z(k) - Porg(3*(bhi-1)+3)
      Px = Pmom(3*(bhi-1)+1)
      Py = Pmom(3*(bhi-1)+2)
      Pz = Pmom(3*(bhi-1)+3)
      Sx = Spin(3*(bhi-1)+1)
      Sy = Spin(3*(bhi-1)+2)
      Sz = Spin(3*(bhi-1)+3)

      rr = dsqrt(nx*nx+ny*ny+nz*nz)
      if(rr.lt.TINYRR) rr=(X(2)-X(1))/2.d0
      nx = nx / rr
      ny = ny / rr
      nz = nz / rr
      PP = dsqrt(Px**2 + Py**2 + Pz**2)
      if(PP .gt. 0.d0) then
        mup = (Px*nx+Py*ny+Pz*nz)/PP
      else
        mup = 0.0
      endif
      SS = dsqrt(Sx**2 + Sy**2 + Sz**2)
      if(SS .gt. 0.d0) then
        mus = (Sx*nx+Sy*ny+Sz*nz)/SS
      else
        mus = 0.0
      endif
      b = 2.d0*rr/M
      ell = 1.d0/(1.d0+b)

      u1 = 5.d0/8.d0*ell*(1.d0-2.d0*ell+2.d0*ell**2-ell**3+ell**4/5.d0)       
      u2 = (1.5d1+1.17d2*ell-7.9d1*ell**2+4.3d1*ell**3-1.4d1*ell**4+2.d0*ell**5 &
          +8.4d1*dlog(ell)/b)/4.d1/b**2
      u3 = ell+ell**2+ell**3-4.d0*ell**4+2.d0*ell**5
      u4 = -b**2*ell**5
      u5 = b*(1.d0+5.d0*b+1.d1*b**2)*ell**5/8.d1

      tmp = (Py*Sz-Pz*Sy)*nx + (Pz*Sx-Px*Sz)*ny + (Px*Sy-Py*Sx)*nz

      u = PP**2/M**2*(u1 + u2*(3.d0*mup**2-ONE)) + &
          2.d0*SS**2/5.d0/M**4*(u3 + u4*(3.d0*mus**2-ONE)) + &
          u5*tmp

     psi(i,j,k) = psi(i,j,k) + u + HLF*M/rr

     tmp = Px * nx + Py * ny + Pz * nz

     Axx(i,j,k) = Axx(i,j,k) + &
                  (HLF *( Px * nx + nx * Px - ( ONE - nx * nx )* tmp ) + &
                  ( nx * Sy * nz - nx * Sz * ny + nx * Sy * nz - nx * Sz * ny ) / rr ) * &
                  THR / ( rr * rr )

     Ayy(i,j,k) = Ayy(i,j,k) + &
                  (HLF *( Py * ny + ny * Py - ( ONE - ny * ny )* tmp ) + &
                  ( ny * Sz * nx - ny * Sx * nz + ny * Sz * nx - ny * Sx * nz ) / rr ) * &
                  THR / ( rr * rr )

     Azz(i,j,k) = Azz(i,j,k) + &
                  (HLF *( Pz * nz + nz * Pz - ( ONE - nz * nz )* tmp ) + &
                  ( nz * Sx * ny - nz * Sy * nx + nz * Sx * ny - nz * Sy * nx ) / rr ) * &
                  THR / ( rr * rr )

     Axy(i,j,k) = Axy(i,j,k) + &
                  (HLF *( Px * ny + nx * Py + nx * ny * tmp ) + &
                  ( nx * Sz * nx - nx * Sx * nz + ny * Sy * nz - ny * Sz * ny ) / rr ) * &
                  THR / ( rr * rr )

     Axz(i,j,k) = Axz(i,j,k) + &
                  (HLF *( Px * nz + nx * Pz + nx * nz * tmp ) + &
                  ( nx * Sx * ny - nx * Sy * nx + nz * Sy * nz - nz * Sz * ny ) / rr ) * &
                  THR / ( rr * rr )

     Ayz(i,j,k) = Ayz(i,j,k) + &
                  (HLF *( Py * nz + ny * Pz + ny * nz * tmp ) + &
                  ( ny * Sx * ny - ny * Sy * nx + nz * Sz * nx - nz * Sx * nz ) / rr ) * &
                  THR / ( rr * rr )
      enddo
    enddo
   enddo
  enddo

  chi = ONE / psi **4 - ONE

  Lap = ONE / ( psi * psi ) - ONE

!~~~~~~ tilde Aij = Aij / Psi^6
  psi = psi * psi * psi * psi * psi * psi

  Axx = Axx / psi
  Ayy = Ayy / psi
  Azz = Azz / psi
  Axy = Axy / psi
  Axz = Axz / psi
  Ayz = Ayz / psi

  gxx = ZEO
  gyy = ZEO
  gzz = ZEO
  gxy = ZEO
  gxz = ZEO
  gyz = ZEO

  trK = ZEO

  Gmx = ZEO
  Gmy = ZEO
  Gmz = ZEO

  Sfx = ZEO
  Sfy = ZEO
  Sfz = ZEO

  dtSfx = ZEO
  dtSfy = ZEO
  dtSfz = ZEO
  
  return

  end subroutine get_lousto_nbhs
