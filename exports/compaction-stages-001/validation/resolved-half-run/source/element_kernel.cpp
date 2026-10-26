// Optional CPU element kernels. No fast-math; mm, N, MPa, tonne, s.
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>
static void mul(const double*a,const double*b,double*c){for(int i=0;i<3;i++)for(int j=0;j<3;j++){c[3*i+j]=0;for(int k=0;k<3;k++)c[3*i+j]+=a[3*i+k]*b[3*k+j];}}
static double determinant(const double*a){return a[0]*(a[4]*a[8]-a[5]*a[7])-a[1]*(a[3]*a[8]-a[5]*a[6])+a[2]*(a[3]*a[7]-a[4]*a[6]);}
static bool geom(const double*x,const int64_t*t,double&v,double*g,double*D){
 for(int i=0;i<3;i++)for(int j=0;j<3;j++)D[3*i+j]=x[3*t[j+1]+i]-x[3*t[0]+i];
 double d=determinant(D);if(!(d>0)||!std::isfinite(d))return false;v=d/6;
 double inv[9]={D[4]*D[8]-D[5]*D[7],D[2]*D[7]-D[1]*D[8],D[1]*D[5]-D[2]*D[4],D[5]*D[6]-D[3]*D[8],D[0]*D[8]-D[2]*D[6],D[2]*D[3]-D[0]*D[5],D[3]*D[7]-D[4]*D[6],D[1]*D[6]-D[0]*D[7],D[0]*D[4]-D[1]*D[3]};
 for(int i=0;i<9;i++)g[3+i]=inv[i]/d;for(int j=0;j<3;j++)g[j]=-g[3+j]-g[6+j]-g[9+j];return true;
}
// Cyclic Jacobi rotations, with a convergence check, never eigenvalue clipping.
static bool eig(const double*input,double*value,double*Q){
 double a[9];for(int i=0;i<3;i++)for(int j=0;j<3;j++){a[3*i+j]=.5*(input[3*i+j]+input[3*j+i]);if(!std::isfinite(a[3*i+j]))return false;Q[3*i+j]=(i==j);}
 bool done=false;
 for(int sweep=0;sweep<24;sweep++){
  double scale=std::max({std::abs(a[0]),std::abs(a[4]),std::abs(a[8]),1e-300});
  if(std::max({std::abs(a[1]),std::abs(a[2]),std::abs(a[5])})<=2e-15*scale){done=true;break;}
  for(int p=0;p<2;p++)for(int q=p+1;q<3;q++){
   double off=a[3*p+q];if(std::abs(off)<=1e-17*scale)continue;
   double tau=(a[3*q+q]-a[3*p+p])/(2*off);
   double z=std::copysign(1.,tau)/(std::abs(tau)+std::hypot(1.,tau));double c=1/std::sqrt(1+z*z),s=z*c;
   double ap=a[3*p+p],aq=a[3*q+q];a[3*p+p]=ap-z*off;a[3*q+q]=aq+z*off;a[3*p+q]=a[3*q+p]=0;
   for(int k=0;k<3;k++)if(k!=p&&k!=q){double kp=a[3*k+p],kq=a[3*k+q];a[3*k+p]=a[3*p+k]=c*kp-s*kq;a[3*k+q]=a[3*q+k]=s*kp+c*kq;}
   for(int k=0;k<3;k++){double kp=Q[3*k+p],kq=Q[3*k+q];Q[3*k+p]=c*kp-s*kq;Q[3*k+q]=s*kp+c*kq;}
  }
 }
 if(!done)return false;for(int k=0;k<3;k++){value[k]=a[3*k+k];if(!(value[k]>0)||!std::isfinite(value[k]))return false;}return true;
}
static bool response(const double*b,double ep,double E,double nu,double y,double H,double*sigma,double&energy,double&J){
 double val[3],Q[9];if(!eig(b,val,Q))return false;double e[3],trace=0;for(int k=0;k<3;k++){e[k]=.5*std::log(val[k]);trace+=e[k];}
 double G=E/(2*(1+nu)),K=E/(3*(1-2*nu));J=std::exp(trace);energy=.5*K*trace*trace+y*ep+.5*H*ep*ep;
 double tau[3];for(int k=0;k<3;k++){double dev=e[k]-trace/3;tau[k]=2*G*dev+K*trace;energy+=G*dev*dev;}
 for(int i=0;i<3;i++)for(int j=0;j<3;j++){double s=0;for(int k=0;k<3;k++)s+=Q[3*i+k]*tau[k]*Q[3*j+k];sigma[3*i+j]=s/J;}return std::isfinite(energy)&&std::isfinite(J);
}
extern "C" int pf_geometry(int64_t ne,const double*x,const int64_t*t,double*v,double*g,double*D){for(int64_t e=0;e<ne;e++)if(!geom(x,t+4*e,v[e],g+12*e,D+9*e))return int(e+1);return 0;}
extern "C" int pf_advance(int64_t ne,const double*b,const double*ep,const double*F,double E,double nu,double y,double H,double*out,double*pe){
 double G=E/(2*(1+nu));
 for(int64_t e=0;e<ne;e++){
  double tmp[9],trans[9],trial[9],value[3],Q[9];mul(F+9*e,b+9*e,tmp);for(int i=0;i<3;i++)for(int j=0;j<3;j++)trans[3*i+j]=F[9*e+3*j+i];mul(tmp,trans,trial);if(!eig(trial,value,Q))return int(e+1);
  double strain[3],s[3],trace=0,seq=0;for(int k=0;k<3;k++){strain[k]=.5*std::log(value[k]);trace+=strain[k];}
  for(int k=0;k<3;k++){s[k]=2*G*(strain[k]-trace/3);seq+=1.5*s[k]*s[k];}seq=std::sqrt(seq);double dg=std::max(0.,(seq-y-H*ep[e])/(3*G+H));
  for(int k=0;k<3;k++)value[k]=std::exp(2*(strain[k]-dg*1.5*s[k]/std::max(seq,1e-30)));
  for(int i=0;i<3;i++)for(int j=0;j<3;j++){double z=0;for(int k=0;k<3;k++)z+=Q[3*i+k]*value[k]*Q[3*j+k];out[9*e+3*i+j]=z;}pe[e]=ep[e]+dg;
 }return 0;
}
extern "C" int pf_internal(int64_t ne,int64_t nn,const double*x,const int64_t*t,const double*b,const double*ep,double E,double nu,double y,double H,double*force,double*v,double*energy,double*J,double*D,double*sigma){
 std::fill(force,force+3*nn,0.);
 for(int64_t e=0;e<ne;e++){
  double g[12];if(!geom(x,t+4*e,v[e],g,D+9*e))return int(e+1);if(!response(b+9*e,ep[e],E,nu,y,H,sigma+9*e,energy[e],J[e]))return int(e+1);
  for(int n=0;n<4;n++)for(int i=0;i<3;i++){double f=0;for(int j=0;j<3;j++)f+=sigma[9*e+3*i+j]*g[3*n+j];force[3*t[4*e+n]+i]-=v[e]*f;}
 }return 0;
}
extern "C" int pf_quality(int64_t ne,const double*x,const int64_t*t,double*q){
 for(int64_t e=0;e<ne;e++){double v,g[12],D[9];if(!geom(x,t+4*e,v,g,D))return int(e+1);double length=0;for(int i=0;i<4;i++)for(int j=i+1;j<4;j++)for(int k=0;k<3;k++){double d=x[3*t[4*e+i]+k]-x[3*t[4*e+j]+k];length+=d*d;}q[e]=12*std::pow(3*v,2./3)/length;}return 0;
}
extern "C" int pf_bound(int64_t ne,int64_t nn,const double*x,const int64_t*t,const double*mass,double E,double nu,const double*sigma,const double*b,int has_sigma,int has_b,double*rows){
 std::fill(rows,rows+3*nn,0.);double G=E/(2*(1+nu)),K=E/(3*(1-2*nu)),lam=K-2*G/3;
 for(int64_t e=0;e<ne;e++){
  double v,g[12],D[9],inflation=1,stress=0;if(!geom(x,t+4*e,v,g,D))return int(e+1);
  if(has_b){double val[3],Q[9];if(!eig(b+9*e,val,Q))return int(e+1);double largest=0;for(int k=0;k<3;k++)largest=std::max(largest,std::abs(std::log(val[k])));inflation=(1+largest)/std::sqrt(val[0]*val[1]*val[2]);}
  if(has_sigma){for(int k=0;k<9;k++)stress+=sigma[9*e+k]*sigma[9*e+k];stress=2*std::sqrt(stress);}
  double invm[4],ones[4],S1=0,Sa[3]={0,0,0};for(int i=0;i<4;i++){double m=mass[t[4*e+i]];if(!(m>0))return int(e+1);invm[i]=1/std::sqrt(m);ones[i]=0;for(int a=0;a<3;a++){ones[i]+=std::abs(g[3*i+a]);Sa[a]+=std::abs(g[3*i+a])*invm[i];}S1+=ones[i]*invm[i];}
  for(int i=0;i<4;i++){double Sd=0;for(int j=0;j<4;j++){double dot=0;for(int a=0;a<3;a++)dot+=g[3*i+a]*g[3*j+a];Sd+=std::abs(dot)*invm[j];}
   for(int a=0;a<3;a++)rows[3*t[4*e+i]+a]+=v*invm[i]*(inflation*(std::abs(lam)*std::abs(g[3*i+a])*S1+G*Sd+G*ones[i]*Sa[a])+stress*Sd);
  }
 }return 0;
}
