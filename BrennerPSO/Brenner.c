#include<stdio.h>
#include<stdio.h>
#include<math.h>
#include <sys/time.h>
#include<stdlib.h>
#include<omp.h>
#include<string.h>
#include "Brenner.h"

const int n = 200;

const float R1 = 1.7,R2 = 2,Re = 1.315;
const float a0 = 0.011304, c0 = 19, d0 = 2.5;
const float delta = 0.80469, beta = 1.5, S = 1.29, De = 6.325;
const float De_S_1 = De/(S-1);
const float DexS_S_1 = De_S_1*S;
const float c0sq = c0*c0, d0sq = d0*d0;
float c0sq_d0sq_1 = 1 + (c0sq/d0sq);
const float pi = 3.14159265359;
float pi_DeltaR = pi/(R2-R1);

float f_ij(float rij){
    float ret;
    if(rij>R2)ret =  0.0;
    else if(rij<R1)ret = 1.0;
    else ret = 0.5*(1+cos(pi_DeltaR*(rij-R1)));
    return ret;     //return rij<R1?1:(rij>R2?0:0.5*(1+cos(pi_DeltaR*(rij-R1))));
}

float F(float xik){
    float ret;
    if(xik<=2)ret = 1;
    else if(xik>3)ret = 0;
    else 0.5*(1+cos(pi*(xik-2) ));
    return ret;
}

float BrennerPotential(float* r){
    float s = 0;
    int i,j,k,m;
    float Vr,Va,rij,rx,ry,rz,rij_Re,rmk;
    float r_x1,r_y1,r_z1,r_x2,r_y2,r_z2;
    float Fij =0,Bij,Bji,B_ij,Gc_ijk,Gc_jik;
    float Ni,Nj,Nconj,F_ij;
    float sqrt_2_Sxbeta = sqrt(2/S)*beta;
    float sqrt_2xSxbeta = sqrt(2*S)*beta;
    float cos_ijk,rik,rjk,fij,fik,fjk,xik,xjk;
    Bij=0;Bji=0;B_ij=0;Ni=0,Nj=0,Nconj=1,Gc_ijk=0;

    #pragma omp simd
    for(i=0;i<n;i++){
        for(j=i+1;j<n;j++){
            Bij=1;Bji=1;B_ij=0;Ni=0,Nj=0,Nconj=1,Gc_ijk=0,Gc_jik=0;
            rx = (r[i*3]-r[j*3]);
            ry = (r[i*3+1]-r[j*3+1]);
            rz = (r[i*3+2]-r[j*3+2]);
            rij = sqrt(rx*rx+ry*ry+rz*rz);
            fij = f_ij(rij);

            for(k=0;k<n;k++){ 
                r_x1 = (r[k*3]-r[i*3]);
                r_y1 = (r[k*3+1]-r[i*3+1]);
                r_z1 = (r[k*3+2]-r[i*3+2]);
                rik = sqrt(r_x1*r_x1+r_y1*r_y1+r_z1*r_z1);
                fik = f_ij(rik);

                r_x2 = (r[k*3]-r[j*3]);
                r_y2 = (r[k*3+1]-r[j*3+1]);
                r_z2 = (r[k*3+2]-r[j*3+2]);
                rjk = sqrt(r_x2*r_x2+r_y2*r_y2+r_z2*r_z2);
                fjk = f_ij(rjk);

                if(k!=j && k!=i){
                    cos_ijk = -(rx*r_x1+ry*r_y1+rz*r_z1)/(rik*rij);
                    cos_ijk = (cos_ijk+1);
                    cos_ijk = cos_ijk*cos_ijk;
                    Gc_ijk = a0*(c0sq_d0sq_1 - (c0sq/(d0sq+cos_ijk)));

                    cos_ijk = (rx*r_x2+ry*r_y2+rz*r_z2)/(rjk*rij);
                    cos_ijk = (cos_ijk+1);
                    cos_ijk = cos_ijk*cos_ijk;
                    Gc_jik = a0*(c0sq_d0sq_1-c0sq/(d0sq+cos_ijk));

                    Bij += Gc_ijk*fik;
                    Bji += Gc_jik*fjk;
                }

                if(k!=j) Ni += fik;
                if(k!=i) Nj += fjk;

                xik = 0;
                xjk = 0;
                for(m=0;m<n;m++){
                    rx = (r[k*3]-r[m*3]);
                    ry = (r[k*3+1]-r[m*3+1]);
                    rz = (r[k*3+2]-r[m*3+2]);
                    rmk = sqrt(rx*rx+ry*ry+rz*rz);
                    if(m!=i) xik += f_ij(rmk);
                    if(m!=j) xjk += f_ij(rmk);
                }
                if(k!=i && k!=j)Nconj += fik*F(xik) + fjk*F(xjk);
            }

            Bij = pow(Bij,-delta);
            Bji = pow(Bji,-delta);
            B_ij = 0.5*(Bij+Bji);

            rij_Re = (rij-Re);
            Vr = exp(-sqrt_2xSxbeta*rij_Re);
            Vr = Vr*De_S_1*fij;
            Va = exp(-sqrt_2_Sxbeta*rij_Re);
            Va = Va*DexS_S_1*fij;
            s += Vr-B_ij*Va; 
        }
    }

    return s;
}
