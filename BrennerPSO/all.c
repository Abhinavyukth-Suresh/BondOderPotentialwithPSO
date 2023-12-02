#include<stdio.h>
#include<stdio.h>
#include<math.h>
#include <sys/time.h>
#include<stdlib.h>
#include<omp.h>
#include<string.h>

const int n = N;
#define n_threads = NTHREADS

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


long gettime(){
	struct timeval currentTime;
	gettimeofday(&currentTime, NULL);
	return currentTime.tv_sec * (int)1e6 + currentTime.tv_usec;
}

/*************** PARTICLE SWARM OPTIMIZER  ****************/


typedef struct {
    size_t swarm_size;
    size_t input_dim;
    float upper_bound;
    float lower_bound;
    float** swarm; 
    float** vel;
    float global_best;
    float* Gbest_vector;
    float* personal_best;
    float** Pbest_vector;
    float (*f)(float*);
}Swarm;     

Swarm init(float (*optim_func)(float*),size_t swarm_size,size_t input_vec_dim,float upper_bound,float lower_bound){
    srand(time(NULL));
    float** swarm =calloc(swarm_size,sizeof(float*));
    float** velocity = calloc(swarm_size,sizeof(float*));
    float** Pbest_vector = calloc(swarm_size,sizeof(float*));
    float global_best = (float)INFINITY;
    float* personal_best = calloc(swarm_size,sizeof(float));
    float* Gbest_vector = calloc(input_vec_dim,sizeof(float));
    
    for(int i=0;i<swarm_size;i++){
        swarm[i] = calloc(input_vec_dim,sizeof(float));
        velocity[i] = calloc(input_vec_dim,sizeof(float));
        Pbest_vector[i] = calloc(input_vec_dim,sizeof(float));
        
        for(int j=0;j<input_vec_dim;j++){
            swarm[i][j] = lower_bound + rand()*(upper_bound-lower_bound)/RAND_MAX;
            Pbest_vector[i][j] = swarm[i][j];
        }

        personal_best[i] = optim_func(swarm[i]);
        if(global_best>personal_best[i]){
            global_best = personal_best[i];
            for(int k=0;k<input_vec_dim;k++){
                Gbest_vector[k] = swarm[i][k];
            }
        }
    }
    Swarm s;
    s.input_dim = input_vec_dim;
    s.swarm_size = swarm_size;
    s.swarm = swarm;
    s.Pbest_vector = Pbest_vector;
    s.upper_bound = upper_bound;
    s.lower_bound = lower_bound;
    s.vel = velocity;
    s.global_best = global_best;
    s.personal_best = personal_best;
    s.f = optim_func;
    s.Gbest_vector = Gbest_vector;
    
    return s;
}

void optimize(Swarm* swarm, int n_iter,float c1,float c2,float w,float chi){ 
    float u1,u2, _delV,_vals,vel;
    float delV = (*swarm).upper_bound-(*swarm).lower_bound;
    for(int iter=0;iter<n_iter;iter++){
        if(iter%100==0){
        printf(" %d of %d with Global best of %f ",iter,n_iter,swarm->global_best);
        fputc('\r',stdout);
        }
        _delV = (delV*iter)/n_iter;
        u1 = ((float)rand())/RAND_MAX;
        u2 = ((float)rand())/RAND_MAX;
        int i,j,k;

        #pragma omp parallel num_threads(n_threads)
        #pragma omp parallel for
        for(i=0;i<(*swarm).swarm_size;i++){
                for(j=0;j<(*swarm).input_dim;j++){
                    (*swarm).vel[i][j] = chi*( ((*swarm).vel[i][j])*w + (u1*c1)*(((*swarm).Pbest_vector[i][j])-((*swarm).swarm[i][j])) + (u2*c2)*(((*swarm).Gbest_vector[j])-((*swarm).swarm[i][j])) );
                    (*swarm).vel[i][j] = (*swarm).vel[i][j]>_delV?_delV:(*swarm).vel[i][j];
                    (*swarm).swarm[i][j] += (*swarm).vel[i][j];
                    (*swarm).swarm[i][j] = ((*swarm).swarm[i][j]>(*swarm).upper_bound)?(*swarm).upper_bound:(((*swarm).swarm[i][j]<(*swarm).lower_bound)?(*swarm).lower_bound:(*swarm).swarm[i][j]);
                }
                _vals = (*swarm).f((*swarm).swarm[i]);
                if(_vals<(*swarm).personal_best[i]){
                    (*swarm).personal_best[i] = _vals;
                    if(_vals<(*swarm).global_best){
                        
                        (*swarm).global_best = _vals;
                        for(k=0;k<(*swarm).input_dim;k++){
                            (*swarm).Pbest_vector[i][k] = (*swarm).swarm[i][k];
                            (*swarm).Gbest_vector[k] = (*swarm).swarm[i][k];
                        }
                    }
                    else{
                        for(k=0;k<(*swarm).input_dim;k++){
                            (*swarm).Pbest_vector[i][k] = (*swarm).swarm[i][k];
                        }
                    }  
                }
        }
    }
}

void save(Swarm swarm){
    char str[35];
    char ext[] = ".bin";
    float a = 3.44867894;
    sprintf(str,"opt/OptimizedSwarm_S%d_%f",n,swarm.global_best);
    strcat(str,ext);
    printf("Swarm saved to file :%s",str);
    FILE* file;
    file = fopen(str,"wb");
    if(file==NULL){
        printf("ERROR in opening file");
        exit(1);
    }
    size_t nwritten = fwrite(&swarm, sizeof(swarm), 1, file);
    fclose(file);
    if(nwritten<1){
        printf("ERROR in writing to file");
        exit(1);
    }
}

Swarm open(char* filename){
    FILE* file;
    size_t nread;
    file = fopen(filename,"rb");
    if(file==NULL){
        printf("ERROR in opening file");
        exit(1);
    }
    Swarm swarm;
    nread = fread(&swarm,sizeof(swarm),1,file);
    fclose(file);
    return swarm;
}


int get_n(){
    return (int)n;
}

float* GetGlobalBestCandidate(float c1, float c2, float w, float chi,size_t SwarmSize, 
                         float upper, float lower, int n_iter){
    printf("%f %f %f %f %d  %f %f %d\n",c1,c2,w,chi,SwarmSize,upper,lower,n_iter);

    size_t dim = n*3;
    Swarm swarm = init(BrennerPotential,SwarmSize,dim,upper,lower);
    //printf("global minima start :%f \n",swarm.global_best);
    float t1 = (float)gettime();
    
    optimize(&swarm,n_iter,c1,c2,w,chi);
    float t2 = (float)gettime();

    printf("\n total time elapsed %f ms\n",(double)(t2-t1)/1000);
    return swarm.Gbest_vector;
}