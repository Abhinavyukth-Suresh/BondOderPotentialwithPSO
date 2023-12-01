#include<stdio.h>
#include<math.h>
#include <sys/time.h>
#include<stdlib.h>
#include<omp.h>
#include<string.h>
#include "Brenner.h"
#include "PSO.h"

long gettime(){
	struct timeval currentTime;
	gettimeofday(&currentTime, NULL);
	return currentTime.tv_sec * (int)1e6 + currentTime.tv_usec;
}

/*************** PARTICLE SWARM OPTIMIZER  ****************/


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

        #pragma omp parallel num_threads(7)
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



