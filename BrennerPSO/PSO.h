#include<stdio.h>

#ifndef PSO_H
#define PSO_H

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


long gettime();
Swarm init(float (*optim_func)(float*),size_t,size_t,float,float);
void optimize(Swarm* swarm, int n_iter,float c1,float c2,float w,float chi);
void save(Swarm swarm);
Swarm open(char* filename);
#endif