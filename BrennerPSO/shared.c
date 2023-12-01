#include<stdio.h>
#include<math.h>
#include <sys/time.h>
#include<stdlib.h>
#include<omp.h>
#include<string.h>
#include "Brenner.h"
#include "PSO.h"

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


