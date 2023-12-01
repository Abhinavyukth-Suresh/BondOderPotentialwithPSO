"""
PSO :
 struct SWARM :
     {
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
    }

 method 1:call -> init( ** ) 
  params -> float (*optim_func)(float*),size_t swarm_size,size_t input_vec_dim,float upper_bound,float lower_bound
  ret -> struct swarm [**initialized]

method 2: optimize ( ** )
 params -> Swarm* swarm, int n_iter,float c1,float c2,float w,float chi
 ret -> None [*void func]

 mehtod save -> save swarm <- *swarm
 method open -> import swarm from file <- char * filename

 method GetGlobalBestCandidate
 params -> float c1, float c2, float w, float chi,size_t SwarmSize, 
                         float upper, float lower, int n_iter
ret float* arr Global best of optimized swarm

"""
if __name__ == '__main__':
    import ctypes
    import numpy as np
    import os

    n = 200
    k = n*3
    brenner = ctypes.CDLL('CDLL/./shared.so').GetGlobalBestCandidate
    brenner.restype = ctypes.POINTER(ctypes.c_float * k)

    c1,c2,w,chi, = 0.15, 0.052, 0.3, 1.0,
    SwarmSize,upper,lower,n_iter = 100,15,0,4
    print(c1,c2,w,chi,SwarmSize,upper,lower,n_iter)
    a = [i for i in brenner(ctypes.c_float(c1),ctypes.c_float(c2),ctypes.c_float(w),ctypes.c_float(chi),
                            ctypes.c_size_t(SwarmSize),ctypes.c_float(upper),ctypes.c_float(lower),ctypes.c_int(n_iter)).contents]

    b = np.array(a)
    print(b)
