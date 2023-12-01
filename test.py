import BrennerPSO as bp


if __name__ == '__main__':
    import ctypes
    import numpy as np
    import os

    n = 200
    k = n*3
    brenner = bp.cdll.GetGlobalBestCandidate #ctypes.CDLL('CDLL/./shared.so').GetGlobalBestCandidate
    brenner.restype = ctypes.POINTER(ctypes.c_float * k)

    c1,c2,w,chi, = 0.15, 0.052, 0.3, 1.0,
    SwarmSize,upper,lower,n_iter = 300,15,0,4
    print(c1,c2,w,chi,SwarmSize,upper,lower,n_iter)
    a = [i for i in brenner(ctypes.c_float(c1),ctypes.c_float(c2),ctypes.c_float(w),ctypes.c_float(chi),
                            ctypes.c_size_t(SwarmSize),ctypes.c_float(upper),ctypes.c_float(lower),ctypes.c_int(n_iter)).contents]

    b = np.array(a)
    print(b)
