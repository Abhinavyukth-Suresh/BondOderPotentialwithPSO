


def setup(setup_=False):
    import os
    real_path = os.getcwd()
    path = os.path.dirname(__file__)
    os.chdir(path)
    try:
        os.remove('shared.so')
    except OSError as e:
        print(e)
        
    list_files = ['shared.c','Brenner.c','PSO.c','Brenner.h','PSO.h']
    list_C = [i for i in list_files if ".c" in i]
    dirs = os.listdir()
    print('\nCHECKING FILES AND RESOURCES')
    r = os.system('gcc --version')
    if r!=0:
        print('GNU COMPILER COLLECTION does not exist. try installing gcc/mingw')
        print('SETUP FAILED!')
        exit(1)
    print('checking for files in '+os.getcwd())
    if not 'shared.c' in dirs:
        print("FILE NOT FOUND ERROR: no file named shared.c")
        exit(1)

    print('collecting shared script from CWD')
    #SETTING UP CDLL
    print('BUILDING SHARED CDLL')
    print('compiling resources .. .. ')
    for i in list_files:
        if i not in dirs:
            print("No file named ",i," exists")
            exit(1)
        print('collecting data from file '+i)

    for i in list_C:
        print(f'compilie -c O3  -msse -mavx -mavx2 -o CDLL/{i[:-2]}.o {i}')
        os.system(f'gcc -c -O3  -msse -mavx -mavx2  -o CDLL/{i[:-2]}.o {i}')
    
    #complied_list = [str(i[:-2]+'.o') for i in list_C]
    complied_list = [str(i[:]) for i in list_C]
    cmp_str = ' '.join(complied_list)

    
    
    print('compile -shared -fPIC -O3  -msse -mavx -mavx2 -o CDLL/shared.so '+cmp_str)
    status = os.system('gcc -shared -fPIC -O3  -msse -mavx -mavx2 -o CDLL/shared.so '+cmp_str)
    if status != 0:
        print("SETUP FAILED")
        print('compilation error')
        exit(1)
    print('completed compilation!\n')
    #CHECKING COMPILED CDLL 
    print('TESTING ... ...')
    import ctypes
    import time
    import numpy as np

    cdll = ctypes.CDLL('CDLL/./shared.so')
    get_n = cdll.get_n
    get_n.restype = ctypes.c_int
    n = get_n()
    k = n*3
    optimizer = cdll.GetGlobalBestCandidate
    optimizer.restype = ctypes.POINTER(ctypes.c_float * k)
    flt_ = ctypes.c_float
    int_ = ctypes.c_int
    c1,c2,w,chi, = 0.15, 0.052, 0.3, 1.0,
    SwarmSize,upper,lower,n_iter = 10,15,0,1

    t = time.time()
    a = [i for i in optimizer(ctypes.c_float(c1),ctypes.c_float(c2),ctypes.c_float(w),ctypes.c_float(chi),
                            ctypes.c_size_t(SwarmSize),ctypes.c_float(upper),ctypes.c_float(lower),ctypes.c_int(n_iter)).contents]
    t2 = time.time()
    print(f"python wrapper based elapsed time from inital epoch : {t2-t}")
    b = np.array(a)

    if (b.dtype!=np.float64) and (b.dtype!=np.float32):
        print('SETUP UNSUCCESSFUL')
        print('non-homogenous return type.')
        exit(1)
    print('for documentation related issues, check sample.py')
    print('SETUP SUCCESSFUL')
    if 'sample.py' not in dirs:
        print('sample.py in not found')
    
    if setup_:
        if not input('do you want to view the documentation? y/n :').lower() == 'y':
            print('setup completed')
            print('exiting')

        else:
            from BrennerPSO import sample
            print('\nDOCUMENTATION')
            print(sample.__doc__)
            os.chdir(real_path)