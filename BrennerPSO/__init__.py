import ctypes
import os
from  BrennerPSO import setup
import platform
import shutil

if platform.uname().system =="Windows":
    os.system('color 0a')
else:
    print("\033[1;32m hello world")

real_path = os.getcwd()
path = os.path.dirname(__file__)
os.chdir(path) 
listdir = os.listdir()

if not 'CDLL' in listdir:
    print('mkdir \\CDLL\\')
    os.mkdir('CDLL')
    
if 'shared.so' not in os.listdir(path+'/CDLL'):
    print('BUILDING UP DYNAMIC LIBRARIES')
    setup.setup()

def recompile(n=200):
    os.chdir(path + '/CDLL')
    print(os.listdir())
    os.chdir(path)
    print('recompiling resources')
    print('BUILDING UP DYNAMIC LIBRARIES')
    setup.setup(n=n)
    os.chdir(real_path)

cdll = ctypes.CDLL('CDLL/./shared.so')
os.chdir(real_path)  

get_n = cdll.get_n
get_n.restype = ctypes.c_int
n = get_n()
k = n*3
GetGlobalBestCandidate = cdll.GetGlobalBestCandidate
GetGlobalBestCandidate.restype = ctypes.POINTER(ctypes.c_float * k)
if platform.uname().system =="Windows":
    os.system('color 07')
else:
    print("\033[1;37;40m \n")