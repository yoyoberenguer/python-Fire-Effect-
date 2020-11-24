###cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, optimize.use_switch=True
try:
    cimport cython
except ImportError:
    print("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")

# EXTERNAL C CODE
cdef extern from 'randnumber.c':

    struct vector2d:
       float x;
       float y;

    float randRangeFloat(float lower, float upper)nogil
    int randRange(int lower, int upper)nogil

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
# RETURN RANDOMIZE INTEGER
cpdef randrange(a, b):
    return randRange(a, b)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
# RETURN RANDOMIZE FLOATING NUMBER
cpdef randrangefloat(a, b):
    return randRangeFloat(a, b)