###cython: boundscheck=False, wraparound=False, nonecheck=False, optimize.use_switch=True

# COMPILATION
# C:\>python setup_project.py build_ext --inplace

# EXECUTABLE
# C:\>pip install pyinstaller
# C:\>pyinstaller --onefile fire_demo.spec

# CYTHON IS REQUIRED
try:
    cimport cython
    from cython.parallel cimport prange
except ImportError:
    raise ImportError("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")

try:
    import numpy
    from numpy import zeros, asarray, ndarray, uint32, uint8, float32
except ImportError:
    raise ImportError("\nNumpy library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")

# PYGAME IS REQUIRED
try:
    import pygame
    from pygame import Color, Surface, SRCALPHA, RLEACCEL, BufferProxy
    from pygame.surfarray import pixels3d, array_alpha, pixels_alpha, array3d
    from pygame.image import frombuffer

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")

try:
    from rand import randrange, randrangefloat
except ImportError:
    raise ImportError("\n<rand> library is missing on your system or rand.pyx is not cynthonized.")


try:
    from hsl cimport struct_hsl_to_rgb, rgb, rgba
except ImportError:
    raise ImportError("\n<hsl> library is missing on your system or hsl.pyx is not cynthonized.")


from libc.stdio cimport printf
from libc.stdlib cimport rand


# ---------------------- INTERFACE -----------------------------
# FUNCTION BELOW CAN BE ACCESS DIRECTLY FROM PYTHON CODE
cpdef make_palette(size: int, height: int, fh: float=0.25, fs: float=255.0, fl: float=2.0):
    return make_palette_c(size, height, fh, fs, fl)
# --------------------------------------------------------------

DEF OPENMP = True

if OPENMP == True:
    DEF THREAD_NUMBER = 8
else:
    DEF THREAD_NUMNER = 1

DEF SCHEDULE = 'static'
DEF ONE_360  = 1.0 / 360.0
DEF ONE_255  = 1.0 / 255.0

# Load C code
cdef extern from 'randnumber.c':
    float randRangeFloat(float lower, float upper)nogil
    int randRange(int lower, int upper)nogil


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline unsigned int rgb_to_int(int red, int green, int blue)nogil:
    """
    CONVERT RGB MODEL INTO A PYTHON INTEGER EQUIVALENT TO THE FUNCTION PYGAME MAP_RGB()
    
    :param red   : Red color value,  must be in range [0..255] 
    :param green : Green color value, must be in range [0..255]
    :param blue  : Blue color, must be in range [0.255]
    :return      : returns a positive python integer representing the RGB values(int32)
    """
    return 65536 * red + 256 * green + blue

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline rgb int_to_rgb(unsigned int n)nogil:
    """
    CONVERT A PYTHON INTEGER INTO A RGB COLOUR MODEL (UNSIGNED CHAR VALUES [0..255]).
    EQUIVALENT TO PYGAME UNMAP_RGB()
    
    :param n : positive integer value to convert 
    :return  : return a C structure rgb containing RGB values
    """
    cdef:
        rgb rgb_

    rgb_.r = n >> 16 & 255  # red int32
    rgb_.g = n >> 8 & 255   # green int32
    rgb_.b = n & 255        # blue int32
    return rgb_

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline rgba int_to_rgba(int n)nogil:
    """
    Type      Capacity
    Int16 -- (-32,768 to +32,767)
    Int32 -- (-2,147,483,648 to +2,147,483,647)
    Int64 -- (-9,223,372,036,854,775,808 to +9,223,372,036,854,775,807)

    CONVERT A PYTHON INTEGER INTO A RGBA COLOUR MODEL.
    EQUIVALENT TO PYGAME UNMAP_RGB()

    :param n : strictly positive integer value to convert (c int32)
    :return  : return a C structure containing RGBA values
    Integer value is unmapped into RGBA values (unsigned char type, [0 ... 255]
    """
    cdef:
        rgba rgba_

    rgba_.a = (n >> 24) & 255  # alpha
    rgba_.r = (n >> 16) & 255  # red
    rgba_.g = (n >> 8) & 255   # green
    rgba_.b = n & 255          # blue

    return rgba_


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline int rgba_to_int(unsigned char red, unsigned char green, unsigned char blue, unsigned char alpha)nogil:
    """
    Int16 -- (-32,768 to +32,767)
    Int32 -- (-2,147,483,648 to +2,147,483,647)
    Int64 -- (-9,223,372,036,854,775,808 to +9,223,372,036,854,775,807)
   
    CONVERT RGBA MODEL INTO A MAPPED PYTHON INTEGER (INT32) EQUIVALENT TO PYGAME MAP_RGB()
    OUTPUT INTEGER VALUE BETWEEN (-2,147,483,648 TO +2,147,483,647).
     
    :param red   : unsigned char; Red color must be in range[0 ...255]
    :param green : unsigned char; Green color value must be in range[0 ... 255]
    :param blue  : unsigned char; Blue color must be in range[0 ... 255]
    :param alpha : unsigned char; Alpha must be in range [0...255] 
    :return: returns a python integer (int32, see above for description) representing the RGBA values
    """
    return (alpha << 24) + (red << 16) + (green << 8) + blue


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef make_palette_c(int width, int height, float fh, float fs, float fl):
    """
    CREATE A PALETTE OF RGB COLORS (WIDTH X HEIGHT)
    
    e.g: 
        # below: palette of 256 colors & surface (width=256, height=50).
        # hue * 6, saturation = 255.0, lightness * 2.0
        palette, surf = make_palette(256, 50, 6, 255, 2)
        palette, surf = make_palette(256, 50, 4, 255, 2)
        
    :param width  : integer, Palette width
    :param height : integer, palette height
    :param fh     : float, hue factor
    :param fs     : float, saturation factor
    :param fl     : float, lightness factor
    :return       : Return a tuple ndarray type uint32 and pygame.Surface (width, height) 
    """
    assert width > 0, "Argument width should be > 0, got %s " % width
    assert height > 0, "Argument height should be > 0, got %s " % height

    cdef:
        unsigned int [:] palette    = ndarray(width, uint32)
        unsigned char [:, :, :] pal = ndarray((width, height, 3), dtype=uint8)
        int x, y
        float h, s, l
        rgb rgb_
        int ii = 0

    with nogil:
        for x in prange(width):
            h, s, l = <float>x * fh,  min(fs, 255.0), min(<float>x * fl, 255.0)
            rgb_ = struct_hsl_to_rgb(h * ONE_360, s * ONE_255, l * ONE_255)
            # build the palette (1d buffer int values)
            palette[x] = rgb_to_int(<int>(rgb_.r * 255.0),
                                    <int>(rgb_.g * 255.0),
                                    <int>(rgb_.b * 255.0 * 0.5))

        # Create a 3d array containing rgb values
        for x in range(width):
            rgb_ = int_to_rgb(palette[ii])
            for y in prange(height):
                pal[x, y, 0] = <unsigned char>rgb_.r
                pal[x, y, 1] = <unsigned char>rgb_.g
                pal[x, y, 2] = <unsigned char>rgb_.b
            ii += 1

    return asarray(palette), pygame.surfarray.make_surface(asarray(pal))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef fire_texture24(int width, int height, int frame, float factor, pal, mask):
    """
    CREATE AN ANIMATED FLAME EFFECT OF SIZES (WIDTH, HEIGHT).
    THE FLAME EFFECT DOES NOT CONTAINS ALPHA TRANSPARENCY 24 bits
    
    e.g:
        width  = 200
        height = 200
        palette, surf = make_palette(256, 50, 4, 255, 2)
        mask = numpy.full((width, height), 255, dtype=numpy.uint8)
        buff = fire_effect24(width, height, 1000, 3.95, palette, mask)
    
    
    :param width  : integer; max width of the effect
    :param height : integer; max height of the effect 
    :param frame  : integer; number of frames for the animation 
    :param factor : float; change the flame height, default is 3.95 
    :param pal    : define a color palette e.g make_palette(256, 50, 6, 255, 2)
    :param mask   : Ideally a black and white texture transformed into a 2d array shapes (w, h) 
                    black pixel will cancel the effect. 
                    The mask should have the exact same sizes than passed argument (width, height) 
    :return: Return a python list containing all the 24-bit surfaces.
    """

    assert isinstance(width, int), \
           "Argument width should be a python int, got %s " % type(width)
    assert isinstance(height, int), \
           "Argument height should be a python int, got %s " % type(height)
    assert isinstance(frame, int), \
           "Argument frame should be a python int, got %s " % type(frame)
    assert isinstance(factor, float), \
           "Argument factor should be a python float, got %s " % type(factor)
    assert isinstance(mask, ndarray), \
           "Argument mask should be a numpy.ndarray, got %s " % type(mask)


    if not frame > 0:
        raise ValueError('Argument frame should be > 0, %s ' % frame)

    if width == 0 or height == 0:
        raise ValueError('Image with incorrect dimensions '
                         '(width>0, height>0) got (width:%s, height:%s)' % (width, height))
    cdef:
        int w, h
    try:
        w, h = mask.shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    if width != w or height != h:
        raise ValueError('Incorrect mask dimensions '
                         'mask should be (width=%s, height=%s), '
                         'got (width=%s, height=%s)' %(width, height, w, h))
    cdef:
        float [:, ::1] fire = zeros((height, width), dtype=float32)
        # flame opacity palette
        unsigned int [::1] alpha = make_palette(256, 1, 1, 0, 2)[0]
        unsigned int [:, :, ::1] out = zeros((height, width, 3), dtype=uint32)
        unsigned int [::1] palette = pal
        unsigned char [:, :] mask_ = mask
        int x = 0, y = 0, i = 0, f
        float d
        rgb rgb_

    list_ = []


    for f in range(frame):
        for x in range(width):
            fire[height-1, x] = randrange(1, 255)

        with nogil:
            for y in prange(0, height - 1):
                for x in range(0, width - 1):
                    if mask_[x, y] != 0:
                        d = (fire[(y + 1) % height, (x - 1 + width) % width]
                                       + fire[(y + 1) % height, x % width]
                                       + fire[(y + 1) % height, (x + 1) % width]
                                       + fire[(y + 2) % height, x % width]) / factor
                        d -= rand() * 0.0001
                        if d > 255.0:
                            d = 255.0
                        if d < 0:
                            d = 0
                        fire[y, x] = d
                        rgb_ = int_to_rgb(palette[<unsigned int>d])
                        out[y, x, 0], out[y, x, 1], out[y, x, 2] = \
                            <unsigned char>rgb_.r, <unsigned char>rgb_.g, <unsigned char>rgb_.b
                    else:
                        out[y, x, 0], out[y, x, 1], out[y, x, 2] = 0, 0, 0
        surface = pygame.image.frombuffer(asarray(out, dtype=uint8), (width, height), 'RGB')
        list_.append(surface)
    return list_

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef fire_texture32(int width, int height, int frame, float factor, pal):
    """
    CREATE AN ANIMATED FLAME EFFECT OF SIZES (WIDTH, HEIGHT).
    THE FLAME EFFECT CONTAINS PER-PIXEL TRANSPARENCY
    
    e.g:
        width, height = 200, 200
        image = pygame.image.load("LOAD YOUR IMAGE")
        image = pygame.transform.smoothscale(image, (width, height))
     
        buff = fire_effect32(width, height, 1000, 3.95, palette)
    
    :param width: integer; max width of the effect
    :param height: integer; max height of the effect 
    :param frame: integer; number of frames for the animation 
    :param factor: float; change the flame height, default is 3.95 
    :param pal: define a color palette e.g make_palette(256, 50, 6, 255, 2)
   
    :return: Return a python list containing all the per-pixel surfaces.
    """

    assert isinstance(width, int), \
           "Argument width should be a python int, got %s " % type(width)
    assert isinstance(height, int), \
           "Argument height should be a python int, got %s " % type(height)
    assert isinstance(frame, int), \
           "Argument frame should be a python int, got %s " % type(frame)
    assert isinstance(factor, float), \
           "Argument factor should be a python float, got %s " % type(factor)


    if not frame > 0:
        raise ValueError('Argument frame should be > 0, %s ' % frame)

    if width == 0 or height == 0:
        raise ValueError('Image with incorrect dimensions '
                         '(width>0, height>0) got (width:%s, height:%s)' % (width, height))

    cdef:
        float [:, ::1] fire = zeros((height, width), dtype=float32)
        # flame opacity palette
        unsigned int [::1] alpha = make_palette(256, 1, 1, 0, 2)[0]
        unsigned int [:, :, ::1] out = zeros((height, width, 4), dtype=uint32)
        unsigned int [::1] palette = pal
        int x = 0, y = 0, i = 0, f
        float d
        rgb rgb_

    list_ = []


    for f in range(frame):
        for x in range(width):
            fire[height-1, x] = randrange(1, 255)

        with nogil:
            for y in prange(0, height - 1):
                for x in range(0, width - 1):
                        d = (fire[(y + 1) % height, (x - 1 + width) % width]
                                       + fire[(y + 1) % height, x % width]
                                       + fire[(y + 1) % height, (x + 1) % width]
                                       + fire[(y + 2) % height, x % width]) / factor
                        d -= rand() * 0.0001
                        if d > 255.0:
                            d = 255.0
                        if d < 0:
                            d = 0
                        fire[y, x] = d
                        rgb_ = int_to_rgb(palette[<unsigned int>d])
                        out[y, x, 0], out[y, x, 1], \
                        out[y, x, 2], out[y, x, 3]  = <unsigned char>rgb_.r, \
                            <unsigned char>rgb_.g, <unsigned char>rgb_.b,  alpha[<unsigned int>d]

        surface = pygame.image.frombuffer(asarray(out, dtype=uint8), (width, height), 'RGBA')
        list_.append(surface)
    return list_



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef fire_surface24(int width, int height, float factor, pal, float [:, ::1] fire):
    """
    
    :param width : integer; max width of the effect
    :param height: integer; max height of the effect
    :param factor: float; factor to reduce the flame effect
    :param pal   : ndarray; Color palette 1d numpy array (colors buffer unsigned int values) 
    :param fire  : ndarray; 2d array (x, y) (contiguous) containing float values 
    :return      : 
    """

    cdef:
        # flame opacity palette
        unsigned int [:, :, ::1] out = zeros((height, width, 3), dtype=uint32)
        unsigned int [::1] palette   = pal
        int x = 0, y = 0
        float d
        unsigned char r=0, g=0, b=0
        unsigned int ii=0
        unsigned c1 = 0, c2 = 0


    with nogil:
        for x in prange(width):
                fire[height - 1, x] = randRange(0, 255)


        for y in range(0, height - 1):
            for x in prange(0, width - 1):

                    c1 = (y + 1) % height
                    c2 = x % width
                    d = (fire[c1, (x - 1 + width) % width]
                       + fire[c1, c2]
                       + fire[c1, (x + 1) % width]
                       + fire[(y + 2) % height, c2]) * factor

                    d -= rand() * 0.0001

                    # Cap the values
                    if d > 255.0:
                        d = 255.0
                    if d < 0:
                        d = 0

                    fire[y, x] = d

                    ii = palette[<unsigned int>d]

                    r = (ii >> 16) & 255  # red int32
                    g = (ii >> 8) & 255   # green int32
                    b = ii & 255          # blue int32

                    out[y, x, 0], out[y, x, 1], out[y, x, 2] = r, g, b

    return asarray(out).transpose(1, 0, 2), fire

