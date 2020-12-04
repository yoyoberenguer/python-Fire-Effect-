
# COMPILATION
# python setup_project.py build_ext --inplace

# INSTALLER
# pyinstaller --onefile fire_demo.spec

# Ver 1.0.2

__version__ = "1.0.2"

try:
    import pygame
except ImportError:
    raise ImportError("\npygame library is missing on your system."
                      "\nTry: \n   C:\\pip inatall pygame on a window command prompt.")

from pygame.display import flip, set_caption
from pygame.pixelcopy import array_to_surface
from pygame.transform import scale2x

try:
    import numpy
except ImportError:
    raise ImportError("\nnumpy library is missing on your system."
                      "\nTry: \n   C:\\pip install numpy on a window command prompt.")
import os
import sys

try:
    import platform
except ImportError:
    raise ImportError("\nplatform library is missing on your system."
                      "\nTry: \n   C:\\pip install platform on a window command prompt.")

try:
    from FireEffect import make_palette, fire_texture24, fire_surface24, fire_texture32
except ImportError:
    raise ImportError("\n<FireEffect> library is missing on your system or FireEffect.pyx is not cynthonized.")

try:
    from rand import randrange, randrangefloat
except ImportError:
    raise ImportError("\n<rand> library is missing on your system or rand.pyx is not cynthonized.")

width = 800
height = 400

# BELOW TESTING FOR METHOD fire_texture32
# width_2 = width // 2
# height_2 = height // 2
# # GENERATE THE SAMPLING FOR HALF WIDTH & HEIGHT TO SPEED UP THE PROCESS
#
# palette, surf = make_palette(width_2, height_2 - 150, 0.1, 60, 1.7)
# mask = numpy.full((width_2, height_2), 255, dtype=numpy.uint8)
# buff = fire_texture32(width_2, height_2, 500, 3.95, palette)
# # ADJUST THE SURFACE (SMOOTHSCALE) TO WINDOW SIZE
# i = 0
# for image in buff:
#     buff[i] = pygame.transform.smoothscale(image, (width, height))
#     i += 1


if __name__ == '__main__':

    pygame.mixer.init()
    pygame.init()


    try:
        SCREEN = pygame.display.set_mode((width, height))
    except pygame.error:
        try:
            os.environ["SDL_VIDEODRIVER"] = ""
            SCREEN = pygame.display.set_mode((width, height))
        except:
            raise Exception('\nCannot initialized pygame video mode!')

    SCREEN.set_alpha(None)

    pygame.display.set_caption("Fire Demo")

    print('Driver          : ', pygame.display.get_driver())
    print(pygame.display.Info())
    # New in pygame 1.9.5.
    try:
        print('Display(s)      : ', pygame.display.get_num_displays())
    except AttributeError:
        pass
    sdl_version = pygame.get_sdl_version()
    print('SDL version     : ', sdl_version)
    print('Pygame version  : ', pygame.version.ver)
    python_version = sys.version_info
    print('Python version  :  %s.%s.%s ' % (python_version[0], python_version[1], python_version[2]))
    print('Platform        : ', platform.version())
    print('Available modes : ', pygame.display.list_modes())

    CLOCK = pygame.time.Clock()
    STOP_GAME = False
    FRAME = 0

    # pygame.event.set_grab(True)
    # control the sharing of input devices with other applications
    # set_grab(bool) -> None
    # When your program runs in a windowed environment, it will share the mouse
    # and keyboard devices with other applications that have focus.
    # If your program sets the event grab to True, it will lock all input into your program.
    # It is best to not always grab the input, since it prevents the user from doing other things on their system.
    pygame.event.clear()

    width = 800
    height = 400
    width_2 = width // 2
    height_2 = height // 2

    # GENERATE THE SAMPLING FOR HALF WIDTH & HEIGHT TO SPEED UP THE PROCESS
    hue = 3.9
    saturation = 93.4
    luminescence = 1.2
    palette, surf = make_palette(width_2, height_2 - 150, hue, saturation, luminescence)

    mask = numpy.full((width_2, height_2), 255, dtype=numpy.uint8)
    fire = numpy.zeros((height, width), dtype=numpy.float32)
    fire_surface_small = pygame.Surface((width_2, height_2)).convert(32, pygame.RLEACCEL)
    fire_surface_x2    = pygame.Surface((width, height)).convert(32, pygame.RLEACCEL)

    fire_sound = pygame.mixer.Sound("firepit.wav")
    fire_sound.play(-1)

    # TWEAKS
    screen_blit = SCREEN.blit
    clock_tick  = CLOCK.tick
    cget_fps    = CLOCK.get_fps
    event_pump  = pygame.event.pump
    event_get   = pygame.event.get
    get_key     = pygame.key.get_pressed

    factor = 1.0/3.95

    while not STOP_GAME:

        # SCREEN.fill((58, 57, 57, 0))
        event_pump()
        keys = get_key()

        # SATURATION +
        if keys[pygame.K_KP_PLUS]:
            saturation += 0.1
            if saturation > 100.0:
                saturation = 100.0
            palette, surf = make_palette(width_2, height_2 - 150, hue, saturation, luminescence)
        # SATURATION -
        if keys[pygame.K_KP_MINUS]:
            saturation -= 0.1
            if saturation < 0.0:
                saturation = 0.0
            palette, surf = make_palette(width_2, height_2 - 150, hue, saturation, luminescence)

        for event in event_get():

            # LUMINESCENCE +
            if keys[pygame.K_LEFT]:
                luminescence += 0.1
                if luminescence > 2.0:
                    luminescence = 2.0
                palette, surf = make_palette(width_2, height_2 - 150, hue, saturation, luminescence)
            # LUMINESCENCE -
            if keys[pygame.K_RIGHT]:
                luminescence -= 0.1
                if luminescence < 0.0:
                    luminescence = 0.0
                palette, surf = make_palette(width_2, height_2 - 150, hue, saturation, luminescence)

            # HUE SHIFT +
            if keys[pygame.K_UP]:
                hue += 0.1
                if hue > 100.0:
                    hue = 100.0
                palette, surf = make_palette(width_2, height_2 - 150, hue, saturation, luminescence)
            # HUE SHIFT -
            if keys[pygame.K_DOWN]:
                hue -= 0.1
                if hue < 0.0:
                    hue = 0.0
                palette, surf = make_palette(width_2, height_2 - 150, hue, saturation, luminescence)

            if keys[pygame.K_F8]:
                flip()
                pygame.image.save(SCREEN, 'screenshot' + str(FRAME) + '.png')

            elif event.type == pygame.QUIT:
                print('Quitting')
                STOP_GAME = True

            if keys[pygame.K_ESCAPE]:
                STOP_GAME = True

        palette, surf = make_palette(width_2, height_2 + randrange(-150, 150), hue, saturation, luminescence)
        surface_, array_ = fire_surface24(width_2, height_2, factor + randrangefloat(-0.002, 0.002), palette, fire)

        array_to_surface(fire_surface_small, surface_)

        scale2x(fire_surface_small, fire_surface_x2)

        screen_blit(fire_surface_x2, (0, 0))
        fire = array_

        clock_tick(800)
        set_caption("Fire Demo (H :%s  S :%s   L :%s)  FPS %s " %
                    (round(hue, 2), round(saturation, 2), round(luminescence, 2), round(cget_fps(), 2)))

        flip()
        FRAME += 1

    pygame.quit()
