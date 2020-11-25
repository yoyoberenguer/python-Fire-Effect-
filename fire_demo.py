try:
    import pygame
except ImportError:
    raise ImportError("\npygame library is missing on your system."
                      "\nTry: \n   C:\\pip inatall pygame on a window command prompt.")
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
    from FireEffect import make_palette, fire_texture24, fire_surface24
except ImportError:
    raise ImportError("\n<FireEffect> library is missing on your system or FireEffect.pyx is not cynthonized.")

width = 800
height = 400
width_2 = width // 2
height_2 = height // 2
# GENERATE THE SAMPLING FOR HALF WIDTH & HEIGHT TO SPEED UP THE PROCESS
palette, surf = make_palette(width_2, height_2 - 150, 4, 60, 1.7)
mask = numpy.full((width_2, height_2), 255, dtype=numpy.uint8)
buff = fire_texture24(width_2, height_2, 500, 3.95, palette, mask)
# ADJUST THE SURFACE (SMOOTHSCALE) TO WINDOW SIZE
i = 0
for image in buff:
    buff[i] = pygame.transform.smoothscale(image, (width, height))
    i += 1

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

    pygame.display.set_caption("JOYSTICK TESTER")

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
    palette, surf = make_palette(width_2, height_2 - 150, 4.0, 110, 1.5)
    mask = numpy.full((width_2, height_2), 255, dtype=numpy.uint8)
    fire = numpy.zeros((height, width), dtype=numpy.float32)
    empty_x2 = pygame.Surface((width, height)).convert()

    fire_sound = pygame.mixer.Sound("fire7s.wav")
    fire_sound.play(-1)
    while not STOP_GAME:

        # SCREEN.fill((58, 57, 57, 0))
        pygame.event.pump()
        for event in pygame.event.get():

            keys = pygame.key.get_pressed()

            if keys[pygame.K_F8]:
                pygame.display.flip()
                pygame.image.save(SCREEN, 'screenshot' + str(FRAME) + '.png')

            elif event.type == pygame.QUIT:
                print('Quitting')
                STOP_GAME = True

            if keys[pygame.K_ESCAPE]:
                STOP_GAME = True

        s, o = fire_surface24(width_2, height_2, 3.95, palette, mask, fire)
        pygame.transform.scale2x(s, empty_x2)
        SCREEN.blit(empty_x2, (0, 0))
        fire = o

        CLOCK.tick(60)

        pygame.display.flip()
        FRAME += 1

    pygame.quit()
