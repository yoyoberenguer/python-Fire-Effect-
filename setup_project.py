from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules =\
[
  Extension("rand", ["rand.pyx"]),
  Extension("hsv", ["hsv.pyx"]),
  Extension("hsl", ["hsl.pyx"]),
  Extension("FireEffect", ["FireEffect.pyx"]),


]

setup(
  name="FireEffect",
  cmdclass={"build_ext": build_ext},
  ext_modules=ext_modules
)