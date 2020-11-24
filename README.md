# python-Fire-Effect-
Old school fire effect with Python/Cython

![alt text](https://github.com/yoyoberenguer/python-Fire-Effect-/blob/main/screenshot101.png)

## REQUIREMENT:
```
- python > 3.0
- numpy arrays
- pygame 
- Cython
- A compiler such visual studio, MSVC, CGYWIN setup correctly
  on your system.
  - a C compiler for windows (Visual Studio, MinGW etc) install on your system 
  and linked to your windows environment.
  Note that some adjustment might be needed once a compiler is install on your system, 
  refer to external documentation or tutorial in order to setup this process.
  e.g https://devblogs.microsoft.com/python/unable-to-find-vcvarsall-bat/
```
## BUILDING PROJECT:
```
In a command prompt and under the directory containing the source files
C:\>python setup_lights.py build_ext --inplace

If the compilation fail, refers to the requirement section and make sure cython 
and a C-compiler are correctly install on your system. 
```
## DEMO
```
Edit the file test_lights.py in your favorite python IDE and run it 
Or run fire_demo.py 

```
