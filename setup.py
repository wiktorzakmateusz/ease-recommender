import os
import sys
import numpy as np
from setuptools import setup, Extension

if sys.platform == "darwin":  # macOS
    brew_prefix = "/opt/homebrew" if os.path.exists("/opt/homebrew") else "/usr/local"
    
    include_dirs = [
        np.get_include(), 
        "src/c_ext", 
        os.path.join(brew_prefix, "include"),
        os.path.join(brew_prefix, "opt", "libomp", "include") 
    ]
    
    library_dirs = [
        os.path.join(brew_prefix, "lib"),
        os.path.join(brew_prefix, "opt", "libomp", "lib")     
    ]
    
    extra_compile_args = ["-O3", "-ffast-math", "-Xpreprocessor", "-fopenmp"]
    extra_link_args = ["-lomp"]

elif sys.platform == "win32":  # Windows
    include_dirs = [np.get_include(), "src/c_ext"]
    library_dirs = []
    
    gsl_root = os.environ.get("GSL_ROOT")
    if gsl_root:
        include_dirs.append(os.path.join(gsl_root, "include"))
        library_dirs.append(os.path.join(gsl_root, "lib"))
    
    extra_compile_args = ["/O2", "/fp:fast", "/openmp"]
    extra_link_args = []

else:  # Linux
    include_dirs = [np.get_include(), "src/c_ext"]
    library_dirs = []
    
    extra_compile_args = ["-O3", "-ffast-math", "-fopenmp"]
    extra_link_args = ["-fopenmp"]

ease_c_module = Extension(
    name="ease_core",
    sources=[
        "src/c_ext/ease_ext.c", 
        "src/c_ext/ease_custom_math.c",
        "src/c_ext/ease_gsl_math.c"
    ],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=["gsl", "gslcblas"],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    ext_modules=[ease_c_module],
)