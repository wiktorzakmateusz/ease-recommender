import sys
import os

# explicit adding GSL binary directory for Windows
if sys.platform == "win32" and sys.version_info >= (3, 8):
    # checks for a user-defined GSL_ROOT environment variable first, 
    # fallback to the default GitHub Actions/vcpkg installation path
    gsl_root = os.environ.get("GSL_ROOT", "C:\\vcpkg\\installed\\x64-windows")
    gsl_bin = os.path.join(gsl_root, "bin")
    
    if os.path.exists(gsl_bin):
        os.add_dll_directory(gsl_bin)

from .model import EASE

__version__ = "0.1.0"
__all__ = [
    "EASE",
]

