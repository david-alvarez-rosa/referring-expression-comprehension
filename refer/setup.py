"""Explanatory message.

To compile and install locally run "python setup.py build_ext --inplace" To
install library to Python site-packages run "python setup.py build_ext install"

"""


from setuptools import setup, Extension
import numpy as np


ext_modules = [
    Extension(
        "external._mask",
        sources=["external/maskApi.c", "external/_mask.pyx"],
        include_dirs=[np.get_include(), "external"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function", "-std=c99"],
    )
]

setup(
    name="external",
    packages=["external"],
    package_dir={"external": "external"},
    install_requires=[
        "setuptools>=18.0",
        "cython>=0.27.3",
        "matplotlib>=2.1.0"
    ],
    version="2.0",
    ext_modules=ext_modules
)
