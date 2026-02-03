from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext_modules = [
    Pybind11Extension(
        "pano_cpp",
        ["pano_cpp.cpp"],
        extra_compile_args=["-O3", "-fopenmp", "-march=native", "-ffast-math"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    name="pano_cpp",
    version="0.1.0",
    author="weizhuo2",
    description="Fast panorama generation using C++ with OpenMP",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
