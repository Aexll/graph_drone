from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "graphx",
        ["graphx.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++"
    ),
]

setup(
    name="graphx",
    ext_modules=ext_modules,
)
