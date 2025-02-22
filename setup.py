from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    ext_modules=[
        Pybind11Extension(
            "hackablegrid._core",
            ["cpp/bindings.cpp", "cpp/src/vec_env.cpp"],
            include_dirs=["cpp/include"],
            cxx_std=17,
            extra_compile_args=["-O3"],
        )
    ],
    cmdclass={"build_ext": build_ext},
)
