cmake_minimum_required(VERSION 3.10)
project(ConvInpGen  LANGUAGES CXX)
set(
    pybind11_DIR
    "/home/sahel/anaconda3/envs/pybind_env/lib/python3.12/site-packages/pybind11/share/cmake/pybind11"
    CACHE PATH "Directory containing pybind11Config.cmake"
)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(ConvInpGen convinputbuffer.cpp)
install(TARGETS ConvInpGen  DESTINATION .)