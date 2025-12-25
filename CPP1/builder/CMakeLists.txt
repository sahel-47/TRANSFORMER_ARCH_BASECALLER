cmake_minimum_required(VERSION 3.17)
project(MATRIX_INNER_ENGINE)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Python_EXECUTABLE "/home/sahel/anaconda3/envs/pybind_env/bin/python")

# Now find_package will be forced to accept this specific python
find_package(Python 3.12 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)
# --- FIX END ---

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Since we forced the correct Python above, this command will now work
# because it will ask the Anaconda python (which has nanobind) instead of System python.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)

list(APPEND CMAKE_PREFIX_PATH "${nanobind_ROOT}")

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(MATRIX_INNER_ENGINE constructor.cpp)