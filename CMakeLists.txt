cmake_minimum_required(VERSION 3.15)
project(ctc_beam_search_decoder)

set(CMAKE_CXX_STANDARD  14)
# set(CMAKE_CXX_FLAGS     "-march=native")

###############################################################################
## file globbing ##############################################################
###############################################################################

# Add sources for library
set(sources ./src/ctc_beam_search_decoder.cpp)

# Source for cpp_demo usage
set(cpp_demo_sources ./cpp_demo/ctc_decoder_demo.cpp)

###############################################################################
## target definitions for shared library ######################################
###############################################################################

include_directories(./src Eigen)

add_library(ctc_beam_search_decoder SHARED ${sources})
set_property(TARGET ctc_beam_search_decoder PROPERTY POSITION_INDEPENDENT_CODE ON)

###############################################################################
## target definitions for example application #################################
###############################################################################

add_executable(cpp_demo ${cpp_demo_sources})

# Include directories for ctc_example only
target_include_directories(cpp_demo PUBLIC ${CMAKE_SOURCE_DIR})
target_link_directories(cpp_demo PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(cpp_demo ctc_beam_search_decoder)
