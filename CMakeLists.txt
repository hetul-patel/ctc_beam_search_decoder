cmake_minimum_required(VERSION 3.15)
project(ctc_beam_search_decoder)

set(CMAKE_CXX_STANDARD  14)
set(CMAKE_CXX_FLAGS     "-O3")

###############################################################################
## file globbing ##############################################################
###############################################################################

# Add sources for library
set(sources ./src/ctc_beam_search_decoder.cpp)

# Source for cpp_demo usage
set(cpp_demo_sources ./cpp_demo/ctc_decoder_demo.cpp)
set(cpp_demo_bf_sources ./cpp_demo/ctc_decoder_demo_batch_first.cpp)

###############################################################################
## target definitions for shared library ######################################
###############################################################################

include_directories(./src Eigen)

add_library(ctc_beam_search_decoder SHARED ${sources})
set_property(TARGET ctc_beam_search_decoder PROPERTY POSITION_INDEPENDENT_CODE ON)

###############################################################################
## target definitions for example application #################################
###############################################################################

# Include directories for ctc_examples only

add_executable(cpp_demo ${cpp_demo_sources})
target_include_directories(cpp_demo PUBLIC ${CMAKE_SOURCE_DIR})
target_link_directories(cpp_demo PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(cpp_demo ctc_beam_search_decoder)

add_executable(cpp_demo_bf ${cpp_demo_bf_sources})
target_include_directories(cpp_demo_bf PUBLIC ${CMAKE_SOURCE_DIR})
target_link_directories(cpp_demo_bf PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(cpp_demo_bf ctc_beam_search_decoder)
