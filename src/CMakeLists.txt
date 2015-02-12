cmake_minimum_required(VERSION 2.7)

project(Imbs CXX C)

get_filename_component(PROJECT_ROOT "${PROJECT_SOURCE_DIR}" ABSOLUTE)

find_package(LibXml2 REQUIRED)
find_package(OpenCV REQUIRED)

if (OpenCV_FOUND)
	message(STATUS "Found OpenCV: ${OpenCV_INCLUDE_DIRS} (found version \"${OpenCV_VERSION}\")")
endif (OpenCV_FOUND)

# Build type configuration.
if (CMAKE_BUILD_TYPE)
	string(TOUPPER ${CMAKE_BUILD_TYPE} _build_type)
	set(CMAKE_BUILD_TYPE ${_build_type} CACHE STRING "" FORCE)
endif (CMAKE_BUILD_TYPE)

# Build type must be either DEBUG or RELEASE.
if (CMAKE_BUILD_TYPE STREQUAL "")
	message("No build type specified: forcing to RELEASE.")
	set(CMAKE_BUILD_TYPE "RELEASE" CACHE STRING "" FORCE)
endif (CMAKE_BUILD_TYPE STREQUAL "")

# Linux bitness type.
if (LINUX_BITNESS)
	string(TOUPPER ${LINUX_BITNESS} _linux_bitness)
	set(LINUX_BITNESS ${_linux_bitness} CACHE STRING "" FORCE)
else (LINUX_BITNESS)
	set(LINUX_BITNESS "" CACHE STRING "" FORCE)
endif (LINUX_BITNESS)

# Linux bitness type must be either 32-BIT or 64-BIT.
if (LINUX_BITNESS STREQUAL "")
	EXECUTE_PROCESS(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
	
	if (${ARCHITECTURE} STREQUAL "x86_64")
		message("No linux bitness type specified. Detected 64-bit architecture, forcing to 64-bit bitness compilation.")
		set(LINUX_BITNESS "64-BIT" CACHE STRING "" FORCE)
	elseif (${ARCHITECTURE} STREQUAL "i686")
		message("No linux bitness type specified. Detected 32-bit architecture, forcing to 32-bit bitness compilation.")
		set(LINUX_BITNESS "32-BIT" CACHE STRING "" FORCE)
	else (${ARCHITECTURE} STREQUAL "i686")
		message(FATAL_ERROR "Unknown architecture found. The library cannot be compiled under this architecture yet.")
	endif (${ARCHITECTURE} STREQUAL "x86_64")
endif (LINUX_BITNESS STREQUAL "")

add_definitions(-Wuninitialized)
add_definitions(-Wreturn-type)
add_definitions(-Wsign-compare)
add_definitions(-Wuninitialized)
add_definitions(-Wunused-function)
add_definitions(-Wunused-label)    
add_definitions(-Wunused-value) 
add_definitions(-Wunused-variable)

if (CMAKE_BUILD_TYPE STREQUAL "DEBUG")
	add_definitions(-ggdb)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -frounding-math")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frounding-math")
elseif (CMAKE_BUILD_TYPE STREQUAL "RELEASE")
	add_definitions(-O3)
else (CMAKE_BUILD_TYPE STREQUAL "DEBUG")
	message("Invalid build type specified: doing nothing.")
endif (CMAKE_BUILD_TYPE STREQUAL "DEBUG")

if (LINUX_BITNESS STREQUAL "32-BIT")
	add_definitions(-m32)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m32")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
elseif (LINUX_BITNESS STREQUAL "64-BIT")
	add_definitions(-m64)
	add_definitions(-fno-elide-constructors)
else (LINUX_BITNESS STREQUAL "32-BIT")
	message(FATAL_ERROR "Invalid linux bitness type specified.")
endif (LINUX_BITNESS STREQUAL "32-BIT")

set(LIBRARY_OUTPUT_PATH ../lib)

include_directories(${LIBXML2_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE Imbs_src "CameraModel/*.cpp" "Core/*.cpp" Utils/*.cpp)

add_library(imbs SHARED ${Imbs_src} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(imbs ${OpenCV_LIBS})

file(READ "$ENV{HOME}/.profile" contents)

# Convert file contents into a CMake list (where each element in the list is one line of the file)
string(REGEX REPLACE ";" "\\\\;" contents "${contents}")
string(REGEX REPLACE "\n" ";" contents "${contents}")

set(found false)

# Checking whether the environment variable has already been set.
foreach (line ${contents})
	string(REGEX MATCH "export Imbs_ROOT=${PROJECT_ROOT}" result "${line}")
	
	if ("${result}" STREQUAL "export Imbs_ROOT=${PROJECT_ROOT}")
		set(found true)
	endif ("${result}" STREQUAL "export Imbs_ROOT=${PROJECT_ROOT}")
endforeach(line)

if (NOT found)
	# Setting the environment variable.
	file(APPEND $ENV{HOME}/.profile "\nexport Imbs_ROOT=${PROJECT_ROOT}")
	
	# Adding the path of the Imbs library
	file(APPEND $ENV{HOME}/.bashrc "\nexport LD_LIBRARY_PATH=/usr/local/lib/Imbs:\${LD_LIBRARY_PATH}")
endif (NOT found)

set(CMAKE_INSTALL_PREFIX /usr/local/include)

# Headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/CameraModel DESTINATION Imbs FILES_MATCHING PATTERN "*.h*")
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/Core DESTINATION Imbs FILES_MATCHING PATTERN "*.h*")
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/Utils DESTINATION Imbs FILES_MATCHING PATTERN "*.h*")

# Shared object
install(TARGETS imbs LIBRARY DESTINATION ../lib/Imbs)

# Make uninstall
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake" IMMEDIATE @ONLY)
add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
