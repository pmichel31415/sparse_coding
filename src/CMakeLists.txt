CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(sparse_coding)

add_definitions(-std=gnu++11)

# For mlpack
find_package(LibXml2 2.6.0 REQUIRED)
include_directories(${LIBXML2_INCLUDE_DIR})
include_directories(${MLPACK_INCLUDE_DIR})

IF(APPLE)
	MESSAGE ( STATUS "Searching for X11..." )
	FIND_PACKAGE ( X11 REQUIRED )
	IF ( X11_FOUND )
	      INCLUDE_DIRECTORIES ( ${X11_INCLUDE_DIR} )
	      LINK_LIBRARIES ( ${X11_LIBRARIES} )
	      MESSAGE ( STATUS " X11_INCLUDE_DIR: " ${X11_INCLUDE_DIR} )
	      MESSAGE ( STATUS " X11_LIBRARIES: " ${X11_LIBRARIES} )
	ENDIF ( X11_FOUND )
ENDIF(APPLE)

ADD_EXECUTABLE(sparse_coding main.cpp CImg.h sparse_coding.h sparse_coding.cpp)
TARGET_LINK_LIBRARIES(sparse_coding ${LIBXML2_LIBRARIES} ${MLPACK_LIBRARY})
