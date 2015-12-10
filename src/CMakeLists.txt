CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(sparse_coding)

add_definitions(-std=gnu++11)

ADD_EXECUTABLE(sparse_coding main.cpp CImg.h)