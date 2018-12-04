cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/../IObjZoneDetect/include/)
include_directories(${PROJECT_SOURCE_DIR}/../IlstmCharsRecognize/include/)

#add_subdirectory(${PROJECT_SOURCE_DIR}/PlateAssist)

find_package(OpenCV 3.4)
find_library(Lstm_LIB IlstmCharsRecognize ${PROJECT_SOURCE_DIR}/../IlstmCharsRecognize/lib)
message(STATUS Found Lstm_LIB:  ${Lstm_LIB})
find_library(IObjZoneDetect_LIB IObjZoneDetect ${PROJECT_SOURCE_DIR}/../IObjZoneDetect/lib)
message(STATUS Found IObjZoneDetect_LIB:  ${IObjZoneDetect_LIB})

file(GLOB SRC src/*.cpp)
add_library(IPlate SHARED ${SRC})
target_link_libraries(IPlate ${OpenCV_LIBS} ${Lstm_LIB} ${IObjZoneDetect_LIB})
#add_library(_VehiclePlate SHARED python/PyIPlateRecognize.cpp)
#target_link_libraries(_VehiclePlate IPlateRecognize ${Boostpython_LIB})

add_executable(main IPlateTest.cpp)
target_link_libraries(main IPlate)

