cmake_minimum_required(VERSION 3.16)
project(cable_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CABLE_BUILD_ROS2 "Build the ROS2 image detection node when ROS2 is available" ON)
option(CABLE_BUILD_BENCHMARK "Build the local benchmark executable" ON)
option(CABLE_ENABLE_HORIZON_BPU "Compile the Horizon BPU backend when hb_dnn.h is available" OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn)
find_package(Threads REQUIRED)

add_library(cable_detector STATIC
  src/cable_detector.cpp
  src/bpu_backend.cpp
)
target_include_directories(cable_detector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(cable_detector PUBLIC ${OpenCV_LIBS} Threads::Threads)
target_compile_options(cable_detector PRIVATE -O3 -ffast-math -fno-math-errno)
target_compile_definitions(cable_detector PUBLIC
  CABLE_SOURCE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"
  $<$<BOOL:${CABLE_ENABLE_HORIZON_BPU}>:CABLE_ENABLE_HORIZON_BPU=1>
)

if(CABLE_BUILD_BENCHMARK)
  add_executable(cable_benchmark src/benchmark.cpp)
  target_link_libraries(cable_benchmark PRIVATE cable_detector)
  target_compile_options(cable_benchmark PRIVATE -O3 -ffast-math -fno-math-errno)
endif()

if(CABLE_BUILD_ROS2)
  find_package(ament_cmake QUIET)
  find_package(rclcpp QUIET)
  find_package(sensor_msgs QUIET)
  find_package(std_msgs QUIET)
  find_package(cv_bridge QUIET)
  if(ament_cmake_FOUND AND rclcpp_FOUND AND sensor_msgs_FOUND AND std_msgs_FOUND AND cv_bridge_FOUND)
    add_executable(cable_detection_node src/ros2_node.cpp)
    ament_target_dependencies(cable_detection_node rclcpp sensor_msgs std_msgs cv_bridge)
    target_link_libraries(cable_detection_node cable_detector)
    target_compile_options(cable_detection_node PRIVATE -O3 -ffast-math -fno-math-errno)
    install(TARGETS cable_detection_node DESTINATION lib/${PROJECT_NAME})
    message(STATUS "ROS2 node enabled")
  else()
    message(STATUS "ROS2 dependencies not found; building standalone detector and benchmark")
  endif()
endif()

install(TARGETS cable_detector
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
if(CABLE_BUILD_BENCHMARK)
  install(TARGETS cable_benchmark RUNTIME DESTINATION bin)
endif()
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(PROGRAMS scripts/rdk_x5_run.sh scripts/run_benchmark.sh
  DESTINATION share/${PROJECT_NAME}/scripts
)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME cable_benchmark_smoke
    COMMAND cable_benchmark --frames 3 --warmup 1 --json ${CMAKE_CURRENT_BINARY_DIR}/smoke.json)
endif()

if(ament_cmake_FOUND)
  ament_package()
endif()
