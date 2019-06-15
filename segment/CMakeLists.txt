cmake_minimum_required(VERSION 2.8.3)
project(segment)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  tf2
  tf2_ros
  pcl_ros
)

find_package(Eigen3 REQUIRED)
find_package(OCTOMAP REQUIRED)
find_package(cilantro REQUIRED)
find_package(OpenMP REQUIRED)
# find_package(Boost REQUIRED)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "${OpenMP_CXX_FLAGS} -std=c++17 -Wall -O3 -Wextra -march=native -s")

catkin_package()

include_directories(
  include/${PROJECT_NAME}
  ${catkin_INCLUDE_DIRS}
  # ${Boost_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
  ${cilantro_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_node src/segment.cpp)

target_link_libraries(${PROJECT_NAME}_node
   ${catkin_LIBRARIES} ${OCTOMAP_LIBRARIES} ${cilantro_LIBRARIES} # ${Boost_LIBRARIES} # ${OpenCV_LIBS} 
)

