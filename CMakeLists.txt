cmake_minimum_required(VERSION 3.5.0)
project(dr_mujoco)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()
# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(control_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

install(DIRECTORY
  launch
  world
  models
  DESTINATION share/${PROJECT_NAME}/
)

# file(GLOB_RECURSE SCRIPT_FILES
#   "scripts/*.py"
# )

install(PROGRAMS
  scripts/simulation.py
  scripts/ball_mover.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
