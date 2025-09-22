# DRmujoco
ROS2-Mujoco simulator for differential robot.

This repository should be excutable as one independent simulator. Any other components put on different repository. 

## Major update and note comparing to Gazebo ver. 

**URDF syntax**

Mujoco require the specific syntax for robot modeling. It is almost same as URDF in Gazebo, but bit different. 

Ref. https://mujoco.readthedocs.io/en/stable/modeling.html

**Dynamics and low-level Control**
In gazebo, we were using plugin for low-layer wheel control implementation. In mujoco, we implement low-level controller as scratch (typical wheel-size and gap model base) but dynamics are calculated by mujoco physics engine.

**CMake**
After ubuntu 24.04 and ROS2 Jazzy, setup.py is not recommended option. The ros2 package is managed with CMake now. 
