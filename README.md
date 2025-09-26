# DRmujoco
ROS2-Mujoco simulator for differential drive robot.

This repository should be excutable as one independent simulator. Any other components put on different repository. 

The documentation page is here: https://hmmt.ee/dr_mujoco/

## Major update and note comparing to Gazebo ver. 

**URDF syntax**

Mujoco require the specific syntax for robot modeling. It is almost same as URDF in Gazebo, but bit different. 

Ref. https://mujoco.readthedocs.io/en/stable/modeling.html

**Dynamics and low-level Control**
In gazebo, we were using plugin for low-layer wheel control implementation. In mujoco, we implement low-level controller as scratch (typical wheel-size and gap model base) but dynamics are calculated by mujoco physics engine.

**CMake**
Possibilities of some components switching to Cpp implementation due to the calculation speed, CMake is used for build system.
