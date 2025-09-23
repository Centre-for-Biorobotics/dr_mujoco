import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('dr_mujoco')

    default_world_path = os.path.join(pkg_share, 'world', 'object_detection.xml')
    default_robot_path = os.path.join(pkg_share, 'models', 'dummy.xml')

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=default_world_path,
        description='Path to the MuJoCo XML world file'
    )

    robot_arg = DeclareLaunchArgument(
        'robot',
        default_value=default_robot_path,
        description='Path to the MuJoCo XML robot file'
    )

    camera_hz_arg = DeclareLaunchArgument(
        'camera_hz',
        default_value='10.0',
        description='Camera update frequency in Hz'
    )

    mujoco_ros_bridge_node = Node(
        package='dr_mujoco',
        executable='simulation.py',
        name='simulation',
        output='screen',
        parameters=[{
            'world_path': LaunchConfiguration('world'),
            'robot_path': LaunchConfiguration('robot'),
            'camera_hz': LaunchConfiguration('camera_hz'),
            'camera_sim': True
        }]
    )

    ball_mover_node = Node(
        package='dr_mujoco',
        executable='ball_mover.py',
        name='ball_mover',
        output='screen'
    )

    return LaunchDescription([
        world_arg,
        robot_arg,
        camera_hz_arg,
        mujoco_ros_bridge_node,
        ball_mover_node
    ])