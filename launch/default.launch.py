import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('dr_mujoco')

    default_world_path = os.path.join(pkg_share, 'world', 'default.xml')
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

    show_frames_arg = DeclareLaunchArgument(
        'show_frames',
        default_value='false',
        description='Whether to show coordinate frames in the simulation'
    )

    alpha_arg = DeclareLaunchArgument(
        'robot_alpha',
        default_value='1.0',
        description='Alpha transparency for the robot model (0.0 to 1.0)'
    )

    mujoco_ros_bridge_node = Node(
        package='dr_mujoco',
        executable='simulation.py',
        name='simulation',
        output='screen',
        parameters=[{
            'world_path': LaunchConfiguration('world'),
            'robot_path': LaunchConfiguration('robot'),
            'show_frames': LaunchConfiguration('show_frames'),
            'robot_alpha': LaunchConfiguration('robot_alpha')
        }]
    )

    return LaunchDescription([
        world_arg,
        robot_arg,
        show_frames_arg,
        alpha_arg,
        mujoco_ros_bridge_node
    ])