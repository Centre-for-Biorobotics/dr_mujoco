#!/usr/bin/env python3
"""
A ROS 2 node to bridge the MuJoCo physics simulator with ROS 2.
"""

import os
import xml.etree.ElementTree as ET
from typing import Optional
import random 
import math  

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int32 


class MujocoRosBridge(Node):
    """
    Bridges the MuJoCo simulation to ROS 2 topics and controls the
    simulation from ROS 2 topics.
    """

    # --- Constants ---
    # ROS Parameters
    PARAM_WORLD_PATH = 'world_path'
    PARAM_ROBOT_PATH = 'robot_path'
    CAMERA_SIM = 'camera_sim'
    CAMERA_SIM_HZ = 'camera_sim_hz'
    RENDERING_SIM_HZ = 'rendering_sim_hz'
    PARAM_ENCODER_CPR = 'encoder.cpr'
    PARAM_ENCODER_NOISY = 'encoder.noisy'
    PARAM_SHOW_FRAMES = 'show_frames'
    PARAM_FRAMES_KIND = 'frames_kind'
    PARAM_ROBOT_ALPHA = 'robot_alpha'

    # Topic Names
    ODOM_TOPIC = '/diff_cont/odom'
    IMAGE_TOPIC = '/camera1/image_raw'
    CMD_VEL_TOPIC = '/diff_cont/cmd_vel'
    DIRECT_VEL_UPDATE_TOPIC = '/move/cmd_vel'
    BALL_POS_TOPIC = '/ball_position'
    LEFT_ENCODER_TOPIC = '/left_encoder/ticks'
    RIGHT_ENCODER_TOPIC = '/right_encoder/ticks'


    # Frame IDs
    ODOM_FRAME_ID = 'odom'
    BASE_LINK_FRAME_ID = 'base_link'
    CAMERA_FRAME_ID = 'front_camera_frame'

    # Names in MuJoCo Model
    LEFT_WHEEL_GEOM = 'left_wheel_geom'
    LEFT_WHEEL_BODY = 'left_wheel'
    RIGHT_WHEEL_BODY = 'right_wheel'
    FRONT_CAMERA = 'front_camera'
    BALL_ACTUATOR = 'ball_actuator'
    LEFT_WHEEL_JOINT = 'base_to_left_wheel'
    RIGHT_WHEEL_JOINT = 'base_to_right_wheel'
    LEFT_WHEEL_ACTUATOR = 'left_wheel_vel'
    RIGHT_WHEEL_ACTUATOR = 'right_wheel_vel'

    # Simulation Settings
    PHYSICS_LOOP_HZ = 100.0 # Hz for mj_step and odom
    RENDERING_HZ = 30.0     # Hz for viewer and camera image
    CAMERA_HZ = 10.0

    IMG_WIDTH = 640
    IMG_HEIGHT = 480

    def __init__(self):
        """Initializes the node."""
        super().__init__('mujoco_ros_bridge')

        # --- Initialization ---
        try:
            self._setup_parameters()
            self._load_and_merge_models()
            self._get_model_parameters()
            self._setup_mujoco_simulation()
            self._setup_ros_communications()
        except Exception as e:
            self._handle_fatal_error(f"Failed to initialize: {e}")
            return

        self.get_logger().info("Mujoco ROS Bridge has been started successfully.")

    def _setup_parameters(self):
        """Declares and retrieves ROS parameters."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_world_path = os.path.join(script_dir, "world/default.xml")
        default_robot_path = os.path.join(script_dir, "models/dummy.xml")

        self.declare_parameter(self.PARAM_WORLD_PATH, default_world_path)
        self.declare_parameter(self.PARAM_ROBOT_PATH, default_robot_path)
        self.declare_parameter(self.CAMERA_SIM, False)
        self.declare_parameter(self.RENDERING_SIM_HZ, self.RENDERING_HZ)
        self.declare_parameter(self.CAMERA_SIM_HZ, self.CAMERA_HZ)
        self.declare_parameter(self.PARAM_ENCODER_CPR, 508.8)
        self.declare_parameter(self.PARAM_ROBOT_ALPHA, 1.0)
        self.declare_parameter(self.PARAM_ENCODER_NOISY, True)    
        self.declare_parameter(self.PARAM_SHOW_FRAMES, False)
        self.declare_parameter(self.PARAM_FRAMES_KIND, 'geom')  
        # body / geom / site / camera / light / contact / world

        self.world_path = self.get_parameter(self.PARAM_WORLD_PATH).get_parameter_value().string_value
        self.robot_path = self.get_parameter(self.PARAM_ROBOT_PATH).get_parameter_value().string_value
        self.camera_sim = self.get_parameter(self.CAMERA_SIM).get_parameter_value().bool_value
        self.rendering_hz = self.get_parameter(self.RENDERING_SIM_HZ).get_parameter_value().double_value
        self.camera_hz = self.get_parameter(self.CAMERA_SIM_HZ).get_parameter_value().double_value
        self.encoder_cpr = self.get_parameter(self.PARAM_ENCODER_CPR).get_parameter_value().double_value
        self.encoder_noisy = self.get_parameter(self.PARAM_ENCODER_NOISY).get_parameter_value().bool_value
        self.robot_alpha = self.get_parameter(self.PARAM_ROBOT_ALPHA).get_parameter_value().double_value
        self.show_frames = self.get_parameter('show_frames').get_parameter_value().bool_value
        self.frames_kind = self.get_parameter('frames_kind').get_parameter_value().string_value

        self.get_logger().info(f"World Path: {self.world_path}")
        self.get_logger().info(f"Robot Path: {self.robot_path}")
        self.get_logger().info(f"Run Camera Simulation: {self.camera_sim}")
        if self.camera_sim:
            self.get_logger().info(f"Camera Simulating Hz: {self.camera_hz}")
        self.get_logger().info(f"Rendering Hz: {self.rendering_hz}")
        self.get_logger().info(f"Encoder CPR: {self.encoder_cpr}")
        self.get_logger().info(f"Encoder Noisy: {self.encoder_noisy}")


        if not self.world_path or not self.robot_path:
            raise ValueError("Parameters 'world_path' and 'robot_path' must be set.")

    def _load_and_merge_models(self):
        """Merges and loads the world and robot XML models."""
        self.get_logger().info("Loading and merging XML models...")
        combined_xml_string = self._create_combined_xml_string(self.world_path, self.robot_path)
        self.model = mujoco.MjModel.from_xml_string(combined_xml_string)
        self.data = mujoco.MjData(self.model)
        self.get_logger().info(f"Models loaded successfully: World='{os.path.basename(self.world_path)}', Robot='{os.path.basename(self.robot_path)}'")
        # Test camera
        try:
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.FRONT_CAMERA)
        except Exception as e:
            self.get_logger().info(f"Camera '{self.FRONT_CAMERA}' not found in the model: {e}")
            raise ValueError(f"Camera '{self.FRONT_CAMERA}' not found in the model: {e}") from e

    def _get_model_parameters(self):
        """Retrieves physical parameters of the robot from the MuJoCo model."""
        try:
            left_wheel_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.LEFT_WHEEL_GEOM)
            self.wheel_radius = self.model.geom_size[left_wheel_geom_id][0]

            left_wheel_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.LEFT_WHEEL_BODY)
            right_wheel_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.RIGHT_WHEEL_BODY)
            left_pos = self.model.body_pos[left_wheel_body_id]
            right_pos = self.model.body_pos[right_wheel_body_id]
            self.wheel_base = np.linalg.norm(left_pos - right_pos)

            left_wheel_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.LEFT_WHEEL_JOINT)
            right_wheel_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.RIGHT_WHEEL_JOINT)

            self.left_wheel_qpos_id = self.model.jnt_qposadr[left_wheel_joint_id]
            self.right_wheel_qpos_id = self.model.jnt_qposadr[right_wheel_joint_id]

            self.get_logger().info(f"Loaded parameters from XML: Wheel Radius={self.wheel_radius:.4f}, Wheel Base={self.wheel_base:.4f}")
            self.get_logger().info(f"QPos index for left wheel: {self.left_wheel_qpos_id}")
            self.get_logger().info(f"QPos index for right wheel: {self.right_wheel_qpos_id}")

        except KeyError as e:
            raise RuntimeError(f"Could not find model component: {e}. Check your XML file.") from e
        except ValueError as e:
            raise RuntimeError(f"Could not find joint for encoder: {e}. Check your robot XML file.") from e


    def _setup_mujoco_simulation(self):
        """Sets up the MuJoCo viewer and renderer."""
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, self.IMG_HEIGHT, self.IMG_WIDTH)
        self.ball_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.BALL_ACTUATOR)

        if self.show_frames:
            kind_map = {
                'body':   mujoco.mjtFrame.mjFRAME_BODY,
                'geom':   mujoco.mjtFrame.mjFRAME_GEOM,
                'site':   mujoco.mjtFrame.mjFRAME_SITE,
                'camera': mujoco.mjtFrame.mjFRAME_CAMERA,
                'light':  mujoco.mjtFrame.mjFRAME_LIGHT,
                'contact':mujoco.mjtFrame.mjFRAME_CONTACT,
                'world':  mujoco.mjtFrame.mjFRAME_WORLD,
            }
            self.viewer.opt.frame = kind_map.get(self.frames_kind.lower(), mujoco.mjtFrame.mjFRAME_BODY)

            try:
                self.model.vis.scale.framelength = 0.05
                self.model.vis.scale.framewidth  = 0.001
            except AttributeError:
                pass

        if  0.0 <= self.robot_alpha <= 1.0:
            for mid in range(self.model.nmat):
                self.model.mat_rgba[mid, 3] = self.robot_alpha
            for gid in range(self.model.ngeom):
                if self.model.geom_matid[gid] < 0:
                    self.model.geom_rgba[gid, 3] = self.robot_alpha

        try:
            self.ball_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.BALL_ACTUATOR)
            self.left_wheel_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.LEFT_WHEEL_ACTUATOR)
            self.right_wheel_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.RIGHT_WHEEL_ACTUATOR)

            if self.left_wheel_actuator_id == -1 or self.right_wheel_actuator_id == -1:
                raise ValueError("Wheel actuators not found.")

            self.get_logger().info(f"Found left wheel actuator with ID: {self.left_wheel_actuator_id}")
            self.get_logger().info(f"Found right wheel actuator with ID: {self.right_wheel_actuator_id}")
            if self.ball_actuator_id != -1:
                self.get_logger().info(f"Found ball actuator with ID: {self.ball_actuator_id}")
        except ValueError as e:
            self.get_logger().warn(f"Could not find required actuators: {e}. Check actuator names in your XML and the script.")

        # Rendering settings
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

    def _setup_ros_communications(self):
        """Sets up ROS 2 Publishers, Subscribers, and Timers."""
        self.odom_pub = self.create_publisher(Odometry, self.ODOM_TOPIC, 10)
        self.image_pub = self.create_publisher(Image, self.IMAGE_TOPIC, 10)
        self.cmd_vel_sub = self.create_subscription(Twist, self.CMD_VEL_TOPIC, self.cmd_vel_callback, 10)
        self.direct_vel_sub = self.create_subscription(Twist, self.DIRECT_VEL_UPDATE_TOPIC, self.direct_cmd_vel_callback, 10)
        self.ball_pos_sub = self.create_subscription(Float32, self.BALL_POS_TOPIC, self.move_ball_callback, 10)

        # Encoder publishers
        self.left_encoder_pub = self.create_publisher(Int32, self.LEFT_ENCODER_TOPIC, 10)
        self.right_encoder_pub = self.create_publisher(Int32, self.RIGHT_ENCODER_TOPIC, 10)

        self.bridge = CvBridge()
        
        # Create a high-frequency timer for physics and odometry
        physics_period = 1.0 / self.PHYSICS_LOOP_HZ
        self.physics_timer = self.create_timer(physics_period, self.physics_timer_callback)

        # Create a lower-frequency timer for rendering
        rendering_period = 1.0 / self.rendering_hz
        self.rendering_timer = self.create_timer(rendering_period, self.rendering_timer_callback)
        if self.camera_sim:
            self.camera_timer = self.create_timer(1.0 / self.camera_hz, self.camera_timer_callback)


    def _handle_fatal_error(self, message: str):
        """Logs a fatal error and shuts down the node."""
        self.get_logger().fatal(message)
        self.get_logger().fatal("[FOR STUDENT] Check your launch file and model xml.")
        # Schedule shutdown and allow current execution to finish
        if rclpy.ok():
            rclpy.shutdown()

    @staticmethod
    def _create_combined_xml_string(world_path: str, robot_path: str) -> str:
        world_tree = ET.parse(world_path)
        robot_tree = ET.parse(robot_path)
        world_root = world_tree.getroot()
        robot_root = robot_tree.getroot()
        world_body = world_root.find('worldbody')
        robot_body = robot_root.find('worldbody')
        if world_body is not None and robot_body is not None:
            for body in list(robot_body):
                world_body.append(body)
        sections_to_merge = ['asset', 'visual', 'actuator', 'sensor', 'tendon', 'keyframe']
        for section in sections_to_merge:
            world_section = world_root.find(section)
            robot_section = robot_root.find(section)
            if robot_section is not None:
                if world_section is None:
                    world_root.append(robot_section)
                else:
                    for element in list(robot_section):
                        world_section.append(element)
        return ET.tostring(world_root, encoding='unicode')

    def direct_cmd_vel_callback(self, msg: Twist):
        """
        Directly sets the robot's linear and angular velocity in the world frame.
        """
        orientation_quat = self.data.qpos[3:7]

        linear_vel_body = np.array([msg.linear.x, msg.linear.y, msg.linear.z])
        angular_vel_body = np.array([msg.angular.x, msg.angular.y, msg.angular.z])

        linear_vel_world = np.zeros(3)
        angular_vel_world = np.zeros(3)

        mujoco.mju_rotVecQuat(linear_vel_world, linear_vel_body, orientation_quat)
        mujoco.mju_rotVecQuat(angular_vel_world, angular_vel_body, orientation_quat)

        self.data.qvel[0:3] = linear_vel_world
        self.data.qvel[3:6] = angular_vel_world


    def cmd_vel_callback(self, msg: Twist):
        linear_x = msg.linear.x
        angular_z = msg.angular.z
        v_right = -linear_x - (self.wheel_base / 2.0) * angular_z
        v_left = linear_x - (self.wheel_base / 2.0) * angular_z
        omega_right = v_right / self.wheel_radius
        omega_left = v_left / self.wheel_radius
        if self.left_wheel_actuator_id != -1:
            self.data.ctrl[self.left_wheel_actuator_id] = omega_left
        if self.right_wheel_actuator_id != -1:
            self.data.ctrl[self.right_wheel_actuator_id] = omega_right
        if self.right_wheel_actuator_id == -1 or self.left_wheel_actuator_id == -1:
            self.get_logger().warn_once("Wheel actuators not found, cannot apply cmd_vel.")
            self.get_logger().warn_once("Check actuator names in your XML and the script.")

    def move_ball_callback(self, msg: Float32):
        if self.ball_actuator_id != -1:
            self.data.ctrl[self.ball_actuator_id] = msg.data

    def physics_timer_callback(self):
        """Main loop for physics simulation and high-frequency data."""
        if not rclpy.ok():
            self.get_logger().info("Shutdown detected in physics loop.")
            return

        mujoco.mj_step(self.model, self.data)

        if not self.viewer.is_running():
            self.get_logger().info("Viewer closed, shutting down node.")
            rclpy.shutdown()
            return

        now = self.get_clock().now().to_msg()
        self._publish_odometry(now)
        self._publish_encoder_ticks() ### ADDED ### Call encoder publisher


    def rendering_timer_callback(self):
        if not rclpy.ok() or not self.viewer.is_running():
            return
        self.viewer.sync()


    def camera_timer_callback(self):
        if not rclpy.ok() or not self.viewer.is_running():
            return
        now = self.get_clock().now().to_msg()
        self._publish_camera_image(now)


    def _publish_odometry(self, stamp: Time):
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = self.ODOM_FRAME_ID
        odom_msg.child_frame_id = self.BASE_LINK_FRAME_ID
        odom_msg.pose.pose.position = Point(x=self.data.qpos[0], y=self.data.qpos[1], z=self.data.qpos[2])
        odom_msg.pose.pose.orientation = Quaternion(w=self.data.qpos[3], x=self.data.qpos[4], y=self.data.qpos[5], z=self.data.qpos[6])
        odom_msg.twist.twist.linear = Vector3(x=self.data.qvel[0], y=self.data.qvel[1], z=self.data.qvel[2])
        odom_msg.twist.twist.angular = Vector3(x=self.data.qvel[3], y=self.data.qvel[4], z=self.data.qvel[5])
        self.odom_pub.publish(odom_msg)


    def _publish_camera_image(self, stamp: Time):
        try:
            self.renderer.update_scene(self.data, camera=self.FRONT_CAMERA)
            pixels = self.renderer.render()
            bgr_image = pixels[..., ::-1]
            image_msg = self.bridge.cv2_to_imgmsg(bgr_image, "bgr8")
            image_msg.header.stamp = stamp
            image_msg.header.frame_id = self.CAMERA_FRAME_ID
            self.image_pub.publish(image_msg)
        except Exception as e:
            self.get_logger().warn(f"Could not publish camera image: {e}")

    def _publish_encoder_ticks(self):
        """Calculates and publishes absolute encoder ticks from joint states."""
        cpr_int = int(self.encoder_cpr)
        if cpr_int <= 0:
            self.get_logger().warn_once("Encoder CPR is not a positive value, cannot calculate ticks.")
            return

        left_wheel_angle = self.data.qpos[self.left_wheel_qpos_id]
        right_wheel_angle = self.data.qpos[self.right_wheel_qpos_id]

        cumulative_left_ticks = (left_wheel_angle / (2 * math.pi)) * self.encoder_cpr
        cumulative_right_ticks = (right_wheel_angle / (2 * math.pi)) * self.encoder_cpr

        if self.encoder_noisy:
            cumulative_left_ticks = random.normalvariate(cumulative_left_ticks, 0.75)
            cumulative_right_ticks = random.normalvariate(cumulative_right_ticks, 0.75)

        absolute_left_ticks = int(cumulative_left_ticks) % cpr_int
        absolute_right_ticks = int(cumulative_right_ticks) % cpr_int
        
        # Create and publish messages
        left_msg = Int32()
        left_msg.data = absolute_left_ticks
        self.left_encoder_pub.publish(left_msg)

        right_msg = Int32()
        right_msg.data = absolute_right_ticks
        self.right_encoder_pub.publish(right_msg)


    def cleanup(self):
        self.get_logger().info("Executing cleanup...")
        if hasattr(self, 'renderer') and self.renderer:
            self.renderer.close()
        if hasattr(self, 'viewer') and self.viewer and self.viewer.is_running():
            self.viewer.close()


def main(args: Optional[list] = None):
    rclpy.init(args=args)
    bridge_node = MujocoRosBridge()
    try:
        if rclpy.ok():
            rclpy.spin(bridge_node)
    except KeyboardInterrupt:
        pass
    finally:
        bridge_node.cleanup()
        bridge_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()