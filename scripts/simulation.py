#!/usr/bin/env python3
import numpy as np
import os
import xml.etree.ElementTree as ET
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist, Pose, Point, Quaternion, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import mujoco
import mujoco.viewer

class MujocoRosBridge(Node):
    def __init__(self):
        super().__init__('mujoco_ros_bridge')
        script_dir = os.path.dirname(os.path.abspath(__file__))
        def_world_path = os.path.join(script_dir, "world/default.xml")
        def_robot_path = os.path.join(script_dir, "models/dummy.xml")

        self.get_logger().info(f"World Path: {def_world_path}")
        self.get_logger().info(f"Robot Path: {def_robot_path}")

        self.declare_parameter('world_path', def_world_path)
        self.declare_parameter('robot_path', def_robot_path)

        world_path = self.get_parameter('world_path').get_parameter_value().string_value
        robot_path = self.get_parameter('robot_path').get_parameter_value().string_value

        if not world_path or not robot_path:
            self.get_logger().fatal("Parameters 'robot_path' must be set.")
            self.get_logger().fatal(f"[FOR STUDENT] Check your launch file and model xml")
            rclpy.shutdown()
            return

        try:
            combined_xml_string = self._merge_models(world_path, robot_path)
            self.model = mujoco.MjModel.from_xml_string(combined_xml_string)
            self.data = mujoco.MjData(self.model)
            self.get_logger().info(f"World = '{os.path.basename(world_path)}'  Robot = '{os.path.basename(robot_path)}'")
        except Exception as e:
            self.get_logger().fatal(f"Failed to load models: {e}")
            self.get_logger().fatal(f"[FOR STUDENT] Check your launch file and model xml")
            rclpy.shutdown()
            return

        try:
            # Got geometry data from xml file
            left_wheel_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'left_wheel_geom')
            self.wheel_radius = self.model.geom_size[left_wheel_geom_id][0]
            left_wheel_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'left_wheel')
            right_wheel_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'right_wheel')
            left_pos = self.model.body_pos[left_wheel_body_id]
            right_pos = self.model.body_pos[right_wheel_body_id]
            self.wheel_base = np.linalg.norm(left_pos - right_pos)

            self.get_logger().info(f"Loaded parameters from XML: Wheel Radius={self.wheel_radius:.4f}, Wheel Base={self.wheel_base:.4f}")
        except KeyError as e:
            self.get_logger().error(f"Could not find model component: {e}. Check your XML file.")
            rclpy.shutdown()
            return
        
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # Create camera simulating randerer
        self.renderer = mujoco.Renderer(self.model, 480, 640)

        # ROS 2 Setup
        self.odom_pub = self.create_publisher(Odometry, '/diff_cont/odom', 10)
        self.image_pub = self.create_publisher(Image, '/camera1/image_raw', 10)
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/diff_cont/cmd_vel', self.cmd_vel_callback, 10)
        self.ball_pos_sub = self.create_subscription(
            Float32, '/ball_position', self.move_ball, 10)

        self.bridge = CvBridge()
        self.timer = self.create_timer(0.005, self.timer_callback)

        self.get_logger().info("MuJoCo ROS Bridge has been started.")

    def _merge_models(self, world_path, robot_path):
        world_tree = ET.parse(world_path)
        robot_tree = ET.parse(robot_path)
        
        world_root = world_tree.getroot()
        robot_root = robot_tree.getroot()

        world_body = world_root.find('worldbody')
        robot_body = robot_root.find('worldbody')
        if world_body is not None and robot_body is not None:
            for body in list(robot_body):
                world_body.append(body)

        sections_to_merge = ['asset', 'visual', 'actuator', 
                             'sensor', 'tendon', 'keyframe']
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

    def cmd_vel_callback(self, msg):
        linear_x = msg.linear.x
        angular_z = msg.angular.z
        
        v_right = linear_x + (self.wheel_base / 2.0) * angular_z
        v_left = linear_x - (self.wheel_base / 2.0) * angular_z

        omega_right = v_right / self.wheel_radius
        omega_left = v_left / self.wheel_radius

        gain = 0.05

        self.data.ctrl[0] = np.clip(omega_left * gain, -0.1, 0.1)  # left_motor
        self.data.ctrl[1] = np.clip(omega_right * gain, -0.1, 0.1) # right_motor

    def move_ball(self, position_msg):
        actuator_id = mujoco.mj_name2id(self.model, 
                                        mujoco.mjtObj.mjOBJ_ACTUATOR, 
                                        'ball_actuator')
        self.data.ctrl[actuator_id] = position_msg.data

    def timer_callback(self):
        mujoco.mj_step(self.model, self.data)

        if self.viewer.is_running():
            self.viewer.sync()
        else:
            self.get_logger().info("Viewer closed, shutting down node.")
            rclpy.shutdown()
            return

        now = self.get_clock().now().to_msg()

        odom_msg = Odometry()
        odom_msg.header.stamp = now
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        odom_msg.pose.pose.position = Point(
            x=self.data.qpos[0], y=self.data.qpos[1], z=self.data.qpos[2])
        odom_msg.pose.pose.orientation = Quaternion(
            x=self.data.qpos[4], y=self.data.qpos[5], z=self.data.qpos[6], w=self.data.qpos[3])
        odom_msg.twist.twist.linear = Vector3(
            x=self.data.qvel[0], y=self.data.qvel[1], z=self.data.qvel[2])
        odom_msg.twist.twist.angular = Vector3(
            x=self.data.qvel[3], y=self.data.qvel[4], z=self.data.qvel[5])
        self.odom_pub.publish(odom_msg)
    
        try:
            self.renderer.update_scene(self.data, camera="front_camera")
            pixels = self.renderer.render()
            bgr_image = pixels[..., ::-1]
            image_msg = self.bridge.cv2_to_imgmsg(bgr_image, "bgr8")
            image_msg.header.stamp = now
            image_msg.header.frame_id = "front_camera_frame"
            self.image_pub.publish(image_msg)
        except Exception as e:
            pass


def main(args=None):
    rclpy.init(args=args)
    bridge_node = MujocoRosBridge()
    
    if rclpy.ok():
        rclpy.spin(bridge_node)
    if bridge_node.renderer:
        bridge_node.renderer.close()
    if bridge_node.viewer and bridge_node.viewer.is_running():
        bridge_node.viewer.close()
        
    bridge_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
