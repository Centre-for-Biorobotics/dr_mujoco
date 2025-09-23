#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

class BallMover(Node):
    def __init__(self):
        super().__init__('ball_mover')
        self.publisher = self.create_publisher(Float32, 'ball_position', 10)
        self.amplitude = 1.0
        self.frequency = 0.10
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.count = 0
        self.max_count = 1 / self.frequency / 0.05

    def timer_callback(self):
        target_pos = self.amplitude * np.sin(2 * np.pi * self.frequency * self.count * 0.05)
        msg = Float32()
        msg.data = target_pos
        self.count += 1
        if self.count > self.max_count:
            self.count = 0
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    ball_mover = BallMover()
    rclpy.spin(ball_mover)
    ball_mover.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()