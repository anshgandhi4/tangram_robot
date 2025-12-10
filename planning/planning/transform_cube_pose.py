import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener
import numpy as np

from scipy.spatial.transform import Rotation as R

class TransformCubePose(Node):
    def __init__(self):
        super().__init__('transform_cube_pose')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cube_pose_sub = self.create_subscription(PointStamped, '/tangram/piece_0_pose', self.cube_pose_callback, 10)
        self.cube_pose_pub = self.create_publisher(PointStamped, '/transformed_cube_pose', 10)

        self.create_timer(0.01, self.tf_test)

        rclpy.spin_once(self, timeout_sec=2)
        self.cube_pose = None

    def tf_test(self):
        transformed = PointStamped()
        transformed.header.frame_id = 'base_link'
        transformed.point.x = 0.12
        transformed.point.y = 0.61
        transformed.point.z = -0.12
        self.cube_pose_pub.publish(transformed)

    def tangram_tf_publish(self):
        transformed = PointStamped()
        transformed.header.frame_id = 'base_link'
        transformed.point.x = 0.12
        transformed.point.y = 0.61
        transformed.point.z = -0.115
        self.cube_pose_pub.publish(transformed)

    def cube_pose_callback(self, msg: PointStamped):
        if self.cube_pose is None:
            self.cube_pose = self.transform_cube_pose(msg)

        self.cube_pose_pub.publish(self.cube_pose)

    def transform_cube_pose(self, msg: PointStamped):
        """ 
        Transform point into base_link frame
        Args: 
            - msg: PointStamped - The message from /cube_pose, of the position of the cube in camera_depth_optical_frame
        Returns:
            PointStamped: point in base_link_frame in form [x, y, z]
        """

        pose = self.tf_buffer.lookup_transform('base_link', 'wrist_3_link', rclpy.time.Time()).transform
        self.get_logger().info(f"Transform: {pose}")

        G = np.eye(4)
        G[:3, :3] = R.from_quat([pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w]).as_matrix()
        G[0, 3] = pose.translation.x
        G[1, 3] = pose.translation.y
        G[2, 3] = pose.translation.z

        point = G @ np.array([msg.point.x, msg.point.y, msg.point.z, 1])

        transformed = PointStamped()
        transformed.header = msg.header
        transformed.header.frame_id = 'base_link'
        transformed.point.x =  point[0]
        transformed.point.y =  point[1]
        transformed.point.z =  point[2]

        return transformed

def main(args=None):
    rclpy.init(args=args)
    node = TransformCubePose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
