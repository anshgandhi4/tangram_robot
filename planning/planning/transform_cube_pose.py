import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped
from tf2_ros import Buffer, TransformListener
import numpy as np

from scipy.spatial.transform import Rotation as R

class TransformCubePose(Node):
    def __init__(self):
        super().__init__('transform_cube_pose')

        self.target_poses = [(0.12, 0.61, 0), (0.12, 0.61, 0), (0.12, 0.61, 0), (0.12, 0.61, 0), (0.12, 0.61, 0), (0.12, 0.61, 0)]

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cube_pose_sub = self.create_subscription(PointStamped, '/tangram/pick_0_pose', self.tangram_tf_publish, 10)
        # self.cube_pose_sub = self.create_subscription(PointStamped, '/tangram/place_0_pose', self.tangram_tf_publish, 10)
        self.cube_pose_pub = self.create_publisher(PoseStamped, '/transformed_cube_pose', 10)

        self.create_timer(0.01, self.tf_test)

        rclpy.spin_once(self, timeout_sec=2)
        self.targ_pose = None

    def tf_test(self):
        target_pose = self.target_poses[0]

        transformed = PoseStamped()
        transformed.header.frame_id = 'base_link'
        transformed.pose.position.x = target_pose[0]
        transformed.pose.position.y = target_pose[1]
        transformed.pose.position.z = -0.12


        theta = target_pose[2]
        transformed.pose.orientation.x = (1/np.sqrt(2)) * np.cos(theta/2)
        transformed.pose.orientation.y = -(1/np.sqrt(2))
        transformed.pose.orientation.z = 0.0
        transformed.pose.orientation.w = (1/np.sqrt(2)) *np.sin(theta/2)

        self.cube_pose_pub.publish(transformed)

    def tangram_tf_publish(self, msg):
        target_pose = self.target_poses[0]

        transformed = PoseStamped()
        transformed.header.frame_id = 'base_link'
        transformed.pose.position.x = target_pose[0]
        transformed.pose.position.y = target_pose[1]
        transformed.pose.position.z = -0.12


        theta = target_pose[2]
        transformed.pose.orientation.x = (1/np.sqrt(2)) * np.cos(theta/2)
        transformed.pose.orientation.y = -(1/np.sqrt(2))
        transformed.pose.orientation.z = 0.0
        transformed.pose.orientation.w = (1/np.sqrt(2)) *np.sin(theta/2)

        self.cube_pose_pub.publish(transformed)

    def targ_pose_callback(self, msg):
        if self.targ_pose is None:
            self.targ_pose = self.transform_cube_pose(msg)

        self.cube_pose_pub.publish(self.cube_pose)

    def transform_cube_pose(self, msg):
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
