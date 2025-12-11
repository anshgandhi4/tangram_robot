import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import numpy as np
from std_msgs.msg import Header
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

import sys
sys.path.append('src')
sys.path.append('src/scrape_dataset')

from scrape_dataset.parse_images import extract_corners_from_image
from scrape_dataset.tangram import Tangram, Piece

class RealSenseSubscriber(Node):
    def __init__(self):
        super().__init__('realsense_subscriber')

        # Publishers
        self.pick_publishers = []
        for i in range(7):
            pub = self.create_publisher(PoseStamped, f'/tangram/pick_{i}_pose', 1)
            self.pick_publishers.append(pub)

        self.place_publishers = []
        for i in range(7):
            pub = self.create_publisher(PoseStamped, f'/tangram/place_{i}_pose', 1)
            self.place_publishers.append(pub)

        self.rect_pub = self.create_publisher(Image, '/tangram/rectified_image', 1)
        self.masked_pub = self.create_publisher(Image, '/tangram/masked_image', 1)
        self.output_pub = self.create_publisher(Image, '/tangram/output_image', 1)

        self.rand = [0.0] * 7#[0.0, np.pi/2, np.pi/4, -np.pi/4, np.pi/6, -np.pi/6, np.pi/3]

        self.base_rot = np.array([[-1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, -1]])

        self.cam_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 1)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        # self.get_logger().info('processing image')
        tangram = extract_corners_from_image(self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8'), self)
        # self.get_logger().info(f'{tangram}')
        if len(tangram.pieces) <= 7:
            for p in range(len(tangram.pieces)):
                piece = tangram.pieces[p]
                # theta = piece.pose[2]
                # self.get_logger().info(f"Detected piece of color {piece.color} with corners: {piece.coords}")

                theta = self.rand[p]
                pick_pose = PoseStamped()
                pick_pose.header = msg.header
                pick_pose.pose.position.x = 0.12
                pick_pose.pose.position.y = 0.61
                pick_pose.pose.position.z = 0.12

                rot_mat_z_axis = np.array([[np.cos(theta), -np.sin(theta), 0],
                                           [np.sin(theta),  np.cos(theta), 0],
                                           [0,              0,             1]])

                quaternion = R.from_matrix(self.base_rot @ rot_mat_z_axis).as_quat()

                pick_pose.pose.orientation.x = quaternion[0]
                pick_pose.pose.orientation.y = quaternion[1]
                pick_pose.pose.orientation.z = quaternion[2]
                pick_pose.pose.orientation.w = quaternion[3]
                # pick_pose.pose.orientation.y = 1.0
                # pick_pose.pose.orientation.w = 0.0
                self.pick_publishers[p].publish(pick_pose)

                place_pose = pick_pose
                place_pose.pose.position.x *= -1
                theta = self.rand[(p + 1) % 7]

                rot_mat_z_axis = np.array([[np.cos(theta), -np.sin(theta), 0],
                                           [np.sin(theta),  np.cos(theta), 0],
                                           [0,              0,             1]])

                quaternion = R.from_matrix(self.base_rot @ rot_mat_z_axis).as_quat()
                place_pose.pose.orientation.x = quaternion[0]
                place_pose.pose.orientation.y = quaternion[1]
                place_pose.pose.orientation.z = quaternion[2]
                place_pose.pose.orientation.w = quaternion[3]
                # place_pose.pose.orientation.y = 1.0
                # place_pose.pose.orientation.w = 0.0
                self.place_publishers[p].publish(place_pose)

def main(args=None):
    rclpy.init(args=args)
    node = RealSenseSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
