import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, TransformStamped
import numpy as np
from std_msgs.msg import Header
from cv_bridge import CvBridge

import sys
sys.path.append('src')
sys.path.append('src/scrape_dataset')

from scrape_dataset.parse_images import extract_corners_from_image
from scrape_dataset.tangram import Tangram, Piece

class RealSenseSubscriber(Node):
    def __init__(self):
        super().__init__('realsense_subscriber')

        # Publishers
        self.tangram_publishers = []
        for i in range(7):
            pub = self.create_publisher(PointStamped, f'/tangram/piece_{i}_pose', 1)
            self.tangram_publishers.append(pub)

        self.rect_pub = self.create_publisher(Image, '/tangram/rectified_image', 1)
        self.output_pub = self.create_publisher(Image, '/tangram/output_image', 1)

        self.cam_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 1)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        # self.get_logger().info('processing image')
        tangram = extract_corners_from_image(self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8'), self.get_logger())
        # self.get_logger().info(f'{tangram}')
        if len(tangram.pieces) <= 7:
            for p in range(len(tangram.pieces)):
                piece = tangram.pieces[p]
                # self.get_logger().info(f"Detected piece of color {piece.color} with corners: {piece.coords}")

                piece_pose = PointStamped()
                piece_pose.header = msg.header
                piece_pose.point.x = float(piece.pose[0] / 100)
                piece_pose.point.y = float(piece.pose[1] / 100)
                piece_pose.point.z = float(piece.pose[2])
                self.tangram_publishers[p].publish(piece_pose)

def main(args=None):
    rclpy.init(args=args)
    node = RealSenseSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
