import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np
from std_msgs.msg import Header
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
from tf2_ros import Buffer, TransformListener, TransformBroadcaster

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

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.rand = [0.0] * 7#[0.0, np.pi/2, np.pi/4, -np.pi/4, np.pi/6, -np.pi/6, np.pi/3]

        self.base_rot = np.array([[-1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, -1]])

        self.cam_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 1)
        self.bridge = CvBridge()

    def construct_g_matrix(self, pose):
        G = np.eye(4)
        G[:3, :3] = R.from_quat([pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w]).as_matrix()
        G[0, 3] = pose.translation.x
        G[1, 3] = pose.translation.y
        G[2, 3] = pose.translation.z
        return G

    def translation_quat_to_pose_stamped(self, translation, quat, msg):
        pose = PoseStamped()
        pose.header = msg.header
        pose.header.frame_id = 'camera_link'
        pose.pose.position.x = translation[0]
        pose.pose.position.y = translation[1]
        pose.pose.position.z = translation[2]
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]
        return pose

    def pose_stamped_to_transform_stamped(self, pose, child_frame_id):
        transform = TransformStamped()
        transform.header = pose.header
        transform.child_frame_id = child_frame_id
        transform.transform.translation.x = pose.pose.position.x
        transform.transform.translation.y = pose.pose.position.y
        transform.transform.translation.z = pose.pose.position.z
        transform.transform.rotation.x = pose.pose.orientation.x
        transform.transform.rotation.y = pose.pose.orientation.y
        transform.transform.rotation.z = pose.pose.orientation.z
        transform.transform.rotation.x = pose.pose.orientation.w
        return transform

    def image_callback(self, msg):
        # self.get_logger().info('processing image')
        tangram = extract_corners_from_image(self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8'), self)
        if tangram is None:
            return

        # self.get_logger().info(f'{tangram}')
        if len(tangram.pieces) <= 7:
            if not tangram.pieces[0].meters:
                return

            try:
                pose = self.tf_buffer.lookup_transform('camera_link', 'ar_marker_0', rclpy.time.Time()).transform
            except:
                self.get_logger().info('still waiting for buffer transform')
                return

            G = self.construct_g_matrix(pose)

            for p in range(len(tangram.pieces)):
                piece = tangram.pieces[p]
                if piece.color != 'green':
                    continue

                theta = piece.pose[2]
                theta = 0.0
                piece_translation = G @ np.array([piece.pose[0], piece.pose[1], 0, 1])
                piece_rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                                           [np.sin(theta),  np.cos(theta), 0],
                                           [0,              0,             1]]) @ G[:3, :3]
                piece_rotation = np.eye(3)
                piece_rotation_quat = R.from_matrix(piece_rotation).as_quat()

                pick_pose = self.translation_quat_to_pose_stamped(piece_translation, piece_rotation_quat, msg)
                self.pick_publishers[p].publish(pick_pose)
                self.tf_broadcaster.sendTransform(self.pose_stamped_to_transform_stamped(pick_pose, f'tangram_pick_{p}'))

                place_pose = pick_pose
                place_pose.pose.position.x *= -1
                # theta = self.rand[(p + 1) % 7]

                # rot_mat_z_axis = np.array([[np.cos(theta), -np.sin(theta), 0],
                #                            [np.sin(theta),  np.cos(theta), 0],
                #                            [0,              0,             1]])

                # quaternion = R.from_matrix(self.base_rot @ rot_mat_z_axis).as_quat()
                # place_pose.pose.orientation.x = quaternion[0]
                # place_pose.pose.orientation.y = quaternion[1]
                # place_pose.pose.orientation.z = quaternion[2]
                # place_pose.pose.orientation.w = quaternion[3]
                # place_pose.pose.orientation.y = 1.0
                # place_pose.pose.orientation.w = 0.0
                self.place_publishers[p].publish(place_pose)
                self.tf_broadcaster.sendTransform(self.pose_stamped_to_transform_stamped(place_pose, f'tangram_place_{p}'))

def main(args=None):
    rclpy.init(args=args)
    node = RealSenseSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
