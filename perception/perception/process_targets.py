import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np
from scipy.spatial.transform import Rotation as R
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from PIL import Image

import sys
sys.path.append('src')
sys.path.append('src/scrape_dataset')

from scrape_dataset.parse_images import extract_corners_from_image

class TargetProcessor(Node):
    def __init__(self):
        super().__init__('target_processor')

        self.place_publishers = []
        for i in range(7):
            pub = self.create_publisher(PoseStamped, f'/tangram/place_{i}_pose', 1)
            self.place_publishers.append(pub)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.piece_transforms = {'blue': None, 'light blue': None, 'green': None, 'yellow': None, 'purple': None, 'hot pink': None, 'red': None}
        self.num_frames = 0
        self.pixel_to_color = {(164, 100, 255): 'blue', (18, 255, 255): 'light blue', (139, 148, 190): 'green', (102, 255, 255): 'yellow', (60, 255, 255): 'purple', (30, 255, 255): 'hot pink', (0, 228, 255): 'red'}

        self.tangram = None
        self.timer0 = self.create_timer(0.1, self.image_callback)
        self.timer = self.create_timer(1, self.publish_place_poses)

        self.get_tangram_from_dataset(np.array(Image.open('/home/cc/ee106a/fa25/class/ee106a-aek/ros2_ws/src/scrape_dataset/tangrams/tangram-bear-solution-50.png')), 'bear')

    def transform_stamped_to_pose_stamped(self, transform):
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = 'base_link'
        pose_stamped.pose.position.x = transform.translation.x
        pose_stamped.pose.position.y = transform.translation.y
        pose_stamped.pose.position.z = transform.translation.z
        pose_stamped.pose.orientation.x = transform.rotation.x
        pose_stamped.pose.orientation.y = transform.rotation.y
        pose_stamped.pose.orientation.z = transform.rotation.z
        pose_stamped.pose.orientation.w = transform.rotation.w
        return pose_stamped

    def z_axis_rot(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                                   [np.sin(theta),  np.cos(theta), 0],
                                   [0,              0,             1]])

    def final_quat(self, rot_mat):
        rot = np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, -1]]) @ rot_mat
        return R.from_matrix(rot).as_quat()

    def publish_place_poses(self):
        for color in self.piece_transforms:
            if self.piece_transforms[color] is not None:
                self.place_publishers[list(self.piece_transforms.keys()).index(color)].publish(self.piece_transforms[color])

    def get_tangram_from_dataset(self, img, prompt):
        tangram = extract_corners_from_image(img, REAL=False, ROS_PUB=False, prompt=prompt)
        if tangram is None:
            return

        if len(tangram.pieces) == 0:
            return

        self.num_frames += 1
        ARUCO_RATIO = None
        RIGHT_BUFFER = 0.55
        UP_BUFFER = 0.08
        for piece in tangram.pieces:
            if piece.shape == 'square':
                piece.coords = piece.coords.astype(np.float32)
                # TODO: ADDING EXTRA TOLERANCE HERE
                ARUCO_RATIO = 1.05 * 0.1016 / np.linalg.norm(piece.coords[0] - piece.coords[1])

        if ARUCO_RATIO is None:
            return

        for p in range(len(tangram.pieces)):
            piece = tangram.pieces[p]
            piece.color = self.pixel_to_color[piece.color]

            piece.pose[0] = float(piece.pose[0] - img.shape[1] // 2) * ARUCO_RATIO + RIGHT_BUFFER
            piece.pose[1] = float(img.shape[0] // 2 - piece.pose[1]) * ARUCO_RATIO + UP_BUFFER
            piece.pose[2] = piece.pose[2] + np.pi

        tangram.sort_by_color()
        self.tangram = tangram

    def image_callback(self):
        if self.tangram is None:
            return

        tangram = self.tangram

        for p in range(len(tangram.pieces)):
            piece = tangram.pieces[p]
            p_col = piece.color

            z_axis_quat = self.z_axis_rot(piece.pose[2] + np.pi)
            final_quat = self.final_quat(z_axis_quat)

            transform_to_marker = TransformStamped()
            transform_to_marker.header.stamp = self.get_clock().now().to_msg()
            transform_to_marker.header.frame_id = 'ar_marker_0'
            transform_to_marker.child_frame_id = f'tangram_place_{p_col}'
            # NOTE: THESE OFFSETS ARE BASED ON THE TRANSLATION FROM EEF TO WRIST 3
            transform_to_marker.transform.translation.x = float(piece.pose[0]) - 0.066 * np.sin(piece.pose[2])
            transform_to_marker.transform.translation.y = float(piece.pose[1]) - 0.066 * np.cos(piece.pose[2])
            transform_to_marker.transform.translation.z = 0.255
            transform_to_marker.transform.rotation.x = final_quat[0]
            transform_to_marker.transform.rotation.y = final_quat[1]
            transform_to_marker.transform.rotation.z = final_quat[2]
            transform_to_marker.transform.rotation.w = final_quat[3]

            self.tf_broadcaster.sendTransform(transform_to_marker)

            try:
                transform = self.tf_buffer.lookup_transform('base_link', f'tangram_place_{p_col}', rclpy.time.Time()).transform
            except:
                self.get_logger().info('target waiting for buffer transform')
                continue

            if self.piece_transforms[p_col] is not None and self.num_frames > 200:
                place_pose = self.piece_transforms[p_col]
            else:
                place_pose = self.transform_stamped_to_pose_stamped(transform)
                self.piece_transforms[p_col] = place_pose

def main(args=None):
    rclpy.init(args=args)
    node = TargetProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
