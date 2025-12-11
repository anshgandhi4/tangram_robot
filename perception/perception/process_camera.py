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

        self.pick_publishers = []
        for i in range(7):
            pub = self.create_publisher(PoseStamped, f'/tangram/pick_{i}_pose', 1)
            self.pick_publishers.append(pub)

        # TODO: ALL PLACE PUBLISHERS ARE TEMPORARY. IDEALLY THE VLM OR DATASET SHOULD OUTPUT THIS
        #       WE NEED A COMMON INTERFACE IN CASE THE VLM DOESNT WORK OR TAKES WAY TOO LONG    
        #       DATASET PARSING NEEDS SOME CHEESE STUFF WHERE WE ASSIGN COLORS TO THE PIECES SO THAT THEY ARE SORTED CORRECTLY
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
        transform.transform.rotation.w = pose.pose.orientation.w
        return transform

    def transform_stamped_to_pose_stamped(self, transform, msg):
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
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

    def image_callback(self, msg):
        tangram = extract_corners_from_image(self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8'), self)
        if tangram is None:
            return

        if len(tangram.pieces) <= 7:
            if not tangram.pieces[0].meters:
                return

            for p in range(len(tangram.pieces)):
                piece = tangram.pieces[p]
                if piece.color != 'green':
                    continue

                piece.pose = np.array([0.0, 0.0, 0.0])

                z_axis_quat = self.z_axis_rot(piece.pose[2] + np.pi)
                final_quat = self.final_quat(z_axis_quat)

                transform = TransformStamped()
                transform.header = msg.header
                transform.header.frame_id = 'ar_marker_0'
                transform.child_frame_id = f'tangram_pick_{p}'
                transform.transform.translation.x = float(piece.pose[0])
                # NOTE: THESE OFFSETS ARE BASED ON THE TRANSLATION FROM EEF TO WRIST 3
                transform.transform.translation.y = float(piece.pose[1]) - 0.04
                transform.transform.translation.z = 0.255
                transform.transform.rotation.x = final_quat[0]
                transform.transform.rotation.y = final_quat[1]
                transform.transform.rotation.z = final_quat[2]
                transform.transform.rotation.w = final_quat[3]
                self.tf_broadcaster.sendTransform(transform)

                try:
                    transform = self.tf_buffer.lookup_transform('base_link', f'tangram_pick_{p}', rclpy.time.Time()).transform
                except:
                    self.get_logger().info('still waiting for buffer transform')
                    continue

                pick_pose = self.transform_stamped_to_pose_stamped(transform, msg)
                self.pick_publishers[p].publish(pick_pose)

                place_pose = pick_pose
                place_pose.pose.position.x *= -1
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
