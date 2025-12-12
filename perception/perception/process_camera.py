import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
from tf2_ros import Buffer, TransformListener, TransformBroadcaster, StaticTransformBroadcaster

import sys
sys.path.append('src')
sys.path.append('src/scrape_dataset')

from scrape_dataset.parse_images import extract_corners_from_image

class RealSenseSubscriber(Node):
    def __init__(self):
        super().__init__('realsense_subscriber')

        self.pick_publishers = []
        for i in range(7):
            pub = self.create_publisher(PoseStamped, f'/tangram/pick_{i}_pose', 1)
            self.pick_publishers.append(pub)

        self.rect_pub = self.create_publisher(Image, '/tangram/rectified_image', 1)
        self.masked_pub = self.create_publisher(Image, '/tangram/masked_image', 1)
        self.output_pub = self.create_publisher(Image, '/tangram/output_image', 1)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_broadcaster = StaticTransformBroadcaster(self)

        self.cam_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 1)
        self.bridge = CvBridge()

        self.piece_transforms = {'blue': None, 'light blue': None, 'green': None, 'yellow': None, 'purple': None, 'hot pink': None, 'red': None}
        self.num_frames = 0
        self.base_to_cam = None

        self.static_base_to_cam = None
        self.static_cam_to_ar0 = None

        self.timer = self.create_timer(1, self.publish_pick_poses)
        self.transform_timer = self.create_timer(0.05, self.broadcast_static_transforms)
        self.process_image_lock = False

    def broadcast_static_transforms(self):
        # this function broadcasts 2 transforms:
        #   1. camera_link -> aruco marker_{marker_id}
        #   2. camera_link -> aruco marker_0 (table aruco)
        if self.static_base_to_cam is None:
            self.get_logger().info('Our static transforms are NONE!!!')
        if self.static_base_to_cam is not None:
            self.static_base_to_cam.header.stamp = self.get_clock().now().to_msg()
            self.static_broadcaster.sendTransform(self.static_base_to_cam)
            self.get_logger().info('Broadcasting static transforms...')
        if self.static_cam_to_ar0 is not None:
            self.static_cam_to_ar0.header.stamp = self.get_clock().now().to_msg()
            self.static_broadcaster.sendTransform(self.static_cam_to_ar0)
            self.get_logger().info('Broadcasting static transforms...')

        
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

    def publish_pick_poses(self):
        for color in self.piece_transforms:
            if self.piece_transforms[color] is not None:
                self.get_logger().info(f'Publishing pick pose for {color}')
                self.pick_publishers[list(self.piece_transforms.keys()).index(color)].publish(self.piece_transforms[color])

    def image_callback(self, msg):
        tangram = extract_corners_from_image(self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8'), node=self, ROS_PUB=True)
        if tangram is None:
            return

        if len(tangram.pieces) == 0 or not tangram.pieces[0].meters:
            return

        self.num_frames += 1
        for p in range(len(tangram.pieces)):
            piece = tangram.pieces[p]
            p_col = piece.color

            # TODO: UNDO THIS
            piece.pose[2] = np.pi
            z_axis_quat = self.z_axis_rot(piece.pose[2] + np.pi)
            final_quat = self.final_quat(z_axis_quat)

            transform_to_marker = TransformStamped()
            transform_to_marker.header = msg.header
            transform_to_marker.header.frame_id = 'ar_marker_0'
            transform_to_marker.child_frame_id = f'tangram_pick_{p_col}'
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
                transform = self.tf_buffer.lookup_transform('base_link', f'tangram_pick_{p_col}', rclpy.time.Time()).transform
                if self.num_frames <= 1000:
                    self.get_logger().info(f'within first {self.num_frames} frames, setting static transforms')
                    base_to_cam_transform = self.tf_buffer.lookup_transform('ar_marker_8', 'camera_link', rclpy.time.Time())
                    base_to_ar0 = self.tf_buffer.lookup_transform('camera_link', 'ar_marker_0', rclpy.time.Time())

                    if base_to_cam_transform is not None:
                        self.get_logger().info('NOT NOOOOONONONONEONOENOEN')
                        self.static_base_to_cam = base_to_cam_transform

                    if base_to_ar0 is not None:
                        self.get_logger().info('NOT NOOOOONONONONEONOENOEN')
                        self.static_cam_to_ar0 = base_to_ar0
                    # self.static_broadcaster.sendTransform(base_to_cam_transform)
                    # self.static_broadcaster.sendTransform(base_to_ar0)
                    # self.get_logger().info('Static transforms set and broadcasted.')
                    
            except:
                self.get_logger().info('still waiting for buffer transform')
                continue

            if self.piece_transforms[p_col] is not None and self.num_frames > 100000:
                pick_pose = self.piece_transforms[p_col]
            else:
                # self.base_to_cam = self.tf_buffer.lookup_transform('base_link', 'camera_link', rclpy.time.Time()).transform
                pick_pose = self.transform_stamped_to_pose_stamped(transform, msg)
                self.piece_transforms[p_col] = pick_pose

def main(args=None):
    rclpy.init(args=args)
    node = RealSenseSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
