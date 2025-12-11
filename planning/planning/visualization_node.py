# ROS Libraries
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener
import numpy as np
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState

# Viser and URDF Libraries
import threading
import time
from pathlib import Path
import viser
from viser.extras import ViserUrdf
import yourdfpy


def get_package_path(package_name: str) -> Path | None:
    """Get the path to a ROS package using robot_descriptions."""
    if package_name == "ur_description":
        try:
            # Use ur5_description (ur5e_description not available)
            from robot_descriptions import ur5_description
            # The package path is the parent of the URDF file
            urdf_path = Path(ur5_description.URDF_PATH)
            pkg_path = urdf_path.parent.parent  # Go up from urdf/ to package root
            print(f"Found ur5_description at: {pkg_path}")
            # List available mesh directories
            meshes_dir = pkg_path / "meshes"
            if meshes_dir.exists():
                print(f"Available mesh dirs: {list(meshes_dir.iterdir())}")
            return pkg_path
        except ImportError as e:
            import traceback
            print(f"ImportError loading ur5_description: {e}")
            traceback.print_exc()
            return None
        except Exception as e:
            import traceback
            print(f"Error loading ur5_description: {e}")
            traceback.print_exc()
            return None
    return None


def make_filename_handler(urdf_dir: Path):
    """Create a filename handler that resolves package:// URIs."""
    # Cache the package path lookup
    _package_cache: dict[str, Path | None] = {}

    def filename_handler(fname: str) -> str:
        if fname.startswith("package://"):
            # Parse package://package_name/path/to/file
            package_path = fname[len("package://"):]
            parts = package_path.split("/", 1)
            package_name = parts[0]
            relative_path = parts[1] if len(parts) > 1 else ""

            # Try to resolve using robot_descriptions (with caching)
            if package_name not in _package_cache:
                _package_cache[package_name] = get_package_path(package_name)
                if _package_cache[package_name]:
                    print(f"Found package '{package_name}' at: {_package_cache[package_name]}")

            pkg_path = _package_cache[package_name]
            if pkg_path is not None:
                resolved = pkg_path / relative_path
                if resolved.exists():
                    return str(resolved)

                # Try remapping ur5e -> ur5 (e-series meshes not available)
                if "ur5e" in relative_path:
                    remapped_path = relative_path.replace("ur5e", "ur5")
                    resolved_remapped = pkg_path / remapped_path
                    if resolved_remapped.exists():
                        print(f"  Remapped ur5e -> ur5: {resolved_remapped}")
                        return str(resolved_remapped)

            # Fallback: try relative to URDF directory
            fallback = urdf_dir / relative_path
            if fallback.exists():
                return str(fallback)

            print(f"Warning: Could not resolve {fname}")
            return fname
        return fname
    return filename_handler


class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')

        # We will utilize the tf buffer to lookup transforms between frames (aruco1, aruco2, )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Flag to track viser initialization
        self.viser_ready = False

        # Start viser server in background thread
        self.get_logger().info('Starting viser server in background thread...')
        self.viser_thread = threading.Thread(target=self._setup_viser, daemon=True)
        self.viser_thread.start()

        # Wait for viser to initialize
        while not self.viser_ready:
            time.sleep(0.1)

        self.get_logger().info('Viser server ready! Starting ROS2 subscriptions...')

        # Subscribe to joint states
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

    def _setup_viser(self):
        """Setup viser server in background thread."""
        try:
            # Start viser server
            self.server = viser.ViserServer()
            self.get_logger().info('Viser server started on http://localhost:8080')

            # Load URDF from assets folder (go up from planning/ to tangram_robot/)
            urdf_path = Path(__file__).parent.parent.parent / "assets" / "ur7e.urdf"

            if not urdf_path.exists():
                self.get_logger().error(f'URDF not found at {urdf_path}')
                return

            self.get_logger().info(f'Loading URDF from {urdf_path}')

            # Load URDF with custom filename handler to resolve package:// URIs
            urdf = yourdfpy.URDF.load(
                urdf_path,
                filename_handler=make_filename_handler(urdf_path.parent),
                load_meshes=True,
                build_scene_graph=True,
                load_collision_meshes=False,
                build_collision_scene_graph=False,
            )

            # Create ViserUrdf object
            self.viser_urdf = ViserUrdf(
                self.server,
                urdf_or_path=urdf,
                load_meshes=True,
                load_collision_meshes=False,
            )

            # Create grid
            trimesh_scene = self.viser_urdf._urdf.scene
            self.server.scene.add_grid(
                "/grid",
                width=2,
                height=2,
                position=(
                    0.0,
                    0.0,
                    trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0,
                ),
            )

            # Get the expected joint names from ViserUrdf
            self.expected_joint_names = list(self.viser_urdf.get_actuated_joint_limits().keys())
            self.get_logger().info(f'Expected joint order: {self.expected_joint_names}')

            # Set viser as ready
            self.viser_ready = True

            # Keep server alive
            while True:
                time.sleep(0.1)

        except Exception as e:
            self.get_logger().error(f'Error setting up viser: {e}')
            import traceback
            traceback.print_exc()

    def _extract_joint_positions(self, msg: JointState) -> list[float]:
        """Extract joint positions in the order expected by ViserUrdf."""
        # Create a mapping from joint name to position
        joint_dict = dict(zip(msg.name, msg.position))

        # Build ordered list matching ViserUrdf's expected order
        ordered_positions = []
        for joint_name in self.expected_joint_names:
            if joint_name in joint_dict:
                ordered_positions.append(joint_dict[joint_name])
            else:
                self.get_logger().warn(
                    f'Joint {joint_name} not in joint_states message',
                    throttle_duration_sec=5.0  # Only warn every 5 seconds
                )
                ordered_positions.append(0.0)

        return ordered_positions

    def joint_state_callback(self, msg: JointState):
        """Updates the viser server with the current joint state of the UR7e robot"""
        if not self.viser_ready:
            return

        # Extract joint positions in correct order
        joint_positions = self._extract_joint_positions(msg)

        # Update viser visualization
        self.viser_urdf.update_cfg(np.array(joint_positions))

def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()