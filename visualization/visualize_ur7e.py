#!/usr/bin/env python3
"""
Basic UR7e visualization using viser and yourdfpy.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import viser
import yourdfpy


def process_xacro_to_urdf(xacro_path: str, ur_type: str = "ur7e") -> str:
    """
    Process xacro file to URDF.

    Args:
        xacro_path: Path to the xacro file
        ur_type: UR robot type (e.g., "ur7e")

    Returns:
        Path to the generated URDF file
    """
    # Get the base directory for ur_description
    ur_description_path = Path(xacro_path).parent.parent

    # Create temporary file for URDF output
    temp_urdf = tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False)
    temp_urdf_path = temp_urdf.name
    temp_urdf.close()

    # Process xacro with required arguments
    cmd = [
        'xacro',
        xacro_path,
        f'ur_type:={ur_type}',
        f'name:=ur',
        f'joint_limit_params:={ur_description_path}/config/{ur_type}/joint_limits.yaml',
        f'kinematics_params:={ur_description_path}/config/{ur_type}/default_kinematics.yaml',
        f'physical_params:={ur_description_path}/config/{ur_type}/physical_parameters.yaml',
        f'visual_params:={ur_description_path}/config/{ur_type}/visual_parameters.yaml',
    ]

    print(f"Processing xacro: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Write URDF to temp file
        with open(temp_urdf_path, 'w') as f:
            f.write(result.stdout)

        print(f"URDF generated at: {temp_urdf_path}")
        return temp_urdf_path

    except subprocess.CalledProcessError as e:
        print(f"Error processing xacro: {e}")
        print(f"stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        print("ERROR: xacro command not found. Please install it:")
        print("  pip install xacro")
        raise


def main():
    """Main visualization function."""
    # Get paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    xacro_path = project_root / "assets" / "ur_description" / "urdf" / "ur.urdf.xacro"

    if not xacro_path.exists():
        raise FileNotFoundError(f"Xacro file not found at: {xacro_path}")

    print("=" * 60)
    print("UR7e Viser Visualization")
    print("=" * 60)

    # Process xacro to URDF
    print("\n1. Processing xacro to URDF...")
    urdf_path = process_xacro_to_urdf(str(xacro_path), ur_type="ur7e")

    # Load URDF with yourdfpy
    print("\n2. Loading URDF with yourdfpy...")
    robot = yourdfpy.URDF.load(urdf_path)
    print(f"   Loaded robot with {len(robot.link_map)} links and {len(robot.joint_map)} joints")

    # Create viser server
    print("\n3. Starting viser server...")
    server = viser.ViserServer(port=8080)
    print("   Viser server started at http://localhost:8080")

    # Add robot to scene
    print("\n4. Adding robot to scene...")

    # Get the robot configuration (all joints at zero)
    cfg = np.zeros(len(robot.actuated_joint_names))

    # Add robot meshes to viser
    # Note: yourdfpy provides mesh data, we need to add it to viser
    scene = robot.scene

    # Add each mesh from the scene
    for node in scene.graph.nodes_geometry:
        for mesh in scene.graph.geometry[node]:
            # Get mesh transform
            transform = scene.graph.get_transform(node)[0]

            # Add mesh to viser
            vertices = mesh.vertices
            faces = mesh.faces

            # Create a unique name for this mesh
            mesh_name = f"{node}_{hash(mesh.vertices.tobytes())}"

            server.add_mesh_simple(
                name=mesh_name,
                vertices=vertices,
                faces=faces,
                position=transform[:3, 3],
                wxyz_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),  # We'll handle rotation separately
                color=(200, 200, 200),
            )

    print("\n5. Visualization ready!")
    print("   - Open http://localhost:8080 in your browser")
    print("   - Press Ctrl+C to exit")
    print("=" * 60)

    # Keep server running
    try:
        while True:
            server.flush()  # Process events
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Clean up temp file
        if os.path.exists(urdf_path):
            os.unlink(urdf_path)


if __name__ == "__main__":
    main()
