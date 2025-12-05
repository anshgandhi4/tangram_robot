#!/usr/bin/env python3
"""
Basic viser server for visualizing the UR7e robot with pedestal.
Uses yourdfpy to load and display the URDF model.
"""
import os
import time
import numpy as np
import viser
import yourdfpy

def main():
    # Path to the URDF file
    urdf_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "assets",
        "ur7e.urdf"
    )

    # Check if URDF exists
    if not os.path.exists(urdf_path):
        print(f"Error: URDF file not found at {urdf_path}")
        print("Please run the xacro conversion first:")
        print("  cd assets")
        print("  conda activate text2tangram")
        print("  xacro ur7e_resolved.urdf.xacro > ur7e.urdf")
        return

    print(f"Loading URDF from: {urdf_path}")

    # Load the URDF
    try:
        robot = yourdfpy.URDF.load(urdf_path)
        print(f"Successfully loaded robot: {robot.name}")
        print(f"Number of links: {len(robot.link_map)}")
        print(f"Number of joints: {len(robot.joint_map)}")
    except Exception as e:
        print(f"Error loading URDF: {e}")
        return

    # Create viser server
    server = viser.ViserServer()
    print(f"Viser server started at http://localhost:8080")

    # Add the robot to the scene
    # Get the robot configuration (all joints at zero)
    cfg = np.zeros(len(robot.actuated_joint_names))

    # Get the robot's scene
    robot_scene = robot.scene

    # Add each mesh from the robot to viser
    for geom in robot_scene.geometry.values():
        # Get mesh vertices and faces
        vertices = geom.vertices
        faces = geom.faces

        # Add mesh to viser
        server.add_mesh_simple(
            name=f"robot_mesh_{id(geom)}",
            vertices=vertices,
            faces=faces,
            color=(0.7, 0.7, 0.7),
        )

    # Add coordinate frame at origin
    server.add_frame(
        name="world",
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
        axes_length=0.3,
        axes_radius=0.01,
    )

    # Add GUI controls for joint angles
    joint_sliders = {}
    for i, joint_name in enumerate(robot.actuated_joint_names):
        joint = robot.joint_map[joint_name]

        # Get joint limits if available
        if hasattr(joint, 'limit') and joint.limit is not None:
            lower = joint.limit.lower if joint.limit.lower is not None else -np.pi
            upper = joint.limit.upper if joint.limit.upper is not None else np.pi
        else:
            lower = -np.pi
            upper = np.pi

        # Create slider for this joint
        slider = server.add_gui_slider(
            name=joint_name,
            min=lower,
            max=upper,
            step=0.01,
            initial_value=0.0,
        )
        joint_sliders[joint_name] = slider

    # Add reset button
    reset_button = server.add_gui_button("Reset Pose")

    print("\nVisualization ready!")
    print("- Open http://localhost:8080 in your browser")
    print("- Use the sliders to control joint angles")
    print("- Use mouse to rotate/pan/zoom the view")

    # Main loop to update robot configuration
    try:
        while True:
            # Check if reset button was clicked
            if reset_button.value:
                for slider in joint_sliders.values():
                    slider.value = 0.0
                reset_button.value = False

            # Get current joint values from sliders
            cfg = np.array([joint_sliders[name].value for name in robot.actuated_joint_names])

            # Update robot configuration
            # Note: For a full interactive visualization, you would update
            # the mesh transforms based on forward kinematics here

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    main()
