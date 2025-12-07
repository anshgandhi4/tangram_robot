from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import tyro
import yourdfpy

import viser
from viser.extras import ViserUrdf


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


def create_robot_control_sliders(
    server: viser.ViserServer, viser_urdf: ViserUrdf
) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
    """Create sliders in GUI that help us move the robot joints.
    We look through the ViserUrdf object to get the actuated joint limits.
    """
    slider_handles: list[viser.GuiInputHandle[float]] = []
    initial_config: list[float] = []
    for joint_name, (lower,upper) in viser_urdf.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi
        initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_pos,
        )
        slider.on_update(  # When sliders move, we update the URDF configuration.
            lambda _: viser_urdf.update_cfg(
                np.array([slider.value for slider in slider_handles])
            )
        )
        slider_handles.append(slider)
        initial_config.append(initial_pos)
    return slider_handles, initial_config


def add_link_frames(
    server: viser.ViserServer, viser_urdf: ViserUrdf, axes_length: float = 0.1
) -> dict[str, viser.FrameHandle]:
    """Add coordinate frame visualization for each link in the URDF."""
    frame_handles = {}
    for link_name in viser_urdf._urdf.link_map.keys(): # dict_keys(['world', 'pedestal', 'controller_box', 'pedestal_feet', 'mounting_plate', 'base_link', 'base_link_inertia', 'shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'ft_frame', 'base', 'flange', 'tool0'])
        frame = server.scene.add_frame(
            f"/urdf/{link_name}/frame",
            axes_length=axes_length,
            axes_radius=axes_length * 0.05,
        )
        frame_handles[link_name] = frame
    return frame_handles # length 17


def main(
    load_meshes: bool = True,
    load_collision_meshes: bool = False,
    show_frames: bool = True,
) -> None:
    # Start viser server.
    server = viser.ViserServer()

    # Load URDF from local assets folder.
    urdf_path = Path(__file__).parent.parent / "assets" / "ur7e.urdf"

    # Load URDF with custom filename handler to resolve package:// URIs
    urdf = yourdfpy.URDF.load(
        urdf_path,
        filename_handler=make_filename_handler(urdf_path.parent),
        load_meshes=load_meshes,
        build_scene_graph=load_meshes,
        load_collision_meshes=load_collision_meshes,
        build_collision_scene_graph=load_collision_meshes,
    )

    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf,
        load_meshes=load_meshes,
        load_collision_meshes=load_collision_meshes,
        collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
    )
    # Add coordinate frames for each link.
    frame_handles = {}
    if show_frames:
        frame_handles = add_link_frames(server, viser_urdf)

    # Create sliders in GUI that help us move the robot joints.
    with server.gui.add_folder("Joint position control"):
        (slider_handles, initial_config) = create_robot_control_sliders(
            server, viser_urdf
        )

    # Add visibility checkboxes.
    with server.gui.add_folder("Visibility"):
        show_meshes_cb = server.gui.add_checkbox(
            "Show meshes",
            viser_urdf.show_visual,
        )
        show_collision_meshes_cb = server.gui.add_checkbox(
            "Show collision meshes", viser_urdf.show_collision
        )
        show_frames_cb = server.gui.add_checkbox(
            "Show frames", show_frames
        )

    @show_meshes_cb.on_update
    def _(_):
        viser_urdf.show_visual = show_meshes_cb.value

    @show_collision_meshes_cb.on_update
    def _(_):
        viser_urdf.show_collision = show_collision_meshes_cb.value

    @show_frames_cb.on_update
    def _(_):
        for frame in frame_handles.values():
            frame.visible = show_frames_cb.value

    # Hide checkboxes if meshes are not loaded.
    show_meshes_cb.visible = load_meshes
    show_collision_meshes_cb.visible = load_collision_meshes

    # Set initial robot configuration.
    viser_urdf.update_cfg(np.array(initial_config))

    # Create grid.
    trimesh_scene = viser_urdf._urdf.scene or viser_urdf._urdf.collision_scene
    server.scene.add_grid(
        "/grid",
        width=2,
        height=2,
        position=(
            0.0,
            0.0,
            # Get the minimum z value of the trimesh scene.
            trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0,
        ),
    )

    # Create joint reset button.
    reset_button = server.gui.add_button("Reset")

    @reset_button.on_click
    def _(_):
        for s, init_q in zip(slider_handles, initial_config):
            s.value = init_q

    # Sleep forever.
    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)