# tangram robot

code for tangram robot project.

## setup instructions

```bash
cd ~
mkdir ros2_ws
cd ros2_ws
git clone https://github.com/anshgandhi4/tangram_robot.git src
```

If this doesn't work (or prompts you to use GitHub login), try using SSH cloning instead:
```bash
git clone git@github.com:anshgandhi4/tangram_robot.git src
```

## run code

Run the following commands in separate distrobox terminals:

```bash
ros2 run ur7e_utils enable_comms
```

```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch planning lab7_bringup.launch.py ar_marker:=ar_marker_<marker id>
```

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 run planning main
```

## packages

* `perception`: filter pointcloud
* `planning`: determine cube pose, unify TF tree, run IK, full pick and place pipeline
* `ros2_aruco`: determine camera pose relative to aruco marker
* `tangram_robot`: nothing for now
