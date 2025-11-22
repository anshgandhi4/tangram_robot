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

```bash
python3 -m venv --system-site-packages ~/tangram_venv
source ~/tangram_venv/bin/activate
pip install -r ~/ros2_ws/src/requirements.txt
```

## run code

Run the following commands in separate distrobox terminals:

```bash
ros2 run ur7e_utils enable_comms
```

```bash
cd ~/ros2_ws
source ~/tangram_venv/bin/activate
colcon build --symlink-install
source install/setup.bash
ros2 launch tangram_robot tangram.launch.py ar_marker:=ar_marker_<marker id>
```

```bash
cd ~/ros2_ws
source ~/tangram_venv/bin/activate
source install/setup.bash
ros2 run tangram_robot main
```

## packages

* `perception`: filter pointcloud
* `planning`: determine cube pose, unify TF tree, run IK
* `ros2_aruco`: determine camera pose relative to aruco marker
* `tangram_robot`: full pick and place pipeline
