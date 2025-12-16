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

```bash
cd src
git clone https://github.com/AntonioMacaronio/sam3.git
pip install requests Pillow
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

## aliases

`d`: `distrobox enter ros2 -- bash`
`s`: `source install/setup.bash`
`b`: `colcon build --symlink-install && s`
`start_arm`: `ros2 run ur7e_utils enable_comms`
`estop`: `ros2 run ur7e_utils reset_state`
`freedrive`: `ros2 run ur7e_utils freedrive`
`tuck`: `ros2 run ur7e_utils tuck`

## packages

* `perception`: basically ros2 wrapper of `parse_images`, goes from camera to published tangram poses
* `planning`: determine cube pose, unify TF tree, run IK
* `ros2_aruco`: determine camera pose relative to aruco marker
* `scrape_dataset`: webscrape sample tangram images from web and process them, processing code also works for real images
* `tangram-vlm`: vlm code to generate tangram plan based on text prompt
* `tangram_robot`: full pick and place pipeline
