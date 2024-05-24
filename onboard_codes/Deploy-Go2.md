# Deploy the model on your real Unitree Go2 robot

This file shows an example of how to deploy the model on the Unittree Go2 robot (with Nvidia Jetson NX).

The code is a quick start of the deployment and fit the simulation as much as possible. You can modify the code to fit your own project.

## Install dependencies on Go2

1. Take Nvidia Jetson Orin as an exmaple, make sure your JetPack and related software are up-to-date.

2. Install ROS and the [unitree ros package for Go2](https://support.unitree.com/home/en/developer/ROS2_service)

3. Set up a folder on your robot for this project, e.g. `parkour`. Then `cd` into it.

4. Create a python virtual env and install the dependencies.

    - Install pytorch on a Python 3 environment.
        ```bash
        sudo apt-get install python3-pip python3-dev python3-venv
        python3 -m venv parkour_venv
        source parkour_venv/bin/activate
        ```
    
    - Download the pip wheel file from [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) with v1.10.0. Then install it with
        ```bash
        pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
        ```

    - Install `ros2-numpy` from [here](https://github.com/nitesh-subedi/ros2_numpy) in a new colcon_ws, where you prefer.
        ```bash
        pip install transformations pybase64
        mkdir -p ros2_numpy_ws/src
        cd ros2_numpy_ws/src
        git clone https://github.com/nitesh-subedi/ros2_numpy.git
        cd ../
        colcon build
        ```

4. Copy folders of this project.
    - copy the `rsl_rl` folder to the `parkour` folder.
    - copy the distilled parkour log folder (e.g. **Jul18_07-22-08_Go2_10skills_fromJul16_07-38-08**) to the `parkour` folder. 
    - copy all the files in `onboard_script/go2` to the `parkour` folder.

3. Install rsl_rl and other dependencies.
    ```bash
    pip install -e ./rsl_rl
    ```

## Run the model on Go2

***Disclaimer:*** *Always put a safety belt on the robot when the robot moves. The robot may fall down and cause damage to itself or the environment.*

1. Put the robot on the ground, power on the robot, and **turn off the builtin sport service**. Make sure your Intel Realsense D435i camera is connected to the robot and the camera is installed where you calibrated the camera.

    > To turn off the builtin sport service, please refer to the [official guide](https://support.unitree.com/home/zh/developer/Basic_motion_control) and [official example](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/low_level/stand_example_go2.cpp#L184)

2. Launch 2 terminals onboard (whether 2 ssh connections from your computer or something else), named T_visual, T_run. Source the Unitree ROS environment, `ros2_numpy_ws` and the python virtual environment in both terminals.

3. In T_visual, run
    ```bash
    cd parkour
    python go2_visual.py --logdir Jul18_07-22-08_Go2_10skills_fromJul16_07-38-08
    ```
    where `Jul18_07-22-08_Go2_10skills_fromJul16_07-38-08` is the logdir of the distilled model.

4. In T_run, run
    ```bash
    cd parkour
    python go2_run.py --logdir Jul18_07-22-08_Go2_10skills_fromJul16_07-38-08
    ```
    where `Jul18_07-22-08_Go2_10skills_fromJul16_07-38-08` is the logdir of the distilled model.

    Currently, the robot will not actually move its motors. You may see the ros topics. If you want to let the robot move, you can add argument `--nodryrun` in the command line, but **be careful**.
