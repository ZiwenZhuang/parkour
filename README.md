# Robot Parkour Learning #

**Project website**: [https://robot-parkour.github.io/](https://robot-parkour.github.io/)

<p align="center">
  <img src="images/teaser.jpeg" width="80%"/>
</p>

This codebase is contains implementation for training and visualizing the result of paper [Robot Parkour Learning](https://openreview.net/forum?id=uo937r5eTE)

To install and run the code, please clone this repository and follow the instructions in [legged_gym/README.md](legged_gym/README.md)

## Repository Structure ##

* `legged_gym`: contains the isaacgym environment and config files.
    - `legged_gym/legged_gym/envs/a1/`: contains all the training config files.
    - `legged_gym/legged_gym/envs/base/`: contains all the environment implementation.
    - `legged_gym/legged_gym/utils/terrain/`: contains the terrain generation code.
* `rsl_rl`: contains the network module and algorithm implementation. You can copy this folder directly to your robot.
    - `rsl_rl/rsl_rl/algorithms/`: contains the algorithm implementation.
    - `rsl_rl/rsl_rl/modules/`: contains the network module implementation.

## Trouble Shooting ##

If you cannot run the distillation part or all graphics computing goes to GPU 0 dispite you have multiple GPUs and have set the CUDA_VISIBLE_DEVICES, please use docker to isolate each GPU.

## To Do ##

The code is currently only for training and visualizing in simulation.

The code and instructions for real robot is on the way.

**Before November 2023, the code for real robot (A1 and Go1) and the checkpoint will be released.**

## Citation ##
If you find this code useful in your research, please consider citing:

```
@inproceedings{
    zhuang2023robot,
    title={Robot Parkour Learning},
    author={Ziwen Zhuang and Zipeng Fu and Jianren Wang and Christopher G Atkeson and S{\"o}ren Schwertfeger and Chelsea Finn and Hang Zhao},
    booktitle={7th Annual Conference on Robot Learning},
    year={2023},
    url={https://openreview.net/forum?id=uo937r5eTE}
}
```