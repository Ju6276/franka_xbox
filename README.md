# Intro:
This package provide a simple Franka arm and Franka hand gripper simulator written in Mujoco.
It includes a state-based and a vision-based Franka gear assembly task with xbox controller environment.
![default](https://github.com/user-attachments/assets/287d8f8d-0643-45ee-a3f4-d85638544abe)

![_cgi-bin_mmwebwx-bin_webwxgetmsgimg__ MsgID=7776618432080307361 skey=@crypt_4b158ffb_db054a24e9d190929e83248be22e3022 mmweb_appid=wx_webfilehelper](https://github.com/user-attachments/assets/58a1e9aa-0d5a-4ebf-b07c-4a41940b97a3)


# Installation:
- cd into `franka_sim`.
- In your conda environment, run `pip install -e .` to install this package.
- run `pip install -r requirements.txt` to install sim dependencies.

# Explore the Environments
- Run `python franka_sim/test/test_gym_env_human.py` to launch a display window and visualize the task.
- Run `python franka_sim/test/teleoperation.py --mode task --task gear --input xbox` to use an Xbox controller on the gear task.
- Run `python franka_sim/test/teleoperation.py --mode task --task gear --input keyboard` to use keyboard teleoperation on the gear task.
- Run `python franka_sim/test/teleoperation.py --mode controller --input keyboard --print-joints` to teleoperate a Franka-only scene and print the solved 7D arm joint command.
- Run `python franka_sim/test/xbox_game_controller.py` to test your xbox controller hardware

## Main Workflow Commands
- Run `Python franka_sim/test/teleoperation.py --mode controller --input xbox --record --save-images --show-camera-view` to record teleoperation data.
- Run `python franka_sim/test/convert_to_lerobot.py --raw-dir dataset_raw --out-dir demo_data_converted --task-name franka_teleop --robot-type franka --fps 50` to convert raw recordings to LeRobot-style dataset
- Run `python franka_sim/test/replay_dataset.py --parquet demo_data_converted/data/chunk-000/episode_000000.parquet --fps 50` to replay a record episode.


Keyboard controls:
- Keep the `FR3 Keyboard Teleoperation` pygame window focused so MuJoCo viewer shortcuts do not receive the teleop keys.
- `W/S` or `Up/Down`: move +Y/-Y
- `D/A` or `Right/Left`: move +X/-X
- `E/Q`: move +Z/-Z
- `L/J`: roll +/-
- `I/K`: pitch +/-
- `O/U`: yaw +/-
- `Z/X` or `[/]`: open/close gripper
- `C`: stop gripper
- `Space`: reset environment
- `Esc`: exit

# Credits:
- This simulation is initially built by [Kevin Zakka](https://kzakka.com/).


# Notes:
- Error due to `egl` when running on a CPU machine:
```bash
export MUJOCO_GL=egl
conda install -c conda-forge libstdcxx-ng
```
# Franka-Research-3-Robot-Simulation-with-Xbox-Controller-Integration-in-MuJoCo
