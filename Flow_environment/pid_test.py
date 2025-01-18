import gym
import csv
import my_ppo_net
from my_ppo_net import Classic_PPO
import os
import argparse
import torch
import numpy as np
from datetime import datetime
from termcolor import colored
import time
from env import flow_field_env_4
from env import flow_field_env_3
from PIDcontroller import heading_controller
import pandas as pd

dt = 0.075
save_dir = './model_output'
action_name = "action_pid_angle.csv"
file_name = "angle_PID_agnet.csv"


def init_PID(sampletime):
    # init heading controller
    controller = heading_controller(sample_time=sampletime)
    # controller.calculate_controller_params(yaw_time_constant=1)
    return controller


def test():
    print("============================================================================================")
    start = np.array([float(240), float(64)])
    target = np.array([float(190), float(64)])
    max_steps = 170
    max_ep_len = max_steps + 10
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10
    env = flow_field_env_4.foil_env(args_1, max_step=max_steps, target_position=target, start_position=start, include_flow=True)
    # env = flow_field_env_3.foil_env(args_1, max_step=max_steps, target_position=target, start_position=start, include_flow=True)
    ########
    # PID控制器
    controller = init_PID(dt)
    ########

    ########################### PID ###########################
    # 用于保存轨迹的列表
    position_PID = []
    speed_PID = []
    angle_PID = []
    action_PID = []
    angle_speed_PID = []

    state, info = env.reset()
    ref_heading = env.tva
    print("position_reset:", env.agent_pos)
    print("ref_heading:", ref_heading)

    ep_return_ppo = 0
    total_steps_ppo = 0
    for i in range(1, max_ep_len + 1):
        current_angle = env.angle
        action_angle = controller.controll(ref_heading, current_angle, state[2])
        action = [-20, 0, action_angle]
        print(f"Step {i}, Action: {action}")
        action_PID.append(np.degrees(action[2]))
        state, reward, terminated, truncated, info = env.step(action)
        print(f"Vel_angle = {env.state[2]}")
        ep_return_ppo += reward
        total_steps_ppo = i
        position_PID.append([info["pos_x"], info["pos_y"]])
        speed_PID.append([info["vel_x"], info["vel_y"]])
        angle_PID.append(np.degrees(info["angle"]))
        angle_speed_PID.append(np.degrees(info["vel_angle"]))
        if terminated:
            break
        if truncated:
            break
    env.close()

    print("Total_Reward:", ep_return_ppo)
    print("Total_Steps:", total_steps_ppo)
    data = {
        "action_PID": action_PID,
        "angle_PID": angle_PID,
        "angle_speed_PID": angle_speed_PID
    }

    # 将数据转换为 Pandas DataFrame
    df = pd.DataFrame(data)

    # 保存为 CSV 文件
    output_filename = os.path.join(save_dir, file_name)
    df.to_csv(output_filename, index=False)
    # output_filename = os.path.join(save_dir, action_name)
    # # 打开文件进行写入
    # with open(output_filename, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["a_x", "a_y", "a_w"])
    #     # 写入每个动作的数据
    #     for action in action_PID:
    #         # 假设 action 是一个包含三个元素的列表
    #         writer.writerow(action)


if __name__ == '__main__':
    test()
