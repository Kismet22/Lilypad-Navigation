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
import pandas as pd
# from env import flow_field_env_2
from env import flow_field_env_3

# ppo_path = './models/PPO_1.pth' # [15, 8]范围模型,锁旋转自由度，定起点终点[240,64]和[73,64]，能实现成功导航
# ppo_path = './models/PPO_a2.pth' # 20241227训练模型
# ppo_path = './models/PPO_a3.pth' # 20241229训练模型
ppo_path = './models/PPO_a4.pth' # 20250109训练模型
save_dir = './model_output'
action_name = "action_ppo_angle.csv"
file_name = "angle_ppo_agnet.csv"


def test():
    print("============================================================================================")

    # max_time_step = 1000
    # max_ep_len = max_time_step + 100
    max_time_step = 4000
    max_ep_len = max_time_step + 100

    # 判断动作空间是否连续
    # continuous action space; else discrete
    has_continuous_action_space = True

    # update policy for K epochs in one PPO update
    K_epochs = 40

    eps_clip = 0.2  # clip rate for PPO
    gamma = 0.99  # discount factor γ

    lr_actor = 0.0001  # learning rate for actor network
    lr_critic = 0.0002  # learning rate for critic network

    action_std_4_test = 0.04  # set std for action distribution when testing.
    random_seed = 10

    #####################################################
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10
    # target = np.array([200, 64])
    # start = np.array([float(150), float(64)])
    # env = flow_field_env_2.foil_env(args_1, max_step=max_time_step, target_position=target, include_flow=True)
    # env = flow_field_env_2.foil_env(args_1, max_step=max_time_step, include_flow=True) # 状态空间维度6
    # env = flow_field_env_1.foil_env(args_1, max_step=max_time_step, include_flow=True) # 状态空间维度14
    start = np.array([float(150), float(64)])
    target = np.array([float(64), float(64)])
    max_steps = 170
    env = flow_field_env_3.foil_env(args_1, max_step=max_steps, target_position=target, start_position=start, include_flow=True)
    # 状态空间
    # state space dimension
    state_dim = env.observation_space.shape[0]

    # 动作空间
    action_output_scale = np.array([])
    action_output_bias = np.array([])
    if has_continuous_action_space:
        # action space dimension
        action_dim = env.action_space.shape[0]
        # 输出动作缩放的相关内容
        action_output_scale = (env.action_space.high - env.action_space.low) / 2.0
        action_output_bias = (env.action_space.high + env.action_space.low) / 2.0

    else:
        # 离散动作空间，输出可选的动作数
        action_dim = env.action_space.n

    # initialize RL agent
    ppo_agent = Classic_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                            has_continuous_action_space,
                            action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
                            continuous_action_output_bias=action_output_bias, mini_batch_size=32)

    ppo_agent.load_full(ppo_path)

    ########################### PPO ###########################
    # 用于保存轨迹的列表
    position_ppo = []
    speed_ppo = []
    angle_ppo = []
    action_ppo = []
    angle_speed_ppo = []

    state, info = env.reset()
    print("position_reset:", env.agent_pos)

    ep_return_ppo = 0
    total_steps_ppo = 0
    for i in range(1, max_ep_len + 1):
        action = ppo_agent.select_action(state)
        print(f"Step {i}, Action: {action}")
        # action_ppo.append(action)
        action_ppo.append(np.degrees(action[2]))
        state, reward, terminated, truncated, info = env.step(action)
        ep_return_ppo += reward
        total_steps_ppo = i
        position_ppo.append([info["pos_x"], info["pos_y"]])
        speed_ppo.append([info["vel_x"], info["vel_y"]])
        angle_ppo.append(np.degrees(info["angle"]))
        angle_speed_ppo.append(np.degrees(info["vel_angle"]))
        if terminated:
            break
        if truncated:
            break
        # clear buffer
        ppo_agent.buffer.clear()
    env.close()

    print("Total_Reward:", ep_return_ppo)
    print("Total_Steps:", total_steps_ppo)
    data = {
        "action_ppo": action_ppo,
        "angle_ppo": angle_ppo,
        "angle_speed_ppo": angle_speed_ppo
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
    #     for action in action_ppo:
    #         # 假设 action 是一个包含三个元素的列表
    #         writer.writerow(action)


if __name__ == '__main__':
    test()
