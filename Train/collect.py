import h5py
import gym
import csv
import pandas as pd
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from termcolor import colored
import time
import math


import my_ppo_net_1
from my_ppo_net_1 import Classic_PPO
import hjbppo
from hjbppo import HJB_PPO
import icm_ppo
from icm_ppo import ICM_PPO

# Env Obstacle Distance
from env_new import train_env_basic_3
device = icm_ppo.device
# roll guide
ppo_path = './models/normal_task/basic2_0611_1.pth' # icm



def save_episode_to_hdf5(file_path, batch_data):
    with h5py.File(file_path, "a") as f:
        # 获取当前文件中已经保存的 episode 数量，确保下一个 episode 索引是正确的
        episode_idx = len(f.keys())  # 获取文件中的组数量（即已有的 episodes 数量）
        
        # 创建一个新的组来保存该 episode
        episode_group = f.create_group(f"episode_{episode_idx}")
        
        # 遍历 batch_data 中的每个键，将每个键的数据作为数据集存储在该组中
        for key, value in batch_data.items():
            if not value:
                continue  # 跳过空数据
            
            # 将每个键的数据保存为数据集
            data = np.array(value, dtype=np.float32)
            episode_group.create_dataset(key, data=data, compression="gzip", chunks=True)
        
        print(f"Episode {episode_idx} saved successfully!")


def contains_nan_inf(arr):
    return np.isnan(arr).any() or np.isinf(arr).any()



def data_collect(save_path):
    print("============================================================================================")
    max_steps = 300
    max_ep_len = max_steps + 20
    
    # continuous action space; else discrete
    has_continuous_action_space = True

    # update policy for K epochs in one PPO update
    K_epochs = 40

    eps_clip = 0.2  # clip rate for PPO
    gamma = 0.99  # discount factor γ

    lr_actor = 0.0001  # learning rate for actor network
    lr_critic = 0.0002  # learning rate for critic network

    action_std_4_test = 0.04  # set std for action distribution when testing.
    switch_range = 0
    random_seed = 0

    # collect flow or not
    _collect_flow_info = False

    #####################################################
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10
    _start = np.array([float(220), float(64)])
    _target = np.array([float(64), float(64)])
    env = train_env_basic_3.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, _include_flow=_collect_flow_info, _plot_flow=False, _proccess_flow=_collect_flow_info, 
    _swich_range=switch_range, _random_range=56, _flow_range=24, _init_flow_num=random_seed, _pos_normalize=True, _is_test=False, u_range=15.0, v_range=15.0)

    state_dim = env.observation_space.shape[0]
    action_output_scale = np.array([])  
    action_output_bias = np.array([])  
    if has_continuous_action_space:
        # action space dimension
        action_dim = env.action_space.shape[0]
        action_output_scale = (env.action_space.high - env.action_space.low) / 2.0
        action_output_bias = (env.action_space.high + env.action_space.low) / 2.0
    else:
        action_dim = env.action_space.n

    # initialize RL agent
    ppo_agent = ICM_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                has_continuous_action_space,
                action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
                continuous_action_output_bias=action_output_bias, icm_alpha=20)
    ppo_agent.load_full_icm(ppo_path)

    #####################################################
    test_time = 1000
    batch_size = 1

    # 根据是否采集 flow 决定 keys
    base_keys = ["episode_id", "state", "action", "pressure", "position", "speed"]
    if _collect_flow_info:
        base_keys.append("flow")

    # 初始化 batch_data
    batch_data = {key: [] for key in base_keys}

    for n in range(test_time):
        # reset
        if _collect_flow_info:
            state, info, flow_info = env.reset()
            flow_info_old = flow_info.copy()
        else:
            state, info = env.reset()
        
        state_old = state.copy()
        info_old = info.copy()

        # episode_data 初始化
        episode_data = {key: [] for key in base_keys}
        episode_data["episode_id"] = n

        for i in range(1, max_ep_len + 1):
            action = ppo_agent.select_action(state)
            step_result = env.step(action)
            
            if _collect_flow_info:
                state, reward, terminated, truncated, info, flow_info = step_result
            else:
                state, reward, terminated, truncated, info = step_result

            # 数据采集
            episode_data["state"].append(state_old)
            episode_data["action"].append(action)
            episode_data["pressure"].append([
                info_old["pressure_1"], info_old["pressure_2"], info_old["pressure_3"],
                info_old["pressure_4"], info_old["pressure_5"], info_old["pressure_6"],
                info_old["pressure_7"], info_old["pressure_8"]
            ])
            episode_data["position"].append([
                info_old["pos_x"], info_old["pos_y"], info_old["angle"]
            ])
            episode_data["speed"].append([
                info_old["vel_x"], info_old["vel_y"], info_old["vel_angle"]
            ])
            if _collect_flow_info:
                episode_data["flow"].append(np.array(flow_info_old, dtype=np.float32))

            # ==== 检查更新后的 state, info, flow_info 是否含 NaN / Inf ====
            has_nan_inf = contains_nan_inf(state) or any(contains_nan_inf(np.array(list(info.values()), dtype=np.float32)))

            if _collect_flow_info:
                has_nan_inf = has_nan_inf or contains_nan_inf(np.array(flow_info, dtype=np.float32))

            # 异常结束
            if has_nan_inf:
                print(colored(f"[Warning] NaN or Inf detected at Episode {n}, Step {i}. Aborting episode.", "red"))
                break

            # 更新 old
            state_old = state.copy()
            info_old = info.copy()
            if _collect_flow_info:
                flow_info_old = flow_info.copy()

            # 正常结束
            if terminated:
                print(f"Episode {n} success")
                break
            
            if truncated:
                print(f"Episode {n} fail")
                break

        # 写入批次缓存
        for key in base_keys:
            batch_data[key].append(episode_data[key])

        # 清空 PPO buffer
        ppo_agent.buffer.clear()

        # 存储
        if (n + 1) % batch_size == 0:
            save_episode_to_hdf5(save_path, batch_data)
            batch_data = {key: [] for key in base_keys}

    # 剩余状态保存
    if batch_data["state"]:
        save_episode_to_hdf5(save_path, batch_data)

    env.close()



if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Collect data and save with a specific name.")
    
    # 添加命令行参数，数字输入
    parser.add_argument("number", type=int, help="Enter a number to generate the save file name")

    # 解析命令行参数
    args = parser.parse_args()

    # 根据输入数字生成 save_path
    save_path = f"./flow_output/flow_data_{args.number}.h5"

    # 调用数据收集函数
    data_collect(save_path)
