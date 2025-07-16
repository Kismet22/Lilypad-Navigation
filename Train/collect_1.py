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
from env_new import train_env_upper_2
device = icm_ppo.device
# roll guide
ppo_path = './models/normal_task/basic2_0611_1.pth' # icm
ppo_path_lc = './models/easy_task/0705_lc.pth' # all direction
# ppo_path_hc = './models/hrl/0710_dim18_action2.pth' # all

def compute_subgoal_forward(pos, angle, r):
    x, y = pos[0], pos[1]
    # angle_1 = angle.item() + np.pi/2
    angle_1 = angle + np.pi/2
    dx = r * np.cos(angle_1)
    dy = r * np.sin(angle_1)
    return x + dx, y + dy

def compute_subgoal(pos, angle, r):
    x, y = pos[0], pos[1]
    # angle_1 = angle.item()
    angle_1 = angle
    dx = r * np.cos(angle_1)
    dy = r * np.sin(angle_1)
    return x + dx, y + dy


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

    _collect_flow_info = False

    #####################################################
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10
    _start = np.array([float(220), float(64)])
    _target = np.array([float(64), float(64)])
    env = train_env_basic_3.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, _include_flow=_collect_flow_info, _plot_flow=False, _proccess_flow=_collect_flow_info, _swich_range=switch_range, _random_range=56, _flow_range=24, 
    _init_flow_num=random_seed, _pos_normalize=True, _is_test=True, _state_dim=14, u_range=15.0, v_range=15.0)

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
    # ppo_agent = ICM_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
    #             has_continuous_action_space,
    #             action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
    #             continuous_action_output_bias=action_output_bias, icm_alpha=20)
    # ppo_agent.load_full_icm(ppo_path)

    ppo_agent = Classic_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space,
                        action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
                        continuous_action_output_bias=action_output_bias)
    ppo_agent.load_full(ppo_path_lc)

    #####################################################
    test_time = 1000
    collect_steps = test_time * max_ep_len
    batch_size = 1
    total_steps = 0
    n = 0

    base_keys = ["episode_id", "state", "action", "pressure", "position", "speed"]
    if _collect_flow_info:
        base_keys.append("flow")
    batch_data = {key: [] for key in base_keys}

    subgoal_threshold = 1
    max_ll_steps = 5
    large_jump_prob = 0.1

    while total_steps < collect_steps:
        # === Reset ===
        if _collect_flow_info:
            state, info, flow_info = env.reset()
            flow_info_old = flow_info.copy()
        else:
            state, info = env.reset()
        state_old = state.copy()
        info_old = info.copy()
        agent_position = env.agent_pos

        # Episode init
        episode_data = {key: [] for key in base_keys}
        episode_data["episode_id"] = n

        lc_angle_old = np.pi / 2
        t = 0
        episode_valid = True

        print(f"Episode {n} reset at position {agent_position}")

        while t < max_ep_len:
            # === High-level sampling ===
            if np.random.rand() < large_jump_prob:
                lc_angle = np.random.uniform(0, np.pi)
            else:
                lc_angle_add = np.random.uniform(-np.pi / 6, np.pi / 6)
                lc_angle = np.clip(lc_angle_old + lc_angle_add, 0, np.pi)
            lc_angle_old = lc_angle

            lc_target_x, lc_target_y = compute_subgoal_forward(pos=agent_position, angle=lc_angle, r=5)

            ll_step = 0
            while ll_step < max_ll_steps:
                # Construct low-level state with subgoal offset
                low_state = state.copy()
                low_state[3] = lc_target_x - agent_position[0]
                low_state[4] = lc_target_y - agent_position[1]
                low_action = ppo_agent.select_action(low_state)

                if _collect_flow_info:
                    state, reward, terminated, truncated, info, flow_info = env.step(low_action)
                else:
                    state, reward, terminated, truncated, info = env.step(low_action)

                # ========== NaN/Inf 检测 ========== 
                check_arrays = [
                    state, 
                    np.array(list(info.values()), dtype=np.float32)
                ]
                if _collect_flow_info:
                    check_arrays.append(np.array(flow_info, dtype=np.float32))
                if any(contains_nan_inf(arr) for arr in check_arrays):
                    print(f"[Warning] NaN or Inf detected at episode {n}, step {t}. Dropping episode.")
                    episode_valid = False
                    break

                # ========== Save data ==========
                episode_data["state"].append(state_old)
                episode_data["action"].append(low_action)
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

                # Update for next step
                state_old = state.copy()
                info_old = info.copy()
                if _collect_flow_info:
                    flow_info_old = flow_info.copy()
                agent_position = env.agent_pos

                # Check subgoal completion
                if np.linalg.norm(agent_position - np.array([lc_target_x, lc_target_y])) < subgoal_threshold or terminated or truncated :
                    ppo_agent.buffer.clear()
                    break

                t += 1
                total_steps += 1
                ll_step += 1

            if terminated or truncated or not episode_valid:
                break

        if episode_valid:
            for key in base_keys:
                batch_data[key].append(episode_data[key])
            print(f"Episode {n} saved with {len(episode_data['state'])} steps.")
        else:
            print(f"Episode {n} dropped due to NaN/Inf.")

        ppo_agent.buffer.clear()

        # Save batch
        if (n + 1) % batch_size == 0 and batch_data["state"]:
            save_episode_to_hdf5(save_path, batch_data)
            print(f"[Batch Saved] Up to episode {n}")
            batch_data = {key: [] for key in base_keys}

        n += 1

    # Save remaining
    if batch_data["state"]:
        save_episode_to_hdf5(save_path, batch_data)
        print("[Final Batch Saved] Remaining episodes.")

    env.close()


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Collect data and save with a specific name.")
    
    # 添加命令行参数，数字输入
    parser.add_argument("number", type=int, help="Enter a number to generate the save file name")

    # 解析命令行参数
    args = parser.parse_args()

    # 根据输入数字生成 save_path
    save_path = f"./flow_output_07/flow_data_{args.number}.h5"

    # 调用数据收集函数
    data_collect(save_path)
