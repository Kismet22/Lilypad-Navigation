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


# New Env
from env_new import train_env_basic_1
from env_new import train_env_basic_2
from env_new import train_env
from env_new import train_env_path_follow
from env_new import train_env_path_follow_1
from env_new import test_env_upper
from env_new import train_env_upper
from env_new import train_env_upper_2

# Old Env
from env import flow_field_env_2_1



# low-level controller
# ppo_path_1 = './models/easy_task/0612_10.pth'
# ppo_path_1 = './models/easy_task/0629_2.pth' # all direction
# ppo_path_1 = './models/easy_task/0704_seed1.pth'
ppo_path_1 = './models/easy_task/0705_lc.pth' # all direction

# high-level controller
# ppo_path = './models/hrl/0618_0.pth' #  forward
# ppo_path = './models/hrl/0703_h.pth'
# ppo_path = './models/hrl/0704_dim18_action1_seed1.pth'
ppo_path = './models/hrl/0710_dim18_action2.pth' # all
# ppo_path = './models/hrl/0710_dim16_action2.pth'
# ppo_path = './models/hrl/0711_dim18_action1.pth'


save_dir = './SuccessTest'
save_dir_1 = './model_output'
# plot_save_dir = './model_output/success_plot_withp.png'

def compute_subgoal_forward(pos, angle, r):
    # print(f"target angle:{angle.item()}")
    x, y = pos[0], pos[1]
    angle_1 = angle.item() + np.pi/2
    dx = r * np.cos(angle_1)
    dy = r * np.sin(angle_1)
    return x + dx, y + dy

def compute_subgoal(pos, angle, r):
    x, y = pos[0], pos[1]
    angle_1 = angle.item()
    dx = r * np.cos(angle_1)
    dy = r * np.sin(angle_1)
    return x + dx, y + dy

def plot_success(all_position_records, success_flags, success_time, total_time, start_center, target_center, save_dir=None):
    """ Plot success/fail/collision trajectories """
    fig, ax = plt.subplots(figsize=(16, 9))
    circles = [
        {"center": (53, 32), "radius": 8},
        {"center": (53, 96), "radius": 8},
        {"center": (109, 64), "radius": 8}
    ]

    min_x = 0
    max_x = 20 * 16
    min_y = 0
    max_y = 8 * 16
    # edge margin
    x_margin = 2
    y_margin = 2
    ax.set_xlim(min_x - x_margin, max_x + x_margin)
    ax.set_ylim(min_y - y_margin, max_y + y_margin)

    for circle in circles:
        ax.add_patch(
            plt.Circle(circle["center"], circle["radius"], color='red', alpha=0.3)
        )

    # random range
    start_circle = plt.Circle(start_center, 56, color='green', fill=False, linestyle='--')
    ax.add_artist(start_circle)
    target_circle = plt.Circle(target_center, 56, color='red', fill=False, linestyle='--')
    ax.add_artist(target_circle)

    # Line settings
    line_alpha = 0.8  # transparency
    line_width = 0.2

    # success_flag 1:success 2:collision 3:fail
    color_map = {1: 'green', 2: 'blue', 0: 'red'}

    for record, success_flag in zip(all_position_records, success_flags):
        x_values = [pos[0] for pos in record]
        y_values = [pos[1] for pos in record]
        line_color = color_map.get(success_flag, 'black')
        ax.plot(x_values, y_values, color=line_color, linestyle='-', linewidth=line_width, alpha=line_alpha)
    ax.set_xlabel('X/m')
    ax.set_ylabel('Y/m')
    ax.set_title(f'Success Rate:{success_time}/{total_time}')
    plt.grid(True)
    if save_dir:
        plt.savefig(save_dir, dpi=300)
    # plt.show()


def read_and_check(file_path):
    """ read .csv flie,convert to numpy array """
    if not os.path.exists(file_path):
        print(f"File {file_path} Not Exist!")
        return None, 0
    try:
        df = pd.read_csv(file_path)
        df_numpy = df.to_numpy()
        _length = df.shape[0]
        return df_numpy, _length
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, 0

def _save_trajectories(file_name, _trajectories):
    """
    save positions as ".csv"
    :param file_name:save path
    :param _trajectories: trajectory list(start_x, start_y, target_x, target_y)
    """
    if not _trajectories:
        print("No trajectories to save.")
        return
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["start_x", "start_y", "target_x", "target_y", "dis_2_target"])
        writer.writerows(_trajectories)
    print(f"trajectories saved to {file_name}")

def success_rate_test_1(_seed=1):
    ########################### Mode 2: Multi-Test-Random ###########################
    print("============================================================================================")
    ########################### Environmenrt ###########################
    max_steps = 400
    max_ep_len = max_steps + 20
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10
    # Flow ID : 0 Random Flow
    random_seed = _seed

     ######################################################
    _start = np.array([float(220), float(64)])
    _target = np.array([float(64), float(64)])
    env = train_env_path_follow.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, _random_range=56, _init_flow_num=random_seed, _pos_normalize=True)
    ######################################################

    ########################### State & Action ###########################
    # state space dimension
    # continuous action space; else discrete
    has_continuous_action_space = True
    state_dim = env.observation_space.shape[0]
    action_output_scale = np.array([])
    action_output_bias = np.array([])
    if has_continuous_action_space:
        # action space dimension
        action_dim = env.action_space.shape[0]
        # action scale
        action_output_scale = (env.action_space.high - env.action_space.low) / 2.0
        action_output_bias = (env.action_space.high + env.action_space.low) / 2.0
    else:
        # discrete
        action_dim = env.action_space.n
    ######################################################

    ########################### PPO ###########################
    # Initialize Parameter
    # update policy for K epochs in one PPO update
    K_epochs = 40
    # clip rate for PPO
    eps_clip = 0.2
    # discount factor γ  
    gamma = 0.99
    # learning rate for actor network  
    lr_actor = 0.0001  
    # learning rate for critic network
    lr_critic = 0.0002  
    # set std for action distribution when testing.
    action_std_4_test = 0.04

    # initialize RL agent
    # ppo_agent = Classic_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
    #                         has_continuous_action_space,
    #                         action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
    #                         continuous_action_output_bias=action_output_bias)
    # ppo_agent.load_full(ppo_path)
    ppo_agent_1 = Classic_PPO(14, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space,
                        action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
                        continuous_action_output_bias=action_output_bias)
    ppo_agent_1.load_full(ppo_path_1)
    ppo_agent = ICM_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space,
                        action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
                        continuous_action_output_bias=action_output_bias, icm_alpha=20)
    ppo_agent.load_full_icm(ppo_path)
    ######################################################

    ########################### Test ###########################
    # Load Position Flie
    _id = _seed
    print(f"Init Seed:{_id}")
    test_time = 1000
    print(f"Total Test Time: {test_time}")

    # record parameters init
    success_times = 0
    collide_times = 0
    timeout_times = 0
    outbound_times = 0
    # navigation_time(success)
    count_time = []
    success_trajectories = []
    collide_trajectories = []
    outbound_trajectories = []
    timeout_trajectories = []
    # Test
    for n in range(test_time):
        # 1 env reset
        strategy_switch = False
        state, info = env.reset()
        state_14 = env.state_14
        print(f"Episode{n} Start")
        print("position_reset:", env.agent_pos)
        for i in range(1, max_ep_len + 1):
            # 2 env step
            if strategy_switch:
                action = ppo_agent_1.select_action(state_14)
            else:
                action = ppo_agent.select_action(state)
            state, reward, _success, _bound, _timeout, _collide, info= env.step(action)
            state_14 = env.state_14
            if env.d_2_target < 24 and not strategy_switch:
                strategy_switch = True
                print(colored(f"**** Strategy Switch ****", 'cyan'))
            if _success:
                print(colored(f"Episode{n} success", 'green'))
                success_times += 1
                success_trajectories.append([env.start_position[0], env.start_position[1],env.target_position[0],env.target_position[1],env.d_2_target_min])
                count_time.append(env.step_counter * env.dt)
                break
            elif _collide:
                print(colored(f"Episode{n} collide", 'yellow'))
                collide_times += 1
                collide_trajectories.append([env.start_position[0], env.start_position[1],env.target_position[0],env.target_position[1],env.d_2_target_min])
                break
            elif _bound:
                print(colored(f"Episode{n} outbound", 'red'))
                outbound_times += 1
                outbound_trajectories.append([env.start_position[0], env.start_position[1],env.target_position[0],env.target_position[1],env.d_2_target_min])
                break
            elif _timeout:
                print(colored(f"Episode{n} timeout", 'blue'))
                timeout_times += 1
                timeout_trajectories.append([env.start_position[0], env.start_position[1],env.target_position[0],env.target_position[1],env.d_2_target_min])
                break
        ppo_agent.buffer.clear()
    env.close()

    average_time = np.mean(count_time)
    print(colored(f"**** Test: {test_time} ****", 'white'))
    print(colored(f"**** Collision: {collide_times} ****", 'yellow'))
    print(colored(f"**** Success: {success_times} ****", 'green'))
    print(colored(f"**** OutBound: {outbound_times} ****", 'red'))
    print(colored(f"**** TimeOut: {timeout_times} ****", 'blue'))
    print("Average Navigation Time:", average_time)
    # name of ppo method
    ppo_name = os.path.splitext(os.path.basename(ppo_path))[0]
    success_name = f"{ppo_name}_success_random_{_id}.csv"
    collide_name = f"{ppo_name}_collide_random_{_id}.csv"
    outbound_name = f"{ppo_name}_outbound_random_{_id}.csv"
    timeout_name = f"{ppo_name}_timeout_random_{_id}.csv"
    success_filename = os.path.join(save_dir, success_name)
    _save_trajectories(file_name=success_filename, _trajectories=success_trajectories)
    collide_filename = os.path.join(save_dir, collide_name)
    _save_trajectories(file_name=collide_filename, _trajectories=collide_trajectories)
    outbound_filename = os.path.join(save_dir, outbound_name)
    _save_trajectories(file_name=outbound_filename, _trajectories=outbound_trajectories)
    timeout_filename = os.path.join(save_dir, timeout_name)
    _save_trajectories(file_name=timeout_filename, _trajectories=timeout_trajectories)


def success_rate_test(_id_in=1):
    ########################### Mode 1: Multi-Test ###########################
    print("============================================================================================")
    ########################### Environmenrt ###########################
    max_steps = 400
    max_ep_len = max_steps + 20
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10
    # Flow ID : 0 Random Flow
    random_seed = 1

     ######################################################
    _start = np.array([float(220), float(64)])
    _target = np.array([float(64), float(64)])
    # env = train_env_path_follow.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, _random_range=56, _init_flow_num=random_seed, _pos_normalize=True)
    env = train_env_path_follow_1.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, _random_range=56, _init_flow_num=random_seed, _pos_normalize=True, _is_test=True)
    ######################################################

    ########################### State & Action ###########################
    # state space dimension
    # continuous action space; else discrete
    has_continuous_action_space = True
    state_dim = env.observation_space.shape[0]
    action_output_scale = np.array([])
    action_output_bias = np.array([])
    if has_continuous_action_space:
        # action space dimension
        action_dim = env.action_space.shape[0]
        # action scale
        action_output_scale = (env.action_space.high - env.action_space.low) / 2.0
        action_output_bias = (env.action_space.high + env.action_space.low) / 2.0
    else:
        # discrete
        action_dim = env.action_space.n
    ######################################################

    ########################### PPO ###########################
    # Initialize Parameter
    # update policy for K epochs in one PPO update
    K_epochs = 40
    # clip rate for PPO
    eps_clip = 0.2
    # discount factor γ  
    gamma = 0.99
    # learning rate for actor network  
    lr_actor = 0.0001  
    # learning rate for critic network
    lr_critic = 0.0002  
    # set std for action distribution when testing.
    action_std_4_test = 0.04

    # initialize RL agent
    # ppo_agent = Classic_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
    #                         has_continuous_action_space,
    #                         action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
    #                         continuous_action_output_bias=action_output_bias)
    # ppo_agent.load_full(ppo_path)
    ppo_agent_1 = Classic_PPO(14, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space,
                        action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
                        continuous_action_output_bias=action_output_bias)
    ppo_agent_1.load_full(ppo_path_1)
    ppo_agent = ICM_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space,
                        action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
                        continuous_action_output_bias=action_output_bias, icm_alpha=20)
    ppo_agent.load_full_icm(ppo_path)
    ######################################################

    ########################### Test ###########################
    # Load Position Flie
    _id = _id_in
    print(f"Init ID:{_id}")
    point_file = "./position_init"

    # start_name = f"start_{_id}.csv"
    # start_filename = os.path.join(point_file, start_name)
    # start_list, start_len = read_and_check(start_filename)

    start_len = 1e5
    target_name = f"target_{_id}.csv"
    # target_name = f"target_near_{_id}.csv"
    target_filename = os.path.join(point_file, target_name)
    target_list, target_len = read_and_check(target_filename)

    test_time = min(start_len, target_len)
    print(f"Total Test Time: {test_time}")

    # record parameters init
    success_times = 0
    collide_times = 0
    timeout_times = 0
    outbound_times = 0
    # navigation_time(success)
    count_time = []
    success_trajectories = []
    collide_trajectories = []
    outbound_trajectories = []
    timeout_trajectories = []
    # Test
    # for n, (start, target) in enumerate(zip(start_list[:test_time], target_list[:test_time])):
    for n, target in enumerate(zip(target_list[:test_time])):
        # 1 env reset
        # start = np.array([float(240), float(64)])
        start = np.array([float(240), float(32)])
        # start = np.array([float(240), float(96)])
        target = target[0]  # 解包 tuple，只保留 numpy array
        strategy_switch = False
        state, info = env.reset(_init_start=start, _init_target=target)
        state_14 = env.state_14
        print(f"Episode{n} Start")
        print("position_reset:", env.agent_pos)
        for i in range(1, max_ep_len + 1):
            # 2 env step
            if strategy_switch:
                action = ppo_agent_1.select_action(state_14)
            else:
                action = ppo_agent.select_action(state)
            state, reward, _success, _bound, _timeout, _collide, info= env.step(action)
            state_14 = env.state_14
            if env.d_2_target < 12 and not strategy_switch:
                strategy_switch = True
                print(colored(f"**** Strategy Switch ****", 'cyan'))
            if _success:
                print(colored(f"Episode{n} success", 'green'))
                success_times += 1
                success_trajectories.append([env.start_position[0], env.start_position[1],env.target_position[0],env.target_position[1],env.d_2_target_min])
                count_time.append(env.step_counter * env.dt)
                break
            elif _collide:
                print(colored(f"Episode{n} collide", 'yellow'))
                collide_times += 1
                collide_trajectories.append([env.start_position[0], env.start_position[1],env.target_position[0],env.target_position[1],env.d_2_target_min])
                break
            elif _bound:
                print(colored(f"Episode{n} outbound", 'red'))
                outbound_times += 1
                outbound_trajectories.append([env.start_position[0], env.start_position[1],env.target_position[0],env.target_position[1],env.d_2_target_min])
                break
            elif _timeout:
                print(colored(f"Episode{n} timeout", 'blue'))
                timeout_times += 1
                timeout_trajectories.append([env.start_position[0], env.start_position[1],env.target_position[0],env.target_position[1],env.d_2_target_min])
                break
        ppo_agent.buffer.clear()
    env.close()

    average_time = np.mean(count_time)
    print(colored(f"**** Test: {test_time} ****", 'white'))
    print(colored(f"**** Collision: {collide_times} ****", 'yellow'))
    print(colored(f"**** Success: {success_times} ****", 'green'))
    print(colored(f"**** OutBound: {outbound_times} ****", 'red'))
    print(colored(f"**** TimeOut: {timeout_times} ****", 'blue'))
    print("Average Navigation Time:", average_time)
    # name of ppo method
    ppo_name = os.path.splitext(os.path.basename(ppo_path))[0]
    success_name = f"{ppo_name}_success_{_id}_32.csv"
    collide_name = f"{ppo_name}_collide_{_id}_32.csv"
    outbound_name = f"{ppo_name}_outbound_{_id}_32.csv"
    timeout_name = f"{ppo_name}_timeout_{_id}_32.csv"
    success_filename = os.path.join(save_dir, success_name)
    _save_trajectories(file_name=success_filename, _trajectories=success_trajectories)
    collide_filename = os.path.join(save_dir, collide_name)
    _save_trajectories(file_name=collide_filename, _trajectories=collide_trajectories)
    outbound_filename = os.path.join(save_dir, outbound_name)
    _save_trajectories(file_name=outbound_filename, _trajectories=outbound_trajectories)
    timeout_filename = os.path.join(save_dir, timeout_name)
    _save_trajectories(file_name=timeout_filename, _trajectories=timeout_trajectories)


def test():
    ########################### Mode 0: Single Test ###########################
    print("============================================================================================")
    ########################### Environmenrt ###########################
    max_steps = 400
    max_ep_len = max_steps + 20
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10
    has_continuous_action_space = True
    true_random = True
    random_seed = 0
    flow_seed = 1

    switch_range = 12

    _is_head = False
    # switch_range = 0
    theta_mode = False
    random_range = 56
    _is_normalize = True
    test_mode = False
    state_dim_mode = 18

    # random point
    # _start = np.array([float(220), float(64)])
    # _target = np.array([float(64), float(64)])
    # # env = test_env_upper.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, _include_flow=True, _plot_flow=True, _proccess_flow=False, _random_range=random_range, _init_flow_num=random_seed, _pos_normalize=_is_normalize, _is_test=True, _state_dim=22)
    # env = train_env_upper_2.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, _include_flow=True, _plot_flow=True, _proccess_flow=False, _random_range=random_range, _init_flow_num=random_seed, 
    # _pos_normalize=_is_normalize, _is_test=False, _state_dim=18, _forward_only=True, _theta_mode=_theta)


    # fixed point
    _start = np.array([float(173), float(81)])
    _target = np.array([float(19.3), float(33.3)])
    ###################################################
    # _angle_in = np.random.uniform(0, np.pi)
    # lc_target_x, lc_target_y = compute_subgoal(pos=_start, angle=_angle_in, r=24)
    # _target = np.array([lc_target_x, lc_target_y])
    ###################################################
    # _start = np.array([float(248), float(49)])
    # _target = np.array([float(101.5), float(49)])
    # env = test_env_upper.foil_env(args_1, max_step=max_steps, start_position=_start, target_position=_target, _include_flow=True, _plot_flow=True, _proccess_flow=False, _random_range=56, _init_flow_num=random_seed, _pos_normalize=True, _is_test=True, _state_dim=22)
    env = train_env_upper_2.foil_env(args_1, max_step=max_steps, start_position=_start, target_position=_target, _include_flow=True, _plot_flow=True, _proccess_flow=False, _random_range=random_range, _init_flow_num=flow_seed, _pos_normalize=_is_normalize, _is_test = test_mode, _state_dim=state_dim_mode, 
    _is_random = true_random, _set_random=random_seed, u_range=20.0, v_range=20.0, _forward_only=_is_head, _obstacle_avoid=True, _theta_mode = theta_mode)


    # state space dimension
    state_dim = env.observation_space.shape[0]

    # high-level controller action space
    action_output_scale = np.array([])
    action_output_bias = np.array([])
    if has_continuous_action_space:
        # action space dimension
        action_dim = env.action_space.shape[0]
        action_output_scale = (env.action_space.high - env.action_space.low) / 2.0
        action_output_bias = (env.action_space.high + env.action_space.low) / 2.0
    else:
        action_dim = env.action_space.n
    

    # low-level controller action space
    lc_state_dim = 14
    lc_action_output_scale = np.array([])
    lc_action_output_bias = np.array([])
    if has_continuous_action_space:
        # low-level controller action space dimension
        lc_action_dim = env.lc_action_space.shape[0]
        lc_action_output_scale = (env.lc_action_space.high - env.lc_action_space.low) / 2.0
        lc_action_output_bias = (env.lc_action_space.high + env.lc_action_space.low) / 2.0
    else:
        lc_action_dim = env.lc_action_space.n
    ######################################################

    ########################### PPO ###########################
    # initialize parameter
    # update policy for K epochs in one PPO update
    K_epochs = 40
    # clip rate for PPO
    eps_clip = 0.2
    # discount factor γ  
    gamma = 0.99
    # learning rate for actor network  
    lr_actor = 0.0001  
    # learning rate for critic network
    lr_critic = 0.0002  
    # set std for action distribution when testing.
    action_std_4_test = 0.04  
    # initialize RL agent
    # ppo_agent = ICM_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
    #                     has_continuous_action_space,
    #                     action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
    #                     continuous_action_output_bias=action_output_bias, icm_alpha=200)
    # ppo_agent.load_full_icm(ppo_path)
    ppo_agent = Classic_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                            has_continuous_action_space,
                            action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
                            continuous_action_output_bias=action_output_bias)
    ppo_agent.load_full(ppo_path)
    ppo_agent_lc = Classic_PPO(
            lc_state_dim, lc_action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
            has_continuous_action_space,
            action_std_init=action_std_4_test,
            continuous_action_output_scale=lc_action_output_scale,
            continuous_action_output_bias=lc_action_output_bias
        )
    ppo_agent_lc.load_full(ppo_path_1)
    ######################################################

    ########################### Test ###########################
    # === 环境初始化 ===
    state, info = env.reset()
    state_14 = env.state_14
    agent_position = env.agent_pos
    prev_state = state.copy()  # 更新上一帧有效状态
    prev_state_14 = state_14.copy()  # 更新上一帧有效状态
    prev_agent_position = agent_position.copy()  # 更新上一帧有效状态
    print("position_reset:", agent_position)
    done = False
    t = 0
    hl_time_step = 0
    subgoal_threshold = 1

    while not done and t < max_ep_len:
        # === 高层决策：每 max_ll_steps 步更新一次 ===
        max_ll_steps = 5
        high_state = state
        high_action = ppo_agent.select_action(high_state)     # 高层输出目标角度
        high_action = np.clip(high_action, env.low, env.high) # 限制动作范围
        hl_time_step += 1
        print(f"===High Level Step {hl_time_step}, Action: {high_action}===")

        # === 将角度转化为子目标坐标 ===
        agent_position = env.agent_pos

        # lc_target_x, lc_target_y = compute_subgoal(pos=agent_position, angle=high_action[0], r=high_action[1])
        if _is_head:
            lc_target_x, lc_target_y = compute_subgoal_forward(pos=agent_position, angle=high_action, r=5)
        else:
            if not theta_mode:
                vec = high_action
                norm = np.linalg.norm(vec) + 1e-6
                vec = vec / norm
                high_action = np.arctan2(vec[1], vec[0])
            lc_target_x, lc_target_y = compute_subgoal(pos=agent_position, angle=high_action, r=5)
        # relative_dist_to_subgoal = np.linalg.norm(agent_position - np.array([lc_target_x, lc_target_y]))
        env.set_lc_target(np.array([lc_target_x, lc_target_y]))
        relative_dist_to_subgoal = np.linalg.norm(agent_position - np.array([lc_target_x, lc_target_y]))

        cumulative_high_reward = 0
        ll_step = 0
        done_lc = False

        # === 低层控制循环 ===
        while ll_step < max_ll_steps:
            agent_angle_old = env.angle

            # 构造低层状态：加上子目标相对位置
            low_state = state_14.copy()
            low_state[3] = lc_target_x - agent_position[0]
            low_state[4] = lc_target_y - agent_position[1]

            # === 执行低层动作 ===
            low_action = ppo_agent_lc.select_action(low_state)
            print(f"===Low Level Step {t}, Action: {low_action}===")
            state, reward, terminated, truncated, info = env.step(low_action)

            agent_position = env.agent_pos
            state_14 = env.state_14
            done = terminated or truncated

            dist_to_subgoal = np.linalg.norm(agent_position - np.array([lc_target_x, lc_target_y]))
            # === 判断低层是否完成子目标、失败或到达最大步数 ===
            if dist_to_subgoal < subgoal_threshold or done or ll_step + 1 >= max_ll_steps:
                done_lc = True
            t += 1
            ll_step += 1
            if done_lc:
                ppo_agent_lc.buffer.clear()
                break
        # 4 clear buffer
        ppo_agent.buffer.clear()
    env.close()
    print("Total_Steps:", t)
    ##############################################################################



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Choose a function to execute.")
    parser.add_argument("mode", type=int, choices=[0, 1, 2], help="Enter 0 to run test(), 1 to run success_rate_test(), 2 to run success_rate_test_1()")
    parser.add_argument("id", type=int, nargs="?", default=0, help="ID for success_rate_test() if mode is 1, Seed for success_rate_test_1() if mode is 2")
    args = parser.parse_args()

    if args.mode == 0:
        test()
    elif args.mode == 1:
        success_rate_test(args.id)
    elif args.mode == 2:
        success_rate_test_1(args.id)

