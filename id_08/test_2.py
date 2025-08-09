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
from env_new import train_env_combine
from env_new import train_env_path_follow




# normal
ppo_path_far = './models/normal_task/basic2_0611_1.pth' # icm
# easy
ppo_path_close = './models/easy_task/0515_5.pth'
# roll
ppo_path_roll = './models/easy_task/0705_lc.pth' # all direction


save_dir = './SuccessTest'
save_dir_1 = './model_output'
# plot_save_dir = './model_output/success_plot_withp.png'


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


def compute_subgoal(pos, angle, r):
    x, y = pos[0], pos[1]
    angle_1 = angle + np.pi/2
    dx = r * np.cos(angle_1)
    dy = r * np.sin(angle_1)
    return x + dx, y + dy


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
        start = np.array([float(240), float(64)])
        # start = np.array([float(240), float(32)])
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
    success_name = f"{ppo_name}_success_{_id}_64.csv"
    collide_name = f"{ppo_name}_collide_{_id}_64.csv"
    outbound_name = f"{ppo_name}_outbound_{_id}_64.csv"
    timeout_name = f"{ppo_name}_timeout_{_id}_64.csv"
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
    # Flow ID : 0 Random Flow
    # random_seed = 0
    random_seed = 1

    switch_range = 12
    # switch_range = 0

    # random point
    # _start = np.array([float(220), float(64)])
    # _target = np.array([float(64), float(64)])
    # env = train_env_combine.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, _include_flow=True, _plot_flow=True, _proccess_flow=False, _swich_range=switch_range, _random_range=56, _init_flow_num=random_seed, _pos_normalize=True, _is_test=True)
    
    # fixed point
    # _start = np.array([float(240), float(64)])
    # _target = np.array([float(64), float(64)])
    _start = np.array([float(173), float(81)])
    _target = np.array([float(19.3), float(33.3)])
    ###################################################
    # _angle_in = np.random.uniform(0, np.pi)
    # lc_target_x, lc_target_y = compute_subgoal(pos=_start, angle=_angle_in, r=24)
    # _target = np.array([lc_target_x, lc_target_y])
    ###################################################
    env = train_env_combine.foil_env(args_1, max_step=max_steps, start_position=_start, target_position=_target, _include_flow=True, _plot_flow=True, _proccess_flow=False, _swich_range=switch_range, _random_range=56, _init_flow_num=random_seed, _pos_normalize=True, _is_test=True)
    ######################################################

    ########################### State&Action ###########################
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
    ppo_agent = ICM_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space,
                        action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
                        continuous_action_output_bias=action_output_bias, icm_alpha=200)
    ppo_agent.load_full_icm(ppo_path_far)

    ppo_agent_near = Classic_PPO(14, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space,
                        action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
                        continuous_action_output_bias=action_output_bias)
    ppo_agent_near.load_full(ppo_path_close)

    ppo_agent_roll = Classic_PPO(14, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                            has_continuous_action_space,
                            action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
                            continuous_action_output_bias=action_output_bias)
    ppo_agent_roll.load_full(ppo_path_roll)
    ######################################################

    ########################### Test ###########################
    speed_ppo = []
    action_ppo = []
    pressure_ppo = []
    # 1 env reset
    strategy_switch = False
    roll_flag = False
    state, info = env.reset()
    state_14 = env.state_14
    print("position_reset:", env.agent_pos)

    ep_return_ppo = 0
    total_steps_ppo = 0
    for i in range(1, max_ep_len + 1):

        # 1. 初始化 roll 弧线轨迹
        if env.safe_range_flag and not roll_flag:
            print(colored("**** ROLL SETTING ****", 'magenta'))
            roll_flag = True
            i_count = 0
            _start = env.agent_pos
            _center = env.cloest_obstacle

            vec = _start - _center - 4
            r = np.linalg.norm(vec)
            theta_start = np.arctan2(vec[1], vec[0])

            delta_theta = state[17] * np.pi * 0.9
            direction = state[16]  # +1: ccw, -1: cw

            # Non-uniform angle change: slower at start, faster later
            rad_step = np.deg2rad(15)
            num_steps = int(np.abs(delta_theta) / rad_step)
            ts = np.linspace(0, 1, num_steps)
            ts = ts**2  # front slow, back fast
            angles = theta_start + direction * delta_theta * ts

            # arc_points = [_start.copy()] * 3  # stay still for 3 steps
            arc_points = []
            arc_points += [
                _center + r * np.array([np.cos(theta), np.sin(theta)])
                for theta in angles
            ]
            arc_points = np.array(arc_points)
            i_len = len(arc_points)
        # if env.safe_range_flag and not roll_flag:
        #     print(colored("**** ROLL SETTING ****", 'magenta'))
        #     roll_flag = True
        #     i_count = 0
        #     _start = env.agent_pos
        #     # Compute arc center and radius
        #     _center = env.cloest_obstacle
        #     vec = _start - _center - 4
        #     r = np.linalg.norm(vec)

        #     # Recompute theta_start
        #     theta_start = np.arctan2(vec[1], vec[0])
        #     delta_theta = state[17] * np.pi * 0.9
        #     direction = state[16]  # +1: counter-clockwise, -1: clockwise

        #     # Arc sampling
        #     rad_step = np.deg2rad(15)
        #     num_steps = int(np.abs(delta_theta) / rad_step)
        #     angles = theta_start + direction * rad_step * np.arange(1, num_steps + 1)  # start from step 1

        #     # Start point
        #     arc_points = [_start.copy()] * 3  # ensure first point is agent's current position
        #     # arc_points = []
        #     arc_points += [
        #         _center + r * np.array([np.cos(theta), np.sin(theta)])
        #         for theta in angles
        #     ]
        #     arc_points = np.array(arc_points)
        #     i_len = len(arc_points)

        # 2. 动作选择和环境步进

        if strategy_switch:
            env.reset_target()
            print(colored("**** Mode Near ****", 'cyan'))
            action = ppo_agent_near.select_action(state_14)
            state, reward, terminated, truncated, info = env.step(action)
            state_14 = env.state_14
            if truncated or terminated:
                break
        elif roll_flag and i_count < i_len:
            print(colored("**** Mode ROLL ****", 'magenta'))
            _target = arc_points[i_count]
            env.set_target(pos_in=_target)
            action = ppo_agent_roll.select_action(state_14)
            action_ppo.append(action)
            state, reward, terminated_roll, truncated, info = env.step(action)
            state_14 = env.state_14
            if terminated_roll:
                i_count += 1
            if i_count >= i_len or truncated:
                roll_flag = False
                print(colored("**** Mode ROLL END ****", 'magenta'))
            if truncated:
                break
        else:
            env.reset_target()
            print(colored("**** Mode Far ****", 'yellow'))
            action = ppo_agent.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            state_14 = env.state_14
            if truncated or terminated:
                break

        # 3. 策略切换
        if env.d_2_end < switch_range and not strategy_switch:
            strategy_switch = True
            print(colored("**** Strategy Switch ****", 'cyan'))

        # 4. 信息记录
        print(f"Step {i}, Action: {action}")
        ep_return_ppo += reward
        total_steps_ppo = i
        pressure_ppo.append([info[f"pressure_{j}"] for j in range(1, 9)])
        speed = math.sqrt(info["vel_x"]**2 + info["vel_y"]**2)
        speed_ppo.append([info["vel_x"], info["vel_y"], info["vel_angle"], speed])

        # 5. 清除 buffer
        ppo_agent.buffer.clear()
    env.close()
    print("Total_Reward:", ep_return_ppo)
    print("Total_Steps:", total_steps_ppo)

    # 5 state save
    action_name = "action_ppo.csv"
    pressure_name = "pressure_ppo.csv"
    velocity_name = "v_ppo.csv"
    action_filename = os.path.join(save_dir_1, action_name)
    pressure_filename = os.path.join(save_dir_1, pressure_name)
    velocity_filename = os.path.join(save_dir_1, velocity_name)
    with open(action_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["a_x", "a_y", "a_w"])
        for action in action_ppo:
            writer.writerow(action)
    with open(pressure_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["p_1", "p_2", "p_3", "p_4", "p_5", "p_6", "p_7", "p_8"])
        for pressure in pressure_ppo:
            writer.writerow(pressure)
    with open(velocity_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["v_x", "v_y", "v_w", "v"])
        for speed in speed_ppo:
            writer.writerow(speed)
    ######################################################



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
