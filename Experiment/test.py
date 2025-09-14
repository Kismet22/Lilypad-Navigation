import gym
import csv
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from termcolor import colored
import time
import math
import random

import my_ppo_net_1
from my_ppo_net_1 import Classic_PPO

import hjbppo
from hjbppo import HJB_PPO

import icm_ppo
from icm_ppo import ICM_PPO


from env_new import train_env
from env_new import train_env_server1
from env_new import train_env_1dim
from env_new import train_env_upper_2



# ppo_path = './models/easy_task/0515_5.pth'
# ppo_path = './models/easy_task/0612_10.pth'
# ppo_path = './models/easy_task/0629_2.pth'
# ppo_path = './models/easy_task/0630_11.pth'
# ppo_path = './models/easy_task/0704_seed1.pth'
# ppo_path = './models/easy_task/0705_lc.pth' # 双向
ppo_path = './models/exp/lc_dim14.pth'

save_dir = './model_output'
# Multi_Episodes Test
fail_name = "fail_ppo.csv"
plot_save_dir = './model_output/success_plot.png'
plot_save_dir_1 = './model_output/success_plot_withp_8.png'


def plot_success(all_position_records, success_flags, success_time, total_time, start_center, target_center, save_dir=None):
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
    # 计算 x 轴和 y 轴的范围，增加一些余量以确保图像能够完整显示轨迹
    x_margin = 2
    y_margin = 2
    ax.set_xlim(min_x - x_margin, max_x + x_margin)
    ax.set_ylim(min_y - y_margin, max_y + y_margin)

    for circle in circles:
        ax.add_patch(
            plt.Circle(circle["center"], circle["radius"], color='red', alpha=0.3)
        )

    # 绘制范围圆
    # start_circle = plt.Circle(start_center, 32, color='green', fill=False, linestyle='--')
    start_circle = plt.Circle(start_center, 56, color='green', fill=False, linestyle='--')
    ax.add_artist(start_circle)
    # target_circle = plt.Circle(target_center, 16, color='red', fill=False, linestyle='--')
    target_circle = plt.Circle(target_center, 56, color='red', fill=False, linestyle='--')
    ax.add_artist(target_circle)

    # 设置每条轨迹的颜色和线条属性
    line_alpha = 0.8  # 轨迹透明度
    line_width = 0.2  # 轨迹线宽

    # 定义一个字典来映射 success_flag 到对应的颜色
    color_map = {1: 'green', 2: 'blue', 0: 'red'}

    # 绘制所有轨迹
    for record, success_flag in zip(all_position_records, success_flags):  # 修改为 success_flags 而非 success_flag
        x_values = [pos[0] for pos in record]  # 获取 x 坐标
        y_values = [pos[1] for pos in record]  # 获取 y 坐标
        
        # 使用 color_map 来获取颜色，如果 success_flag 不是 1、2 或 0，默认为黑色
        line_color = color_map.get(success_flag, 'black')  # 默认为黑色，如果 success_flag 不是 0, 1 或 2
        # 绘制轨迹
        ax.plot(x_values, y_values, color=line_color, linestyle='-', linewidth=line_width, alpha=line_alpha)

    ax.set_xlabel('X/m')
    ax.set_ylabel('Y/m')
    ax.set_title(f'Success Rate:{success_time}/{total_time}')
    plt.grid(True)
    # 保存图像
    if save_dir:
        plt.savefig(save_dir, dpi=300)
    # plt.show()


def compute_subgoal(pos, angle, r):
    x, y = pos[0], pos[1]
    angle_1 = angle + np.pi/2
    dx = r * np.cos(angle_1)
    dy = r * np.sin(angle_1)
    return x + dx, y + dy

def success_rate_test():
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

    action_std_4_test = 0.001  # set std for action distribution when testing.
    random_seed = 10

    #####################################################
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10
    _start = np.array([float(220), float(32)])
    _target = np.array([float(220), float(96)])
    env = train_env_1dim.foil_env(args_1, max_step=max_steps, target_center=_target, start_center=_start, _include_flow=False, _plot_flow=False, _proccess_flow=False, _random_range=24, _init_ax=-15)
    # env = train_env_server1.foil_env(args_1, max_step=max_steps, target_center=_target, start_center=_start, _include_flow=False, _plot_flow=False, _proccess_flow=False, _random_range=24, _init_flow_num=0)

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
    ppo_agent = Classic_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space,
                        action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
                        continuous_action_output_bias=action_output_bias)
    ppo_agent.load_full(ppo_path)
    # ppo_agent = HJB_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
    #                     has_continuous_action_space, delta_t=env.dt, action_std_init=action_std_4_test,
    #                     continuous_action_output_scale=action_output_scale,
    #                     continuous_action_output_bias=action_output_bias)
    # ppo_agent.load_full(ppo_path)
    # ppo_agent = ICM_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
    #             has_continuous_action_space,
    #             action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
    #             continuous_action_output_bias=action_output_bias, icm_alpha=0.2)
    # ppo_agent.load_full_icm(ppo_path)

    #####################################################
    test_time = 200
    success_times = 0
    count_time = []
    # positions_record = []
    # all_positions = []
    # success_flags = []
    # failed_trajectories = []

    for n in range(test_time):
        # _is_collision = False
        state, info = env.reset()
        for i in range(1, max_ep_len + 1):
            action = ppo_agent.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            # positions_record.append([info["pos_x"], info["pos_y"]])
            if terminated:
                print(f"Episode{n} success")
                success_times += 1
                # if _is_collision:
                #     success_flags.append(2)
                # else:
                #     success_flags.append(1)
                count_time.append(env.step_counter * env.dt)
                break
            if truncated:
                print(f"Episode{n} fail")
                # success_flags.append(0)
                # failed_trajectories.append([env.start_position[0], env.start_position[1],env.target_position[0],env.target_position[1]])
                break

        # clear buffer
        # all_positions.append(positions_record)
        # positions_record = []
        ppo_agent.buffer.clear()

    env.close()
    average_time = np.mean(count_time)
    print(colored(f"**** Test: {test_time} ****", 'yellow'))
    print(colored(f"**** Success: {success_times} ****", 'green'))
    print(colored(f"**** Fail: {test_time - success_times} ****", 'red'))
    print(colored(f"****Average Navigation Time:{average_time} ****",'blue'))

    # output_filename = os.path.join(save_dir, fail_name)
    # with open(output_filename, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["start_x", "start_y", "target_x", "target_y"])
    #     for traj in failed_trajectories:
    #         writer.writerow(traj)
    # plot_success(all_position_records=all_positions, success_flags=success_flags, success_time=success_times, total_time=test_time, target_center=_target, start_center=_start, save_dir=plot_save_dir_1)



def test():
    print("============================================================================================")
    #####################################################
    # 1 RL init
    # continuous action space; else discrete
    has_continuous_action_space = True
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
    # set std for action distribution when testing
    action_std = 0.0001
    # Flow ID: 0 Random
    # random_seed = 0
    random_seed = 1
    #####################################################

    #####################################################
    # 2 Envirionment Reset
    # Max Steps
    max_steps = 3000
    max_ep_len = max_steps + 20
    # Port Setting
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10
    #####################################################
    # 参数
    a = 50   # 横向振幅（宽度的一半）
    b = 20   # 纵向振幅（高度的一半）
    n_points = 500  # 轨迹采样点数

    # 起点目标
    start_x = 180
    start_y = 72

    # 找一个起点相位（这里我们直接设 t=0 对应起点）
    t = np.linspace(0, 2*np.pi, n_points)

    # 原始曲线（相对于 0,0）
    x = a * np.sin(t)
    y = b * np.sin(2*t)

    # 平移到 (180, 72)
    x += start_x
    y += start_y

    points = np.vstack((x, y)).T
    arc_points = points

    # # ================================
    # # 参数设置
    # # ================================
    # center_top = np.array([180.0, 72.0])     # 上圆圆心
    # center_bottom = np.array([180.0, 36.0])  # 下圆圆心
    # r = 16                                   # 半径
    # deg_step = 5                             # 角度步长（度）

    # # ================================
    # # 1. 上半圆（从左侧 180° 到右侧 0°，逆时针）
    # # ================================
    # angles_deg_top_half = np.arange(180, 360 + deg_step, deg_step)
    # angles_rad_top_half = np.deg2rad(angles_deg_top_half)
    # arc_top_half = np.array([
    #     center_top + r * np.array([np.cos(theta), np.sin(theta)])
    #     for theta in angles_rad_top_half
    # ])

    # # ================================
    # # 2. 下半圆（从右侧 0° 到左侧 180°，逆时针）
    # # ================================
    # angles_deg_bottom_half = np.arange(0, 180 + deg_step, deg_step)
    # angles_rad_bottom_half = np.deg2rad(angles_deg_bottom_half)
    # arc_bottom_half = np.array([
    #     center_bottom + r * np.array([np.cos(theta), np.sin(theta)])
    #     for theta in angles_rad_bottom_half
    # ])

    # # ================================
    # # 3. 下半圆另一半（180° 到 360°）
    # # ================================
    # angles_deg_bottom_half2 = np.arange(180, 360 + deg_step, deg_step)
    # angles_rad_bottom_half2 = np.deg2rad(angles_deg_bottom_half2)
    # arc_bottom_half2 = np.array([
    #     center_bottom + r * np.array([np.cos(theta), np.sin(theta)])
    #     for theta in angles_rad_bottom_half2
    # ])

    # # ================================
    # # 4. 上半圆另一半（0° 到 180°）
    # # ================================
    # angles_deg_top_half2 = np.arange(0, 180 + deg_step, deg_step)
    # angles_rad_top_half2 = np.deg2rad(angles_deg_top_half2)
    # arc_top_half2 = np.array([
    #     center_top + r * np.array([np.cos(theta), np.sin(theta)])
    #     for theta in angles_rad_top_half2
    # ])

    # # ================================
    # # 合并轨迹
    # # ================================
    # arc_points = np.vstack([
    #     arc_top_half, arc_bottom_half, arc_bottom_half2, arc_top_half2
    # ])


    # last_point = np.array([float(150), float(64)])
    # step_size = 5.0  # 每步平移 4 个单位
    # total_distance = 20
    # num_steps = int(total_distance / step_size)  # 5 步

    # # 生成连续右移轨迹
    # extended_points = np.array([
    #     last_point - np.array([i * step_size, 0.0])
    #     for i in range(1, num_steps + 1)
    # ])

    # # 假设 arc_points 是前一段的圆弧
    # _start_2 = extended_points[-1]  # 上一段的终点作为新的起点
    # _center_2 = np.array([109.0, 64.0])

    # # 半径
    # r2 = np.linalg.norm(_start_2 - _center_2)

    # deg_step = 4.5
    # rad_step = np.deg2rad(deg_step)

    # # 起始角度（以圆心为参考）
    # vec2 = _start_2 - _center_2
    # theta_start_2 = np.arctan2(vec2[1], vec2[0])

    # # 顺时针方向，角度减少 π 弧度
    # theta_end_2 = theta_start_2 - np.pi

    # # 角度序列（顺时针，用负方向步长）
    # angles_2 = np.arange(theta_start_2, theta_end_2 - rad_step, -rad_step)

    # # 生成第二段圆弧上的点
    # arc_points_2 = np.array([
    #     _center_2 + r2 * np.array([np.cos(theta), np.sin(theta)])
    #     for theta in angles_2
    # ])

    # # 拼接两段轨迹
    # arc_points = np.vstack((extended_points, arc_points_2))

    
    # ##################################################################
    # _center = np.array([float(220), float(64)])
    # _start = np.array([float(240), float(64)])
    # # 半径
    # r = np.linalg.norm(_start - _center)

    # # 起始角度（以圆心为基准，arctan2(y, x)）
    # vec = _start - _center
    # theta_start = np.arctan2(vec[1], vec[0])  # 弧度制起始角

    # # 构造角度序列：从 theta_start 开始，步长 4.5°（转为弧度），转 90° = π/2
    # deg_step = 4.5
    # rad_step = np.deg2rad(deg_step)
    # theta_end = theta_start + np.pi

    # angles = np.arange(theta_start, theta_end + rad_step, rad_step)

    # # 生成圆弧上的点
    # arc_points = np.array([
    #    _center + r * np.array([np.cos(theta), np.sin(theta)])
    #    for theta in angles
    # ])
    # # 假设 arc_points 是前一段的圆弧
    # _start_2 = arc_points[-1]  # 上一段的终点作为新的起点
    # _center_2 = np.array([180.0, 64.0])

    # # 半径
    # r2 = np.linalg.norm(_start_2 - _center_2)

    # # 起始角度（以圆心为参考）
    # vec2 = _start_2 - _center_2
    # theta_start_2 = np.arctan2(vec2[1], vec2[0])

    # # 顺时针方向，角度减少 π 弧度
    # theta_end_2 = theta_start_2 - np.pi

    # # 角度序列（顺时针，用负方向步长）
    # angles_2 = np.arange(theta_start_2, theta_end_2 - rad_step, -rad_step)

    # # 生成第二段圆弧上的点
    # arc_points_2 = np.array([
    #     _center_2 + r2 * np.array([np.cos(theta), np.sin(theta)])
    #     for theta in angles_2
    # ])

    # # 拼接两段轨迹
    # arc_points = np.vstack((arc_points, arc_points_2))
    # ###################################################################################

    # ###################################################################################
    # # ================================
    # # 生成圆形轨迹
    # # ================================
    # _center = np.array([180.0, 64.0])  # 圆心
    # r = 20                             # 半径（像素）
    # deg_step = 2.5                      # 角度步长（度）

    # # 左半圆角度范围：从 90° 到 270°
    # angles_deg = np.arange(90, 270 + deg_step, deg_step)
    # angles_rad = np.deg2rad(angles_deg)

    # circle_points_left = np.array([
    #     _center + r * np.array([np.cos(theta), np.sin(theta)])
    #     for theta in angles_rad
    # ])

    # last_point = circle_points_left[-1]
    # step_size = 4.0  # 每步平移 4 个单位
    # total_distance = 20.0
    # num_steps = int(total_distance / step_size)  # 5 步

    # # 生成连续右移轨迹
    # extended_points = np.array([
    #     last_point + np.array([i * step_size, 0.0])
    #     for i in range(1, num_steps + 1)
    # ])


    # last_point = extended_points[-1]  # shape: (2,)
    # # 参数
    # r = 10
    # deg_step = 5
    # angles_deg = np.arange(-90, 90 + deg_step, deg_step)  # -90 to 90 degrees, inclusive
    # angles_rad = np.deg2rad(angles_deg)

    # # 圆心在左侧，画右半圆
    # _center = last_point + np.array([0.0, r])

    # # 生成右半圆上的点
    # circle_points_right = np.array([
    #     _center + r * np.array([np.cos(theta), np.sin(theta)])
    #     for theta in angles_rad
    # ])


    # last_point = circle_points_right[-1]  # shape: (2,)
    # # 参数
    # r = 10
    # deg_step = 5
    # angles_deg = np.arange(270, 90- deg_step, -deg_step)
    # angles_rad = np.deg2rad(angles_deg)

    # _center = last_point + np.array([0.0, r])

    # circle_points_left_1 = np.array([
    #     _center + r * np.array([np.cos(theta), np.sin(theta)])
    #     for theta in angles_rad
    # ])

    # last_point = circle_points_left_1[-1]
    # step_size = 4.0  # 每步平移 4 个单位
    # total_distance = 40.0
    # num_steps = int(total_distance / step_size)  # 5 步

    # # 生成连续右移轨迹
    # extended_points_1 = np.array([
    #     last_point + np.array([i * step_size, 0.0])
    #     for i in range(1, num_steps + 1)
    # ])

    # last_point = extended_points_1[-1]  # shape: (2,)
    # # 参数
    # r = 20
    # deg_step = 2.5
    # angles_deg = np.arange(90, 270 + deg_step, deg_step)  # -90 to 90 degrees, inclusive
    # angles_rad = np.deg2rad(angles_deg)

    # _center = last_point + np.array([0.0, -r])

    # circle_points_left_2 = np.array([
    #     _center + r * np.array([np.cos(theta), np.sin(theta)])
    #     for theta in angles_rad
    # ])
    
    # last_point = circle_points_left_2[-1]
    # step_size = 4.0  # 每步平移 4 个单位
    # total_distance = 20.0
    # num_steps = int(total_distance / step_size)  # 5 步

    # # 生成连续右移轨迹
    # extended_points_2 = np.array([
    #     last_point + np.array([i * step_size, 0.0])
    #     for i in range(1, num_steps + 1)
    # ])

    # arc_points  = np.vstack([circle_points_left, extended_points, circle_points_right, circle_points_left_1,extended_points_1, circle_points_left_2, extended_points_2])
    ####################################################################################

    # 起点与目标点
    _start = arc_points[0]
    i_count = 1
    _target = arc_points[i_count]
    ###################################################################
    

    env = train_env_server1.foil_env(args_1, max_step=max_steps, target_position=_target, start_position=_start, _include_flow=True, _plot_flow=True, _proccess_flow=False, _random_range=24, 
    _init_flow_num = random_seed, u_range=15.0, v_range=15.0, w_range=5.0, custom_trajectory=arc_points)
    #####################################################
    
    #####################################################
    # 4 Space Info
    # state space dimension
    state_dim = env.observation_space.shape[0]
    action_output_scale = np.array([])
    action_output_bias = np.array([])
    if has_continuous_action_space:
        # action space dimension
        action_dim = env.action_space.shape[0]
        # y = scale * x + bias
        action_output_scale = (env.action_space.high - env.action_space.low) / 2.0
        action_output_bias = (env.action_space.high + env.action_space.low) / 2.0
    else:
        action_dim = env.action_space.n
    #####################################################

    #####################################################
    # 5 Initialize RL Agent
    # ppo_agent = HJB_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
    #                     has_continuous_action_space, delta_t=env.dt, action_std_init=action_std,
    #                     continuous_action_output_scale=action_output_scale,
    #                     continuous_action_output_bias=action_output_bias)
    # ppo_agent.load_full(ppo_path)
    
    ppo_agent = Classic_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space,
                        action_std, continuous_action_output_scale=action_output_scale,
                        continuous_action_output_bias=action_output_bias)
    ppo_agent.load_full(ppo_path)

    # ppo_agent = ICM_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
    #                 has_continuous_action_space,
    #                 action_std, continuous_action_output_scale=action_output_scale,
    #                 continuous_action_output_bias=action_output_bias, icm_alpha=0.2)
    # ppo_agent.load_full_icm(ppo_path)
    #####################################################

  
    ########################### Single Test ###########################
    action_name = "action.csv"
    pressure_name = "pressure_ppo.csv"
    velocity_name = "v.csv"

    # position_ppo = []
    # angle_ppo = []
    action_ppo = []
    speed_ppo = []
    pressure_ppo = []

    state, info = env.reset()
    print("Position Reset At:", env.agent_pos)

    ep_return_ppo = 0
    total_steps_ppo = 0
    for i in range(1, max_ep_len + 1):
        action = ppo_agent.select_action(state)
        print(f"Step {i}, Action: {action}")
        action_ppo.append(action)
        state, reward, terminated, truncated, info = env.step(action)
        print(f"Angle:{np.degrees(env.angle)}")
        ep_return_ppo += reward
        total_steps_ppo = i
        # position_ppo.append([info["pos_x"], info["pos_y"]])
        # angle_ppo.append(info["angle"])
        speed = math.sqrt(info["vel_x"]**2 + info["vel_y"]**2)
        speed_ppo.append([info["vel_x"], info["vel_y"], info["vel_angle"], speed])
        pressure_ppo.append([info["pressure_1"], info["pressure_2"], info["pressure_3"], info["pressure_4"], info["pressure_5"], info["pressure_6"], info["pressure_7"], info["pressure_8"]])
        if terminated:
            i_count += 1
            _target = arc_points[i_count]
            
            # _angle_in = np.random.uniform(0, np.pi)
            # lc_target_x, lc_target_y = compute_subgoal(pos=env.agent_pos, angle=_angle_in, r=5)
            # _target = np.array([lc_target_x, lc_target_y])
            env.set_target(pos_in = _target)
            # break
        if truncated:
            break
        # clear buffer
        ppo_agent.buffer.clear()
    env.close()

    # Print Info
    print("Total_Reward:", ep_return_ppo)
    print("Total_Steps:", total_steps_ppo)
    # Savig State
    output_filename = os.path.join(save_dir, action_name)
    output_filename_2 = os.path.join(save_dir, pressure_name)
    output_filename_3 = os.path.join(save_dir, velocity_name)
    with open(output_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["a_x", "a_y", "a_w"])
        for action in action_ppo:
            writer.writerow(action)
    with open(output_filename_2, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["p_1", "p_2", "p_3", "p_4", "p_5", "p_6", "p_7", "p_8"])
        for pressure in pressure_ppo:
            writer.writerow(pressure)
    with open(output_filename_3, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["v_x", "v_y", "v_w", "v"])
        for speed in speed_ppo:
            writer.writerow(speed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Choose a function to execute.")
    parser.add_argument("mode", type=int, choices=[0, 1], help="Enter 0 to run test(), 1 to run success_rate_test()")
    args = parser.parse_args()
    if args.mode == 0:
        test()
    elif args.mode == 1:
        success_rate_test()
