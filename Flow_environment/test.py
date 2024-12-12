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
# from env import flow_field_env_2
from env import flow_field_env_1

# ppo_path = './models/PPO_xonly.pth'
# ppo_path = './models/PPO_xy6.pth'
ppo_path = './models/PPO_xy14.pth'
# ppo_path = './models/PPO_xy14_2.pth' # 最新
# ppo_path = './models/PPO_xy14_4.pth'
# ppo_path = './models/PPO_xy14_3.pth' # 奖励最大
save_dir = './model_output'
action_name = "action_ppo.csv"


"""
def success_rate_test(env_in, agent_in, test_time, _max_ep_len, _agent_name, _if_plot=False):
    success_times = 0
    count_time = []
    positions_record = []
    all_positions = []
    success_flags = []
    for n in range(test_time):
        state, info = env_in.reset()  # gymnasium库改为在这里设置seed
        for i in range(1, _max_ep_len + 1):
            action = agent_in.select_action(state)
            state, reward, terminated, truncated, info = env_in.step(action)
            positions_record.append([info["x"], info["y"]])
            if terminated:
                print(f"Episode{n} success")
                success_times += 1
                success_flags.append(1)
                count_time.append(env_in.t_step * env_in.dt)
                break
            if truncated:
                print(f"Episode{n} fail")
                success_flags.append(0)
                break
        # clear buffer
        all_positions.append(positions_record)
        positions_record = []
        agent_in.buffer.clear()
        env_in.close()
    average_time = np.mean(count_time)
    if _if_plot:
        plot_success(all_positions, success_time=success_times, total_time=test_time, x_center=env_in.default_target[0],
                     y_center=env_in.default_target[1], success_flag=success_flags, method_name=_agent_name)
    return success_times, average_time
"""


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
    # env = flow_field_env_2.foil_env(args_1, max_step=max_time_step, target_position=target, include_flow=True)
    # env = flow_field_env_2.foil_env(args_1, max_step=max_time_step, include_flow=True) # 状态空间维度6
    env = flow_field_env_1.foil_env(args_1, max_step=max_time_step, include_flow=True) # 状态空间维度14
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

    state, info = env.reset()
    print("position_reset:", env.agent_pos)

    ep_return_ppo = 0
    total_steps_ppo = 0
    for i in range(1, max_ep_len + 1):
        action = ppo_agent.select_action(state)
        print(f"Step {i}, Action: {action}")
        action_ppo.append(action)
        state, reward, terminated, truncated, info = env.step(action)
        ep_return_ppo += reward
        total_steps_ppo = i
        position_ppo.append([info["pos_x"], info["pos_y"]])
        speed_ppo.append([info["vel_x"], info["vel_y"]])
        angle_ppo.append(info["angle"])
        if terminated:
            break
        if truncated:
            break
        # clear buffer
        ppo_agent.buffer.clear()
    env.close()

    print("Total_Reward:", ep_return_ppo)
    print("Total_Steps:", total_steps_ppo)
    # output_filename = save_dir + action_name
    output_filename = os.path.join(save_dir, action_name)
    # 打开文件进行写入
    with open(output_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["a_x", "a_y", "a_w"])
        # 写入每个动作的数据
        for action in action_ppo:
            # 假设 action 是一个包含三个元素的列表
            writer.writerow(action)


if __name__ == '__main__':
    test()
