import h5py
import gym
import os
import argparse
import torch
import numpy as np
from datetime import datetime
from termcolor import colored
import time

####### PPO_METHOD #######
import icm_ppo
from icm_ppo import ICM_PPO
import my_ppo_net_1
from my_ppo_net_1 import Classic_PPO
import hjbppo
from hjbppo import HJB_PPO

####### ENVIRONMENT #######
from env import flow_field_env_2
# Normal task Basic
from env_new import train_env_basic
from env_new import train_env_basic_1
from env_new import train_env_basic_2
from env_new import train_env_basic_3

####### DEVICE #######
# evice = my_ppo_net_1.device
# device = hjbppo.device
device = icm_ppo.device

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


def train(model_id, render_mode=None, checkpoint_path=None):
    print("============================================================================================")

    ####### 训练环境变量设置 #######
    # 环境名称，用于保存
    state_dim_mode = 18
    env_name = f'Flow_env_Obstacle_july_dim_{state_dim_mode}'

    # 判断动作空间是否连续
    # continuous action space; else discrete
    has_continuous_action_space = True

    # 一个episode中的最大时间步数,存在环境时间限制时,最大时间步应该大于环境时间限制
    # max time_steps in one episode
    max_steps = 300
    max_ep_len = max_steps + 20

    # 结束训练的总训练步数
    # break training loop if timesteps > max_training_timesteps
    max_training_timesteps = int(8e5)

    # 打印/保存episode奖励均值
    # Note : print/log frequencies should be more than max_ep_len
    # print avg reward in the interval (in num timesteps)
    print_freq = max_ep_len * 5
    # log avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 5
    # 存储模型间隔
    save_model_freq = int(1e4)
    # 初始方差
    action_std = 0.5  # starting std for action distribution (Multivariate Normal)
    # 方差更新缩减值
    action_std_decay_rate = 0.02  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    # 最小方差
    min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
    # 方差缩减频率
    action_std_decay_freq = int(5e4)  # action_std decay frequency (in num timesteps)
    #####################################################

    ################ 强化学习超参数设置 ################
    # 网络更新频率
    # update policy every n timesteps
    update_timestep = max_ep_len * 10
    # K_epochs = 10
    K_epochs = 40

    eps_clip = 0.2  # clip rate for PPO
    gamma = 0.99  # discount factor γ

    lr_actor = 0.0001  # learning rate for actor network
    lr_critic = 0.0002  # learning rate for critic network

    flow_seed = 0  # set flow seed if required (0 = no random seed)
    # flow_seed = 1

    true_random = False
    if true_random:
        random_seed = 0
    else:
        random_seed = 100

    pretrained_model_ID = model_id  # 训练ID, change this to prevent overwriting weights in same env_name folder
    #####################################################

    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10

    # _start = np.array([float(220), float(64)])
    # _target = np.array([float(64), float(64)])
    # env = flow_field_env_2.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target)

    _start = np.array([float(220), float(64)])
    _target = np.array([float(64), float(64)])
    env = train_env_basic_2.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, _random_range=56, _init_flow_num=flow_seed, _pos_normalize=True, _state_dim=state_dim_mode, 
    _is_random = true_random, _set_random=random_seed, u_range=15.0, v_range=15.0)
    # env = train_env_basic_3.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, _random_range=56, _init_flow_num=flow_seed, _pos_normalize=True, _state_dim=state_dim_mode, 
    # _is_random = true_random, _set_random=random_seed, u_range=15.0, v_range=15.0)

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

    ###################### 保存训练信息 ######################
    # log files for multiple runs are NOT overwritten

    ###################### 强化学习方法文件夹 ######################
    log_dir = "PPO_logs"
    ######################

    # 创建日志文件夹（如果不存在）
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 训练环境文件夹
    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建新的日志文件或使用已存在的日志文件
    ###################### 训练过程记录文件 ######################
    rl_method = '/PPO'
    ######################
    log_f_name = log_dir + rl_method + env_name + "_log_" + str(pretrained_model_ID) + "_seed_" + str(random_seed) + ".csv"

    print("current logging model ID for " + env_name + " : ", pretrained_model_ID)
    print("logging at : " + log_f_name)

    # 检查日志文件是否已经存在，如果存在，直接继续使用
    if os.path.exists(log_f_name):
        print(f"Warning: Log file for model ID {pretrained_model_ID} already exists. Appending new data.")
        # 读取文件最后一行并打印
        try:
            with open(log_f_name, "r") as log_f:
                lines = log_f.readlines()  # 读取所有行
                if lines:
                    last_line = lines[-1]  # 获取最后一行
                    print("文件的最后一行内容是:")
                    print(last_line)
                else:
                    print("日志文件为空，无法读取最后一行。")
        except Exception as e:
            print(f"读取文件失败: {e}")
    else:
        # 如果文件不存在，创建新文件并写入表头
        try:
            with open(log_f_name, "w+") as log_f:
                log_f.write('episode,timestep,avg_return_in_period,action_std,ratio\n')
                print("表头写入成功")
        except Exception as e:
            print(f"写入文件失败: {e}")
    log_f.close()
    ###################### 保存模型信息 ######################

    ###################### 文件夹 ######################
    directory = "PPO_preTrained"
    ######################

    # 如果文件夹已经存在，则不再创建
    if not os.path.exists(directory):
        os.makedirs(directory)

    ###################### 文件名 ######################
    method_name = "PPO_{}_ID_{}_seed_{}"
    ######################

    # 根据训练环境、模型ID和种子号生成文件夹路径
    directory = directory + '/' + env_name + '/' + method_name.format(env_name, pretrained_model_ID, random_seed) + '/'

    # 检查文件夹是否存在，不存在时创建
    if not os.path.exists(directory):
        os.makedirs(directory)

    print("save checkpoint directory : " + directory)
    ##################################################### 

    ############# 超参数信息打印 #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("RL_METHOD update frequency : " + str(update_timestep) + " timesteps")
    print("RL_METHOD K epochs : ", K_epochs)
    print("RL_METHOD epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        # torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    print("============================================================================================")
    #####################################################

    ################# training procedure ################
    # initialize RL agent
    # ppo_agent = HJB_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
    #                     has_continuous_action_space, delta_t=env.dt, action_std_init=action_std,
    #                     continuous_action_output_scale=action_output_scale,
    #                     continuous_action_output_bias=action_output_bias)
    # ppo_agent = Classic_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
    #                         has_continuous_action_space,
    #                         action_std, continuous_action_output_scale=action_output_scale,
    #                         continuous_action_output_bias=action_output_bias)
    ppo_agent = ICM_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                        has_continuous_action_space,
                        action_std, continuous_action_output_scale=action_output_scale,
                        continuous_action_output_bias=action_output_bias, icm_alpha=200)

    # 加载 checkpoint 如果提供了路径
    if checkpoint_path:
        print(f"Loading checkpoint from: {checkpoint_path}")
        ppo_agent.load_full(checkpoint_path)
        # ppo_agent.load_full_icm(checkpoint_path)
        print("Model loaded successfully.")
        print("============================================================================================")
        # 当前时间步
        time_step = 0
        # 当前回合数
        i_episode = 0
        # 当前的最大平均回报
        max_model_avg_return = -np.inf
        # 从日志文件中读取最后一行，获取 time_step, i_episode 和 max_model_avg_return
        if os.path.exists(log_f_name):
            with open(log_f_name, "r") as log_f:
                # 读取所有行
                lines = log_f.readlines()
                if lines:
                    # 从最后一行中提取 time_step, i_episode 和 max_model_avg_return
                    try:
                        # 假设日志格式为: episode,timestep,avg_return_in_period,action_std
                        last_line_values = last_line.strip().split(',')
                        i_episode = int(last_line_values[0])  # 提取回合数
                        time_step = int(last_line_values[1])  # 提取时间步
                        max_model_avg_return = float(last_line_values[2])  # 提取最大平均回报
                        _init_std = float(last_line_values[3])
                        action_std = _init_std
                        log_f.close()
                        print(f"Loaded values from the last log entry: i_episode = {i_episode}, time_step = {time_step}, max_model_avg_return = {max_model_avg_return}")
                    except ValueError as e:
                        print(f"解析最后一行数据失败: {e}")
                else:
                    print("日志文件为空，无法读取最后一行。")
        else:
            print("日志文件不存在，无法从中加载数据。")

    else:
        print("Model created successfully.")
        # 当前时间步
        time_step = 0
        # 当前回合数
        i_episode = 0
        # 当前的最大平均回报
        max_model_avg_return = -np.inf


    # printing and logging variables
    print_running_return = 0
    print_running_episodes = 0
    log_running_return = 0
    log_running_episodes = 0
    cur_model_running_return = 0
    cur_model_running_episodes = 0
    
    # write training data
    log_f = open(log_f_name, "a")
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")
    icm_ratio = 1
    batch_size = 1
    save_path = f"./train_output/flow_data_seed_{random_seed}.h5"
    batch_data = {
        "episode_id": [], "state": [], "action": [], "pressure": [], "position": [], "speed": []
    }
    # training loop
    while time_step <= max_training_timesteps:
        print(f"episode:{i_episode}")
        collect_terminate = False
        state, info = env.reset()
        state_old = state.copy()
        info_old = info.copy()
        print("position_reset:", env.agent_pos)
        episode_data = {
            "episode_id": i_episode,
            "state": [], "action": [], "pressure": [], "position": [], "speed": []
        }
        current_ep_return = 0
        # 在最大更新步长以内
        for t in range(1, max_ep_len + 1):
            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            if not collect_terminate:
                current_data = {
                        "state": state_old,
                        "action": action,
                        "pressure": [
                            info_old["pressure_1"], info_old["pressure_2"], info_old["pressure_3"],
                            info_old["pressure_4"], info_old["pressure_5"], info_old["pressure_6"],
                            info_old["pressure_7"], info_old["pressure_8"]
                        ],
                        "position": [info_old["pos_x"], info_old["pos_y"], info_old["angle"]],
                        "speed": [info_old["vel_x"], info_old["vel_y"], info_old["vel_angle"]],
                    }
                
                has_nan_inf = any( contains_nan_inf(np.array(current_data[key])) for key in current_data)

                if has_nan_inf:
                    print(colored(f"[Warning] NaN or Inf detected at episode {n}, step {t}. Aborting episode safely.",'red'))
                    collect_terminate = True
                else:
                    episode_data["state"].append(state_old)
                    episode_data["action"].append(action)
                    episode_data["pressure"].append(current_data["pressure"])
                    episode_data["position"].append(current_data["position"])
                    episode_data["speed"].append(current_data["speed"])            
            state_old = state.copy()
            info_old = info.copy()
            done = terminated or truncated

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            # saving next_state
            state_next = torch.tensor(state, device=device, dtype=torch.float32)
            ppo_agent.buffer.next_states.append(state_next)

            # time_step不断增加，不会减少
            time_step += 1
            current_ep_return += reward

            # update RL agent
            # 模型检查更新
            if time_step % update_timestep == 0:
                # 计算上一个model的平均return per episode
                cur_model_avg_return = cur_model_running_return / cur_model_running_episodes
                if cur_model_avg_return > max_model_avg_return:
                    # 保存更好的模型
                    print("Better Model.")
                    checkpoint_path = directory + f"best_model_time_step{time_step}_std{ppo_agent.action_std}.pth"
                    print("saving model at : " + checkpoint_path)

                    ############################################################
                    # ppo_agent.save_full(checkpoint_path)
                    ppo_agent.save_full_icm(checkpoint_path)
                    ############################################################

                    max_model_avg_return = cur_model_avg_return
                cur_model_running_return = 0
                cur_model_running_episodes = 0

                print("Policy Updated. At Episode : {}  Timestep : {}".format(i_episode, time_step))
                ppo_agent.update()
                
                ############################################################
                icm_ratio = ppo_agent.icm_average_ratio
                ############################################################

            # if continuous action space; then decay action std of output action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                # 每action_std_decay_freq步，降低一次网络的初始化方差
                action_std = ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file， 多个episode进行一次记录，记录的是这几个episode的avg return
            if time_step % log_freq == 0:
                # log average return till last episode
                log_avg_return = log_running_return / log_running_episodes
                log_avg_return = round(log_avg_return, 4)
                log_f.write('{},{},{},{},{}\n'.format(i_episode, time_step, log_avg_return, action_std, icm_ratio))
                log_f.flush()
                log_running_return = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_return = print_running_return / print_running_episodes
                print_avg_return = round(print_avg_return, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Return : {}".format(i_episode, time_step,
                                                                                        print_avg_return))

                print_running_return = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                checkpoint_path = directory + f"timestep_{time_step}_std_{ppo_agent.action_std:.2f}.pth"
                print("saving model at : " + checkpoint_path)

                ############################################################
                # ppo_agent.save_full(checkpoint_path)
                ppo_agent.save_full_icm(checkpoint_path)
                ############################################################

                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                print(colored(f"**** Episode Reward:{current_ep_return} **** ", "yellow"))
                break
            if t == max_ep_len:
                print(colored("**** Episode Terminated **** Reaches Training Episode Time limit.", 'blue'))

        print_running_return += current_ep_return
        print_running_episodes += 1

        log_running_return += current_ep_return
        log_running_episodes += 1

        cur_model_running_return += current_ep_return
        cur_model_running_episodes += 1

        # 写入批次缓存
        for key in batch_data:
            batch_data[key].append(episode_data[key])

        # 存储
        if (i_episode + 1) % batch_size == 0:
            print(colored("**** Episode Saved ****", 'green'))
            save_episode_to_hdf5(save_path, batch_data)
            batch_data = {key: [] for key in batch_data}
        
        i_episode += 1

    if batch_data["state"]:
        print(colored("**** Episode Saved ****", 'green'))
        save_episode_to_hdf5(save_path, batch_data)
    log_f.close()
    env.terminate()
    # env.close()

    # print total_step training time
    print("============================================================================================")
    _end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", _end_time)
    print("Total training time  : ", _end_time - start_time)
    print("============================================================================================")
    


if __name__ == '__main__':
    # 创建一个命令行接口，通过命令行传递参数来控制训练过程的开始 ID、训练次数以及是否使用 GPU
    """
    (conda Env: pytorch310)python ./train_test.py start_id train_times
    """
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Train PPO with optional arguments")
    # 添加位置参数
    parser.add_argument('start_id', type=int, help='Starting ID for training')
    parser.add_argument('train_times', type=int, help='Number of training times')
    # 添加可选的 --force-cpu 标志
    # parser.add_argument('--use-gpu', action='store_true', help='Use GPU if CUDA is available')
    # 添加可选的 --checkpoint-path 参数，用于传递模型检查点文件夹路径
    parser.add_argument('--checkpoint-path', type=str, default=None, help='Path to checkpoint folder')

    # 解析命令行参数
    args = parser.parse_args()

    # 从 args 中获取参数
    start_id = args.start_id
    train_times = args.train_times
    checkpoint_path = args.checkpoint_path  # 获取 checkpoint 路径

    total_delay = 1  # 总延迟时间（s）
    interval = total_delay if total_delay < 10 else 10  # 每10秒打印一次信息

    for remaining in range(total_delay, 0, -interval):
        print(f"程序将在 {remaining} 秒后开始...")
        time.sleep(interval)  # 等待10秒

    print("开始执行程序...")

    start_time = datetime.now().replace(microsecond=0)
    for i in range(start_id, start_id + train_times):
        print(f'======================== Training ID = {i} ========================')
        train(i, checkpoint_path=checkpoint_path)
        # if i:
        #     train(i)
        # else:
        #     train(i, checkpoint_path=checkpoint_path)

    print(
        colored("============================================================================================", 'red'))
    print(colored(f"Training_TEST for {train_times} times Ended.", 'red'))
    end_time = datetime.now().replace(microsecond=0)
    print(colored(f"Started training_test at (GMT) : {start_time}", 'red'))
    print(colored(f"Finished training_test at (GMT) : {end_time}", 'red'))
    print(colored(f"Total training_test time  : {end_time - start_time}", 'red'))
    print(
        colored(f"============================================================================================", 'red'))

    pass
