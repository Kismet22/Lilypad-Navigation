import gym
import my_ppo_net
from my_ppo_net import Classic_PPO
import os
import argparse
import torch
import numpy as np
from datetime import datetime
from termcolor import colored
import time
from env import flow_field_env

# 将设备设置与hjbppo.py中相同
device = my_ppo_net.device


def train(model_id, render_mode=None):
    print("============================================================================================")

    ####### 训练环境变量设置 #######
    # 环境名称，用于保存
    env_name = 'Flow_env'

    # 判断动作空间是否连续
    # continuous action space; else discrete
    has_continuous_action_space = True

    # 一个episode中的最大时间步数,存在环境时间限制时,最大时间步应该大于环境时间限制
    # max time_steps in one episode
    max_steps = 400
    max_ep_len = max_steps + 20
    # 结束训练的总训练步数
    # break training loop if timesteps > max_training_timesteps
    max_training_timesteps = int(1e6)

    # 打印/保存episode奖励均值
    # Note : print/log frequencies should be more than max_ep_len
    # print avg reward in the interval (in num timesteps)
    print_freq = max_ep_len * 5
    # log avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 5

    # 存储模型间隔
    # save model frequency (in num timesteps)
    save_model_freq = int(1e4)

    # 注意，这里的标准差信息都是针对网络直接输出而言，即最终网络激活函数的归一化输出[-1, 1]作为均值的方差
    # 初始方差
    action_std = 0.4  # starting std for action distribution (Multivariate Normal)
    # 方差更新缩减值
    action_std_decay_rate = 0.012  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    # 最小方差
    min_action_std = 0.04  # minimum action_std (stop decay after action_std <= min_action_std)
    # 方差缩减频率
    action_std_decay_freq = int(2e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ################ 强化学习超参数设置 ################

    # 网络更新频率
    # update policy every n timesteps
    update_timestep = max_ep_len * 15
    # update policy for K epochs in one PPO update
    K_epochs = 40

    eps_clip = 0.2  # clip rate for PPO
    gamma = 0.99  # discount factor γ

    lr_actor = 0.0001  # learning rate for actor network
    lr_critic = 0.0002  # learning rate for critic network

    # seed = 0 开始状态不固定，随机初始化
    random_seed = 0  # set random seed if required (0 = no random seed)
    # random_seed = 1

    pretrained_model_ID = model_id  # 训练ID, change this to prevent overwriting weights in same env_name folder
    #####################################################

    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10
    env = flow_field_env.foil_env(args_1, max_step=max_steps)

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

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 训练环境文件夹
    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # get number of log files in log directory
    # current_num_files = next(os.walk(log_dir))[2]
    # run_num = len(current_num_files)

    # create new log file for each run
    ######################  训练过程记录文件 ######################
    rl_method = '/PPO'
    ######################

    log_f_name = log_dir + rl_method + env_name + "_log_" + str(pretrained_model_ID) + ".csv"

    print("current logging model ID for " + env_name + " : ", pretrained_model_ID)
    print("logging at : " + log_f_name)
    if os.path.exists(log_f_name):
        raise ValueError(f"Check model ID. Current ID = {pretrained_model_ID} is already used.")
    #####################################################

    ###################### 保存模型信息 ######################

    ######################  文件夹 ######################
    directory = "PPO_preTrained"
    ######################

    if not os.path.exists(directory):
        os.makedirs(directory)

    ######################  文件名 ######################
    method_name = "PPO_{}_ID_{}_seed_{}"
    ######################

    directory = directory + '/' + env_name + '/' + method_name.format(env_name, pretrained_model_ID,
                                                                      random_seed) + '/'
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
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    print("============================================================================================")
    #####################################################

    ################# training procedure ################

    # initialize RL agent
    ppo_agent = Classic_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                            has_continuous_action_space,
                            action_std, continuous_action_output_scale=action_output_scale,
                            continuous_action_output_bias=action_output_bias)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    # 表头标题写入
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,avg_return_in_period,action_std\n')
    log_f.close()

    # 重新打开文件读取内容
    with open(log_f_name, "r") as f:
        content = f.read()
        print("文件内容如下:")
        print(content)

    # 检查表头是否正确
    if 'episode,timestep,avg_return_in_period,action_std\n' in content:
        print("表头写入成功")
    else:
        print("表头写入失败")

    # 追加写入，防止覆盖原表头
    log_f = open(log_f_name, "a")
    # printing and logging variables
    print_running_return = 0
    print_running_episodes = 0
    log_running_return = 0
    log_running_episodes = 0

    cur_model_running_return = 0
    cur_model_running_episodes = 0
    max_model_avg_return = -np.inf

    time_step = 0
    i_episode = 0

    # log average return till last episode
    log_f.write('{},{},{},{}\n'.format(111, 222, 333, 444))
    log_f.flush()

    # training loop
    while time_step <= max_training_timesteps:
        # 在最大训练步长以内
        state, info = env.reset()
        print("position_reset:", env.agent_pos)

        current_ep_return = 0

        # 在最大更新步长以内
        for t in range(1, max_ep_len + 1):
            # select action with policy
            action = ppo_agent.select_action(state)

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

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
                    ppo_agent.save_full(checkpoint_path)
                    max_model_avg_return = cur_model_avg_return
                cur_model_running_return = 0
                cur_model_running_episodes = 0

                print("Policy Updated. At Episode : {}  Timestep : {}".format(i_episode, time_step))
                ppo_agent.update()

            # if continuous action space; then decay action std of output action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                # 每action_std_decay_freq步，降低一次网络的初始化方差
                action_std = ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file， 多个episode进行一次记录，记录的是这几个episode的avg return
            if time_step % log_freq == 0:
                # log average return till last episode
                log_avg_return = log_running_return / log_running_episodes
                log_avg_return = round(log_avg_return, 4)

                log_f.write('{},{},{},{}\n'.format(i_episode, time_step, log_avg_return, action_std))
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
                ppo_agent.save_full(checkpoint_path)
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

        i_episode += 1

    log_f.close()
    # env.close()
    env.terminate()

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
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU if CUDA is available')
    # 解析命令行参数
    args = parser.parse_args()
    # 从 args 中获取参数
    start_id = args.start_id
    train_times = args.train_times

    total_delay = 1  # 总延迟时间（s）
    interval = total_delay if total_delay < 10 else 10  # 每10秒打印一次信息

    for remaining in range(total_delay, 0, -interval):
        print(f"程序将在 {remaining} 秒后开始...")
        time.sleep(interval)  # 等待10秒

    print("开始执行程序...")

    start_time = datetime.now().replace(microsecond=0)
    for i in range(start_id, start_id + train_times):
        print(f'======================== Training ID = {i} ========================')
        train(i)

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
