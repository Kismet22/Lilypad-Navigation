import torch
import numpy as np
import torch.nn as nn
from torch.distributions import MultivariateNormal
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
from torch.optim.lr_scheduler import ExponentialLR

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


def force_cpu_as_device():
    global device
    device = torch.device('cpu')
    print("============================================================================================")
    print("Device is FORCED set to : cpu")
    print("============================================================================================")


def init_weights(m):
    if isinstance(m, nn.Linear):
        # 使用正交初始化方法来初始化线性层 m 的权重
        torch.nn.init.orthogonal_(m.weight)


class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layer_dim, scale, bias, icm_alpha=0.5):
        super().__init__()
        self.act_dim = action_dim
        self.encode_dim = 64
        self.hidden_layer_dim = hidden_layer_dim
        self.icm_alpha = icm_alpha
        # scale and bias should be tensor
        self.scale = torch.tensor(scale, device=device)
        self.bias = torch.tensor(bias, device=device)

        # 最终输出为(batch_size, encode_dim)
        self.state_encode = nn.Sequential(
            nn.Linear(state_dim, self.hidden_layer_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_layer_dim, self.encode_dim),
            nn.Tanh()
        )

        # inverse
        # encode_state_t + encode_state_t+1 -> action_t
        # 最终输出为(batch_size, action_dim)
        self.inverse = nn.Sequential(
            nn.Linear(self.encode_dim * 2, self.hidden_layer_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_layer_dim, action_dim),
            nn.Tanh()
        )

        # forward
        # encode_state_t + action_t -> encode_state_t+1
        # 最终输出为(batch_size, encode_dim)
        self.icm_forward_net = nn.Sequential(
            nn.Linear(self.encode_dim + action_dim, self.hidden_layer_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_layer_dim, self.encode_dim),
            nn.Tanh()
        )

        self.state_encode.apply(init_weights)
        self.inverse.apply(init_weights)
        self.icm_forward_net.apply(init_weights)

        # # 初始化优化器
        # self.optimizer = torch.optim.Adam([
        #     # self.policy.actor.parameters() 提取actor网络的所有参数
        #     {'params': self.inverse.parameters(), 'lr': 1e-4},
        #     {'params': self.icm_forward_net.parameters(), 'lr': 1e-4}
        # ])

    def icm_forward(self, actions_in, states_in, next_states_in):
        # normalize
        if torch.isnan(next_states_in).any():
            #TODO:对Nan异常值的处理
            print("NaN detected in next_states_in!")
            epsilon = 1e-4  # 你可以调整这个值
            noise = torch.randn_like(states_in) * epsilon  # 生成一个小的随机噪声
            next_states_in = torch.where(torch.isnan(next_states_in), states_in + noise, next_states_in)
        encode_states = self.state_encode(states_in)
        encode_next_states = self.state_encode(next_states_in)


        # TODO:研究这里的维度为什么会发生改变,用.shape or .size()
        # 确保 actions_in 是 (N, action_dim)
        # 如果 actions_in 是 (N,)，则改为 (N, 1) 或者 (N, action_dim)
        if actions_in.dim() == 1:
            actions_in = actions_in.unsqueeze(-1)  # 将其转换为 (N, 1)

        # 逆模型：根据当前状态和下一状态预测动作
        inverse_in = torch.cat((encode_states, encode_next_states), dim=-1)
        predict_actions = self.inverse(inverse_in)  # 预测连续动作
        if torch.isnan(predict_actions).any():
            print("NaN detected in predict_actions!")
        predict_actions = predict_actions * self.scale + self.bias

        # 计算逆向损失（MSE损失，针对连续动作）
        inv_loss = nn.MSELoss()(predict_actions, actions_in).mean()

        # 前向模型：根据当前状态和动作预测下一状态
        forward_in = torch.cat((encode_states, actions_in), dim=-1)
        predict_next_states = self.icm_forward_net(forward_in)  # 预测下一状态

        # 基于前向模型的MSE损失
        # [batch_size, t, n]
        icm_forward_loss = 0.5 * nn.MSELoss(reduction='none')(predict_next_states, encode_next_states)
        icm_rewards = self.icm_alpha * icm_forward_loss.mean(dim=-1)  # [batch_size, t]
        # print(f"ICM_REWARD:{icm_reward}")
        icm_forward_loss = icm_forward_loss.mean()  # 标量
        return inv_loss, icm_forward_loss, icm_rewards


################################## PPO Policy ##################################
# TODO:增加一个对完整state的保存
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.next_states = []

    # 删除整个经验池中的数据
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.next_states[:]

    # 删除最后一个值
    def del_last(self):
        del self.actions[-1]
        del self.states[-1]
        del self.logprobs[-1]
        del self.rewards[-1]
        del self.state_values[-1]
        del self.is_terminals[-1]
        del self.next_states[-1]

    # 打印每一个元素的长度
    def print_lengths(self):
        print(f"Actions length: {len(self.actions)}")
        print(f"States length: {len(self.states)}")
        print(f"Logprobs length: {len(self.logprobs)}")
        print(f"Rewards length: {len(self.rewards)}")
        print(f"State values length: {len(self.state_values)}")
        print(f"Is terminals length: {len(self.is_terminals)}")
        print(f"Next states length: {len(self.next_states)}")


class ActorCritic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_layer_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        # 输入网络的矩阵形状为(batch_size, state_dim)
        self.has_continuous_action_space = has_continuous_action_space
        self.hidden_layer_dim = hidden_layer_dim

        if has_continuous_action_space:
            self.action_dim = action_dim
            # 一维方差列表，长度为动作维度个，值为初始化标准差的平方(方差)
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, self.hidden_layer_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_layer_dim, action_dim),
                # 最终输出为(batch_size, action_dim)
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, self.hidden_layer_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_layer_dim, action_dim),
                # 最终输出为(batch_size, action_dim)
                nn.Softmax(dim=-1)  # 依照最后一个维度,在该维度上进行Softmax归一化操作,即在action_dim这个维度上进行归一化
            )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, self.hidden_layer_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_layer_dim, self.hidden_layer_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_layer_dim, 1)
        )
        # TODO: Norm Para
        # 矩阵正交初始化操作
        self.actor.apply(init_weights)
        self.critic.apply(init_weights)

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        """
        给定输入状态，返回一个动作
        :param state:
        :return: 动作、动作的log概率、state状态函数（critic直接输出）
        """
        if self.has_continuous_action_space:
            # 动作的均值
            # (batch_size, action_dim)
            action_mean = self.actor(state)
            # 动作的协方差矩阵
            # .diag创建一个对角阵，对角线上的值为对应位置的方差 (action_dim, action_dim)
            # .unsqueeze的作用是让个数与batch_size相匹配
            # (1, (action_dim, action_dim))
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            # 创建一个多元正态分布
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        # 从创建的分布中采样一个动作
        action = dist.sample()
        # 计算采样的动作 action 在概率分布 dist 下的对数概率
        action_logprob = dist.log_prob(action)
        # 通过critic网络对当前状态 state 进行评价，得到对应状态估值
        state_val = self.critic(state)
        # .detach()该tensor不参与梯度计算
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        """
        评估state-action pair
        :param state:
        :param action:
        :return: action_log_probs, state_values, dist_entropy
        """
        # action:(batch_size, action_dim)
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            assert not torch.isnan(action_mean).any(), "action_mean contains NaN values!"
            assert not torch.isinf(action_mean).any(), "action_mean contains inf values!"
            # cov_mat = torch.diag_embed(action_var).to(device)
            cov_mat = torch.diag_embed(action_var).to(device) + torch.eye(self.action_dim, device=device) * 1e-4
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                # 初始，action:(batch_size, 1)
                # 转化为(batch_size, 1)
                # .reshape(-1, self.action_dim),矩阵根据第二个元素调整
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_log_probs, state_values, dist_entropy


class ICM_PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic,
                 gamma, K_epochs, eps_clip, has_continuous_action_space,
                 action_std_init=0.6, hidden_layer_dim=128,
                 continuous_action_output_scale=1.0, continuous_action_output_bias=0.0, icm_alpha=0.5):
        """
        PPO constructor.
        :param state_dim: 状态空间维数
        :param action_dim: 动作空间维数
        :param lr_actor: actor网络的学习率
        :param lr_critic: critic网络的学习率
        :param gamma: 奖励值累加时的衰减率 r＋γr_after
        :param K_epochs: 更新优化的轮数epochs
        :param eps_clip: PPO2方法的裁剪率
        :param has_continuous_action_space: 动作空间是否连续
        :param action_std_init: 连续动作空间的分布标准差
        :param continuous_action_output_scale: 连续动作空间时，将默认输出[-1, 1]缩放的倍数
        :param continuous_action_output_bias:  连续动作空间时，将默认输出[-1, 1]缩放后的偏置
        # 解释:网络输出经过归一化，范围为[-1, 1]
        # 实际动作空间范围为[a, b]
        # 映射方式:scale = (b - a)/(1 - (-1)) bias = b - (b - a)/2 = (b + a)/2
        """
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            # 连续动作空间时的动作矩阵方差
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.icm_alpha = icm_alpha
        self.continuous_action_output_scale = continuous_action_output_scale
        self.continuous_action_output_bias = continuous_action_output_bias

        self.buffer = RolloutBuffer()  # 数据池

        # PPO policy_net
        self.policy = ActorCritic(state_dim, action_dim, hidden_layer_dim, has_continuous_action_space,
                                  action_std_init).to(device)
        self.icm = ICM(state_dim=state_dim, action_dim=action_dim, hidden_layer_dim=hidden_layer_dim,
                       scale=continuous_action_output_scale, bias=continuous_action_output_bias, icm_alpha=self.icm_alpha).to(device)
        # icm_reward / reward
        self.icm_average_ratio = 1

        # 设置优化器，对策略A2C的actor和critic分别进行优化
        # 在强化学习中，优化器会通过反向传播来更新 actor 和 critic 的参数
        # 这里使用的是 Adam 优化器，它是一种基于一阶矩估计的自适应学习率优化算法
        # self.optimizer = torch.optim.Adam([
        #     # self.policy.actor.parameters() 提取actor网络的所有参数
        #     {'params': self.policy.actor.parameters(), 'lr': lr_actor},
        #     {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        # ])

        self.optimizer = torch.optim.Adam([
            # self.policy.actor.parameters() 提取actor网络的所有参数
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
            {'params': self.icm.inverse.parameters(), 'lr': 2e-4},
            {'params': self.icm.icm_forward_net.parameters(), 'lr': 2e-4}
        ])

        # PPO old policy_net
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_layer_dim, has_continuous_action_space,
                                      action_std_init).to(device)
        # 旧策略网络保存上一个策略的所有参数
        # .loat_state_dict:写入网络数据
        # .state_dict():读取网络数据
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 第二部分的损失值:A2C网络的MSE损失值
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        # 动作矩阵方差衰减函数
        # 对动作矩阵的方差进行一定数值上的减少，但不低于最低的方差值
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")
        return self.action_std

    def select_action(self, input_state):
        """
        输入state,返回要采取的action,并保存action, action_log_prob, state_val至buffer
        :param input_state:
        :return: 经过线性变换的action
        """

        if isinstance(input_state, dict):
            state = list(input_state.values())
        else:
            state = input_state

        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32).to(device)
        elif isinstance(state, list):
            state = torch.tensor(state, dtype=torch.float32).to(device)
        elif isinstance(state, torch.Tensor):
            state = state.to(device)
        else:
            raise TypeError(f"Unsupported state type: {type(state)}")

        # 其余代码保持不变
        # 先处理连续动作情况，使用old_policy做一次act
        if self.has_continuous_action_space:
            # old_policy不保存梯度
            with torch.no_grad():
                # 走一步，保存动作、动作的log概率、value_function
                action, action_log_prob, state_val = self.policy_old.act(state)

            # 在经验池中保存old_policy的决策的相关数据
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_log_prob)
            self.buffer.state_values.append(state_val)
            # 对action做线性变换
            action_raw = action.detach().cpu().numpy().flatten()
            # action输出的范围为(-1, 1)
            action_out = action_raw * self.continuous_action_output_scale + self.continuous_action_output_bias
            return action_out

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_log_prob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_log_prob)
            self.buffer.state_values.append(state_val)

            # .item()用于将包含一个元素的张量转换为 Python 的标量值。这个方法只能用于张量中仅包含一个元素的情况
            return action.item()

    def update(self):
        """
        更新K次网络。buffer中action, action_log_prob, state_val在select_action时保存，reward与terminal从外部加入。
        :return: None
        """
        # Step1: 提取state action state_value和log_probs到gpu
        # buffer中原本是众多连续的一维张量
        # 操作目的，将多个状态一起按同一批次进行处理
        # convert list to tensor
        # .stack()声明了堆叠方式,将张量在第'dim=n'的方向上进行了堆叠,在这里是第0维(batch_size)维度
        # .detach()分离张量，使得它不再参与梯度计算，这是为了确保训练中的状态张量不再影响后续的梯度更新
        # .squeeze()这个函数用于移除张量中维度为 1 的所有维度 (N,1)会变成(N,)
        states_on_device = []
        for state in self.buffer.states:  # 检查并转换 self.buffer.states 中的每个项
            if isinstance(state, torch.Tensor):  # 如果已经是张量，确保它在正确的设备上
                states_on_device.append(state.to(device))
            else:  # 如果不是张量，将其转换为张量并放到正确的设备上
                states_on_device.append(torch.tensor(state, dtype=torch.float32, device=device))
        old_states = torch.squeeze(torch.stack(states_on_device, dim=0)).detach().to(device)  # 进行堆叠和其他操作

        next_states_on_device = []
        for state in self.buffer.next_states:  # 检查并转换 self.buffer.next_states 中的每个项
            if isinstance(state, torch.Tensor):  # 如果已经是张量，确保它在正确的设备上
                next_states_on_device.append(state.to(device))
            else:  # 如果不是张量，将其转换为张量并放到正确的设备上
                next_states_on_device.append(torch.tensor(state, dtype=torch.float32, device=device))
        old_next_states = torch.squeeze(torch.stack(next_states_on_device, dim=0)).detach().to(device)  # 进行堆叠和其他操作

        actions_on_device = []
        for action in self.buffer.actions:  # 检查并转换 self.buffer.actions 中的每个项
            if isinstance(action, torch.Tensor):  # 如果已经是张量，确保它在正确的设备上
                actions_on_device.append(action.to(device))
            else:  # 如果不是张量，将其转换为张量并放到正确的设备上
                actions_on_device.append(torch.tensor(action, dtype=torch.float32, device=device))
        old_actions = torch.squeeze(torch.stack(actions_on_device, dim=0)).detach().to(device)  # 进行堆叠和其他操作

        old_log_probs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        if torch.isnan(old_states).any() or torch.isinf(old_states).any():
            print("NaN or Inf detected in old_states")

        # Step2: 通过icm计算icm_reward
        # icm_alpha = 0.5
        with torch.no_grad():
            # .detach():防止与原始计算图共享梯度
            old_actions_detached = old_actions.detach()
            old_states_detached = old_states.detach()
            old_next_states_detached = old_next_states.detach()
            _, _, icm_rewards = self.icm.icm_forward(actions_in=old_actions_detached, states_in=old_states_detached,
                                                     next_states_in=old_next_states_detached)

        # Step3: Monte Carlo estimate of returns
        discount_rewards = []
        reward_icm_ratios = []  # 存储每步的 reward / icm_reward 比例
        discounted_reward = 0
        for reward, icm_reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(icm_rewards),
                                                   reversed(self.buffer.is_terminals)):
            if is_terminal:  # 当一个时间步是终止状态时，之后的折扣回报将从零开始计算,这里是翻转之后的，则应该表示终止之前的计算
                discounted_reward = 0
            combined_reward = reward + icm_reward  # 外部奖励 + 内部奖励
            discounted_reward = combined_reward + (self.gamma * discounted_reward)
            discount_rewards.insert(0, discounted_reward)
            # 计算 reward / icm_reward，避免除零错误
            if icm_reward != 0:
                if reward >=400:
                    reward_icm_ratios.append(abs((reward-500) / icm_reward))
                else:
                    reward_icm_ratios.append(abs(reward / icm_reward))
            else:
                reward_icm_ratios.append(float('inf'))  # 如果 icm_reward 为 0，则设为无穷大（或其他适当的值）
        # 计算平均比例，忽略 inf 值
        valid_ratios = [r for r in reward_icm_ratios if r != float('inf')]
        average_ratio = sum(valid_ratios) / len(valid_ratios) if valid_ratios else 0

        print(f"Average Reward/ICM Ratio: {average_ratio:.4f}")
        self.icm_average_ratio = average_ratio

        # 倒着处理buffer，为每一步生成一个monte-carlo的value-function采样，代表在某个状态下一直MC采样到末尾得到的return
        # 折扣回报的计算方式，符合r = r + γΣr
        # reversed()反转操作
        # zip()将两个可迭代的对象配对到一起

        discount_rewards = torch.tensor(discount_rewards, dtype=torch.float32).to(device)  # 将数值转化为一个可用于网络计算的tensor形式

        # Normalizing the discount_reward
        # discount_rewards = (discount_rewards - discount_rewards.mean()) / (discount_rewards.std() + 1e-7)

        # Step4:calculate advantages
        advantages = discount_rewards.detach() - old_state_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # Finding Surrogate Loss
        # PPO第一部分损失函数的计算过程
        # A(a|s)为r与critic_value的差值
        # old_state_values是老的critic网络对每个状态的v值估计，而rewards是经过采样【且归一化】后的真实reward

        # Optimize policy for K epochs
        # 在K轮epochs中，优势函数不改变
        for _ in range(self.K_epochs):
            # Evaluating old actions and values，在新的policy下，评价老的数据。拿到一块时间步的log_probs, state_values
            # dist_entropy是网络内部的交叉熵，第三部分损失函数
            old_states = old_states.requires_grad_(True)  # 确保 old_states 可计算梯度

            log_probs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old)，因为带着个log所以用exp，评价新老policy的差异
            # exp(log(new)- log(old)) = new/old
            ratios = torch.exp(log_probs - old_log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            # 第一项是给actor网络用的，第二项是给critic网络用的，第三项也是给actor的，避免策略过早收敛
            # 虽然loss写在了一起，但两个网络会各自计算和自己相关的梯度，这没什么问题
            state_values = state_values.squeeze()  # 删除维度为1的轴,使两者的维度一致

            icm_inverse_loss, icm_forward_loss, _ = self.icm.icm_forward(actions_in=old_actions, states_in=old_states,
                                                                         next_states_in=old_next_states)
            icm_loss = icm_inverse_loss + icm_forward_loss  # 两个loss已经进行了.mean()操作
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, discount_rewards) - 0.01 * dist_entropy
            loss = loss.mean() + icm_loss
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            # loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # # ICM_update
        # for _ in range(self.K_epochs):
        #     icm_inverse_loss, icm_forward_loss, _ = self.icm.icm_forward(actions_in=old_actions, states_in=old_states, next_states_in=old_next_states, icm_alpha=icm_alpha)
        #     # 计算 ICM 的损失，并更新 ICM 网络
        #     icm_loss = icm_inverse_loss + icm_forward_loss
        #     icm_loss.backward()
        #     self.icm.optimizer.step()

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        # 保存网络数据
        # 将self.policy_old.state_dict()的数据以字典的形式保存到checkpoint_path中去
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def save_icm(self, checkpoint_path):
        # 保存网络数据
        # 将self.icm.state_dict()的数据以字典的形式保存到checkpoint_path中去
        torch.save(self.icm.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        # 加载网络数据
        # 'map_location'参数用于控制模型参数加载到的设备
        # lambda storage, loc: storage是一个匿名函数，将确保模型参数被加载到与保存时相同的设备上，或者将其映射到当前可用的设备
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def load_icm(self, checkpoint_path):
        self.icm.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    # 加一组更完整的保存的function，能把Adam中自适应学习率的部分也保存好，可以接续训练
    def save_full(self, filename):
        # 不止保存了网络结构数据，同时保存了优化器的参数
        print("=> Saving checkpoint")
        checkpoint = {
            "model_state_dict": self.policy_old.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        torch.save(checkpoint, filename)


    def load_full(self, filename):
        # 将网络数据和优化器数据都加载出来
        print("=> Loading checkpoint")
        checkpoint = torch.load(filename, weights_only=False)
        self.policy_old.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_icm_dict'])
        self.policy.load_state_dict(self.policy_old.state_dict())  # Ensure both policies are synced


    def save_full_icm(self, filename):
        # 不止保存了网络结构数据，同时保存了优化器的参数
        print("=> Saving checkpoint")
        checkpoint = {
            "model_state_dict": self.policy_old.state_dict(),
            "model_icm_dict": self.icm.state_dict(),
            "optimizer_icm_dict": self.optimizer.state_dict()
        }
        torch.save(checkpoint, filename)


    def load_full_icm(self, filename):
        # 将网络数据和优化器数据都加载出来
        print("=> Loading checkpoint")
        checkpoint = torch.load(filename, weights_only=False)
        self.policy_old.load_state_dict(checkpoint['model_state_dict'])
        self.icm.load_state_dict(checkpoint['model_icm_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_icm_dict'])  # load
