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


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    # 删除整个经验池中的数据
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    # 删除最后一个值
    def del_last(self):
        del self.actions[-1]
        del self.states[-1]
        del self.logprobs[-1]
        del self.rewards[-1]
        del self.state_values[-1]
        del self.is_terminals[-1]


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
                # 最终输出为(batch_size, action_dim),范围在[-1, 1]
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

            # print("action mean:", action_mean) # todo

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
        :return: action_logprobs, state_values, dist_entropy
        """
        # action:(batch_size, action_dim)
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
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
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class Classic_PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic,
                 gamma, K_epochs, eps_clip, has_continuous_action_space,
                 action_std_init=0.6, hidden_layer_dim=128,
                 continuous_action_output_scale=1.0, continuous_action_output_bias=0.0, mini_batch_size=32):
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
        self.continuous_action_output_scale = continuous_action_output_scale
        self.continuous_action_output_bias = continuous_action_output_bias
        self.mini_batch_size = mini_batch_size

        self.buffer = RolloutBuffer()  # 数据池

        # PPO policy_net
        self.policy = ActorCritic(state_dim, action_dim, hidden_layer_dim, has_continuous_action_space,
                                  action_std_init).to(device)

        # 设置优化器，对策略A2C的actor和critic分别进行优化
        # 在强化学习中，优化器会通过反向传播来更新 actor 和 critic 的参数
        # 这里使用的是 Adam 优化器，它是一种基于一阶矩估计的自适应学习率优化算法
        self.optimizer = torch.optim.Adam([
            # self.policy.actor.parameters() 提取actor网络的所有参数
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
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
        输入state,返回要采取的action,并保存action, action_logprob, state_val至buffer
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
                action, action_logprob, state_val = self.policy_old.act(state)

            # 在经验池中保存old_policy的决策的相关数据
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)
            # 对action做线性变换
            action_raw = action.detach().cpu().numpy().flatten()
            # print("action_raw:", action_raw)
            # action输出的范围为(-1, 1)
            action_out = action_raw * self.continuous_action_output_scale + self.continuous_action_output_bias
            # print("action_out:", action_out)
            return action_out

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            # .item()用于将包含一个元素的张量转换为 Python 的标量值。这个方法只能用于张量中仅包含一个元素的情况
            return action.item()

    def update(self):
        """
        更新K次网络。buffer中action, action_logprob, state_val在select_action时保存,reward与terminal从外部加入。
        :return: None
        """
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0

        # 倒着处理buffer，为每一步生成一个monte-carlo的value-function采样
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Convert rewards to tensor and move to device
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

        # 将buffer中的state、action等按mini-batch分批读取
        batch_size = len(self.buffer.states)
        num_batches = batch_size // self.mini_batch_size  # 根据mini_batch_size计算batch的数量

        # 优化策略K轮
        for _ in range(self.K_epochs):
            for i in range(num_batches):
                # 逐步加载mini-batch数据
                start_idx = i * self.mini_batch_size
                end_idx = (i + 1) * self.mini_batch_size

                # 检查并转换 states 数据类型
                mini_batch_states = []
                for state in self.buffer.states[start_idx:end_idx]:
                    if isinstance(state, torch.Tensor):
                        mini_batch_states.append(state.to(device))
                    else:
                        mini_batch_states.append(torch.tensor(state, dtype=torch.float32, device=device))
                mini_batch_states = torch.stack(mini_batch_states)

                # 检查并转换 actions 数据类型
                mini_batch_actions = []
                for action in self.buffer.actions[start_idx:end_idx]:
                    if isinstance(action, torch.Tensor):
                        mini_batch_actions.append(action.to(device))
                    else:
                        mini_batch_actions.append(torch.tensor(action, dtype=torch.float32, device=device))
                mini_batch_actions = torch.stack(mini_batch_actions)

                # 检查并转换 logprobs 数据类型
                mini_batch_logprobs = []
                for logprob in self.buffer.logprobs[start_idx:end_idx]:
                    if isinstance(logprob, torch.Tensor):
                        mini_batch_logprobs.append(logprob.to(device))
                    else:
                        mini_batch_logprobs.append(torch.tensor(logprob, dtype=torch.float32, device=device))
                mini_batch_logprobs = torch.stack(mini_batch_logprobs)

                # 检查并转换 state_values 数据类型
                mini_batch_state_values = []
                for state_value in self.buffer.state_values[start_idx:end_idx]:
                    if isinstance(state_value, torch.Tensor):
                        mini_batch_state_values.append(state_value.to(device))
                    else:
                        mini_batch_state_values.append(torch.tensor(state_value, dtype=torch.float32, device=device))
                mini_batch_state_values = torch.stack(mini_batch_state_values)

                # 使用 rewards 来计算 advantages
                # rewards中保存的是全局计算得到的价值,这样设置没有问题
                mini_batch_rewards = rewards[start_idx:end_idx]  # Define mini_batch_rewards here
                mini_batch_advantages = mini_batch_rewards - mini_batch_state_values

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(mini_batch_states, mini_batch_actions)

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - mini_batch_logprobs.detach())

                surr1 = ratios * mini_batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mini_batch_advantages
                state_values = state_values.squeeze(-1)

                # Final loss of clipped objective PPO
                loss = (-torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, mini_batch_rewards) -
                        0.01 * dist_entropy)
                # 梯度优化过程
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        # 保存网络数据
        # 将self.policy_old.state_dict()的数据以字典的形式保存到checkpoint_path中去
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        # 加载网络数据
        # 'map_location'参数用于控制模型参数加载到的设备
        # lambda storage, loc: storage是一个匿名函数，将确保模型参数被加载到与保存时相同的设备上，或者将其映射到当前可用的设备
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    # 加一组更完整的保存的function，能把Adam中自适应学习率的部分也保存好，可以接续训练
    def save_full(self, filename):
        # 不止保存了网络结构数据，同时保存了优化器的参数
        print("=> Saving checkpoint")
        checkpoint = {
            "model_state_dict": self.policy_old.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        torch.save(checkpoint, filename)

    def load_full(self, filename, _init_std=None):
        # 将网络数据和优化器数据都加载出来
        print("=> Loading checkpoint")
        checkpoint = torch.load(filename, weights_only=False)
        self.policy_old.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy.load_state_dict(self.policy_old.state_dict())  # Ensure both policies are synced
