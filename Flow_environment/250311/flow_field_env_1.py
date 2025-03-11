from xmlrpc.client import ServerProxy
import subprocess
import json
import time
import random
import numpy as np
import argparse
from gym.spaces import Box
import os
import signal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from termcolor import colored
from math import *
# 该环境用于训练使用
# 仅输出位置信息，用作对比



def is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

class foil_env:
    def __init__(self, config=None, info='', local_port=None, network_port=None, target_center=None, target_position=None, start_center=None, start_position=None, max_step=None, include_flow=False):
        # n = 20 * 16  m = 8 * 16 r = 3 * 16 每个圆柱的半径是1*16
        # m 和 n 对应模拟窗格的大小
        # 实际的位移大小应该是Δx/16
        # 三角形排布 [n/6, n/6, n/6+r*cos(theta)] [m/2 + r/2, m/2 - r/2, m/2]
        # 智能体位置：[6*n/8 , m/2](240, 64)
        self.x_range = 20 * 16
        self.y_range = 8 * 16
        self.body_r = 1 * 16
        self.n = 20 * 16
        self.m = 8 * 16
        self.r = 3 * 16
        # 智能体椭圆信息
        self.ellipse_b = self.body_r/4 # 竖直摆放的椭圆 b = 1.5a
        self.ellipse_a = self.ellipse_b / 1.5
        # self.circles = [
        #     {"center": (self.n/6, self.m/2 + self.r/2), "radius": self.body_r/2},
        #     {"center": (self.n/6, self.m/2 - self.r/2), "radius": self.body_r/2},
        #     {"center": (self.n/6 + self.r*cos(pi/6), self.m/2), "radius": self.body_r/2}
        # ]
        self.circles = [
            {"center": (53, 32), "radius": self.body_r/2},
            {"center": (53, 96), "radius": self.body_r/2},
            {"center": (109, 64), "radius": self.body_r/2}
        ]
        


        # TODO:对比环境，将状态空间大小降为6
        # self.observation_dim = 6


        self.observation_dim = 14
        self.action_dim = 3
        self.step_counter = 0
        self.steps_default = 300
        if max_step:
            self.t_step_max = max_step
        else:
            self.t_step_max = self.steps_default
        self.state = np.zeros(self.observation_dim)
        # TODO:change action range
        # 参考:[15, 15, 10]
        # 20250206,选择范围为[20, 20, 5]
        low = np.array([-20.0, -20, -5], dtype=np.float32)  # 每个维度的下界
        high = np.array([20.0, 20, 5], dtype=np.float32)  # 每个维度的上界
        self.low = low
        self.high = high
        self.agent_pos = np.array([0.0, 0.0])
        self.agent_pos_history = []
        self.action_interval = config.action_interval
        self.dt = 0.075
        self.unwrapped = self
        self.unwrapped.spec = None
        self.observation_space = Box(low=-1e6, high=1e6, shape=[self.observation_dim])
        self.action_space = Box(low=low, high=high, shape=(self.action_dim,), dtype=np.float32)
        self.local_port = local_port
        
        # 起终点位置选取
        ##############################################################
        self.target_default = np.array([self.n/6 + self.r*cos(pi/6)/2, self.m/2])
        if target_center is not None:
            self.target_default = target_center
        # [240, 64]
        self.start_default = np.array([6*self.n/8, self.m/2])
        if start_center is not None:
            self.start_default = start_center
        self.target_position = self.target_default
        self.start_position = self.start_default
        self._is_fixed_target = True
        self._is_fixed_start = True
        self.target_r = 0.5  # 到达终点判定的范围
        if target_position is not None:
            # 如果手动输入目标点坐标
            self.target_default = target_position
            self.target_position = target_position
        else:
            # 否则随机选取目标点坐标
            self._is_fixed_target = False
        if start_position is not None:
            # 如果手动输入起点坐标
            self.start_default = start_position
            self.start_position = start_position
        else:
            # 否则随机选取起点坐标
            self._is_fixed_start = False
        # 初始距离，用于距离奖励函数的归一化
        self.d_init = 10
        ##############################################################

        # TODO:考虑加入角度信息
        # self._previous_angle = 0
        self.angle = 0
        self.vel_angle = 0
        
        self.reward = 0
        self.done = False
        self.d_2_target = 0
        self.pressure = []
        self.pre_pressure = []
        # 速度相关记录
        self.u_flow = []
        self.v_flow = []
        self.speed = 0
        self.flow_speed = 0
        self.include_flow = include_flow
        self._roll_count = 0
        self._collision_count = 0
        self._roll_punish = -20
        self.flow_info = []

        # 绘图
        self.frame_pause = 0.03 # 渲染间隔时间
        self.fig, self.ax = plt.subplots(figsize=(16, 9)) # 创建子图

        # TODO:从这里多传一段信息，增加一个随机选取起点的功能
        flow_flag = 'true' if self.include_flow else 'false'
        while True:
            port = random.randint(6000, 8000)
            if not is_port_in_use(port):
                break
        port = port if network_port == None else network_port
        # print("Foil_env start!")
        if local_port == None:
            command = (
                f'xvfb-run -a /home/zhengshizhan/workspace/processing-4.3/processing-java '
                f'--sketch=/home/zhengshizhan/project1/Navigation_understand-master/PinballCFD_server --run '
                f'{port} {flow_flag} {info}'  # 注意加上空格分隔参数
            )
            # 使用 subprocess 启动命令
            self.server = subprocess.Popen(command, shell=True)
            # wait the server to start
            time.sleep(20)
            print("server start")
            self.proxy = ServerProxy(f"http://localhost:{port}/")
        else:
            self.proxy = ServerProxy(f"http://localhost:{local_port}/")


    def normalize_angle(self, angle):
        return atan2(sin(angle), cos(angle))


    # def pressure_reward(self):
    #     pressure = self.pressure
    #     pre_pressure = self.pre_pressure

    #     # Threshold
    #     pressure_threshold = 5.0

    #     over_threshold_pressure = np.maximum(np.abs(pressure) - pressure_threshold, 0)
    #     penalty_pressure = np.sum(over_threshold_pressure)

    #     # 如果有超过阈值的压力项，给予负奖励
    #     reward_penalty = -5 * penalty_pressure if penalty_pressure > 0 else 0

    #     # 返回总奖励
    #     return reward_penalty



    def get_flow_velocity(self, r_large):
        """
        Compute the velocity field around the agent using np.pad to handle out-of-bounds indices.
        
        Parameters:
        r_large : Half-length of the larger square sensing range (8)
        
        Returns:
        velocities : A flattened list in the order of [vx0, vy0, vx1, vy1, ...] (416 * 1)
        """
        v_x = self.u_flow
        v_y = self.v_flow
        x, y = self.agent_pos
        a = self.ellipse_b  # 小正方形半边长
        # 获取原始流场大小
        H, W = v_x.shape

        # 检查是否有 NaN 位置
        if np.isnan(x) or np.isnan(y):
            print(f"Warning: Agent position contains NaN (x={x}, y={y}).")
            return self.flow_info
        
        # 如果 x 或 y 过大，进行特殊处理
        if x < -r_large or y < -r_large or x >= H + r_large or y >= W + r_large:
            print(f"Warning: Agent position out of bounds (x={x}, y={y}).")
            return self.flow_info

        # 设定安全填充量
        safety_margin = 4  # 可根据实际情况调整
        pad_width = r_large + safety_margin

        # 进行镜像填充，pad 的大小足够大，确保边界外的索引可以访问
        v_x_padded = np.pad(v_x, pad_width=pad_width, mode='reflect')
        v_y_padded = np.pad(v_y, pad_width=pad_width, mode='reflect')

        # 更新坐标，使其适配填充后的数据
        x_padded = int(x + pad_width)
        y_padded = int(y + pad_width)

        velocities = []

        # 遍历较大正方形区域
        for i in range(x_padded - r_large, x_padded + r_large + 1):
            for j in range(y_padded - r_large, y_padded + r_large + 1):
                # 检查是否在较大正方形范围内
                if abs(i - x_padded) <= r_large and abs(j - y_padded) <= r_large:
                    # 排除小正方形区域
                    if abs(i - x_padded) > a or abs(j - y_padded) > a:
                        # 取填充后的数据
                        vx = v_x_padded[i, j]
                        vy = v_y_padded[i, j]
                        # 交错存储 [vx0, vy0, vx1, vy1, ...]
                        velocities.append(vx)
                        velocities.append(vy)

        return velocities


    def step(self, action):
        # 动作裁剪
        action = self.clip_action(action)
        _truncated = False
        _terminated = False
        _reward = 0
        state_1 = self.state
        self.step_counter += 1
        pos_cur = self.agent_pos
        step_1 = float(action[0])  # x 加速度
        step_2 = float(action[1])  # y 加速度
        step_3 = float(action[2])  # 旋转加速度
        action_json = {"v1": step_1, "v2": step_2, "v3": step_3}
        action_json_0 = {"v1": 0, "v2": 0, "v3": 0}
        for i in range(self.action_interval):  # 10步
            res_str = self.proxy.connect.Step(json.dumps(action_json))
            if self.include_flow:
                step_state, vflow_x, vflow_y = self.parseStep(res_str, self.include_flow)  # 解析返回的状态
                # 展示时返回流场内容，训练时不返回
                v_x = np.array(vflow_x, dtype=np.float32)
                v_y = np.array(vflow_y, dtype=np.float32)
            else:
                step_state = self.parseStep(res_str, self.include_flow)  # 解析返回的状态
            state_1 = np.array(step_state, dtype=np.float32)  # 将状态转换为 NumPy 数组
        # step:info
        _info = {"vel_x": state_1[0], "vel_y": state_1[1], "vel_angle": state_1[2],
                 "pos_x": state_1[3], "pos_y": state_1[4], "angle": state_1[5],
                 "pressure_1": state_1[6], "pressure_2": state_1[7], "pressure_3": state_1[8],
                 "pressure_4": state_1[9], "pressure_5": state_1[10], "pressure_6": state_1[11],
                 "pressure_7": state_1[12], "pressure_8": state_1[13]}
        
        # step agent position
        self.agent_pos = [state_1[3], state_1[4]]
        self.agent_pos_history.append(self.agent_pos)
        new_pos = self.agent_pos
        d = np.linalg.norm(self.agent_pos - self.target_position)
        self.angle = state_1[5]
        self.vel_angle = abs(state_1[2]) # [1e-2, 1e-1]
        state_1[3], state_1[4] = self.target_position[0] - state_1[3], self.target_position[1] - state_1[4]
        self.pressure = state_1[6:14]

        # TODO:Without Pressure
        # self.state = state_1[:6]

        # Step Reward
        # Condition 1: Out of Bound
        if self.is_out_of_bounds():
            _truncated = True
            _reward = -500

            # TODO:HJB
            # _reward = 0
            print(colored("**** Episode Finished **** Hit Boundary.", 'red'))
        else:
            # Condition 2: Hit Circles
            if self.is_in_circle(self.circles):
                self._collision_count += 1

                # TODO:Version 2
                if self._collision_count > 5:
                    _truncated = True
                    _reward = 0
                    print(colored("**** Episode Finished **** Too Many Collisions.", 'red'))
                _reward = -100

                # TODO:HJB
                # _reward = -self.dt -10 * (d - self.d_2_target) - 10

                # _reward = -2 * self.dt -10 * (d - self.d_2_target) -100

                # TODO:Normalize
                # _reward = -self.dt - 100 * (d - self.d_2_target)/self.d_init - 10
                print(colored(f"**** Forbidden Move **** Hit Circles{int(self._collision_count)}.", 'yellow'))

        
            # Condition 3: Time Limit
            elif self.step_counter >= self.t_step_max or self.done:
                _truncated = True
                _reward = -0
                print(colored("**** Episode Finished **** Reaches Env Time limit.", 'blue'))
              
            else:
                # Condition 4: Reach Target
                if self.is_reach_target():
                    _terminated = True
                    _reward = 500
                    print(colored("**** Episode Finished **** SUCCESS.", 'green'))
                else:
                    # TODO:ICM
                    _reward = -2 * self.dt -10 * (d - self.d_2_target)
                    # TODO:HJB
                    # _reward = -self.dt -10 * (d - self.d_2_target)
                    # TODO:Normalize
                    # _reward = -self.dt -100 * (d - self.d_2_target)/self.d_init

                    # Condition 4: Check Roll 360
                    current_roll_count = self.angle / (2 * pi)
                    full_turns = int(current_roll_count) - int(self._roll_count)
                    if full_turns != 0:
                        print(colored(f"**** Over Roll:{int(self._roll_count)} to {int(current_roll_count)} ****", 'blue'))
                        # Roll Punish
                        _reward += abs(full_turns) * self._roll_punish  
                    self._roll_count = current_roll_count
        
        # Update
        self.d_2_target = d
        self.pre_pressure = self.pressure
        self.reward = _reward
        if self.include_flow:
            self.u_flow = v_x
            self.v_flow = v_y
            r_sensor = 8
            self.flow_info = self.get_flow_velocity(r_large=r_sensor)

            # print("Speed Shape:", len(self.flow_info))
            # print("Flow_speed_x:", v_x[int(self.agent_pos[0])][int(self.agent_pos[1])])
            # print("Agent_speed_x:", state_1[0])
            # print("Flow_speed_y:", v_y[int(self.agent_pos[0])][int(self.agent_pos[1])])
            # # (53, 32)
            # print("Flow_speed_circle:", sqrt(v_x[109][64]**2 +v_x[109][64]**2 ))
            # print("Agent_speed_y:", state_1[1])
            
            # # 打印 u_flow 和 v_flow 的尺寸
            # # print("u_flow shape:", np.shape(self.u_flow))
            # # print("v_flow shape:", np.shape(self.v_flow))
            # if np.isnan(v_x).any():
            #     print("Warning: v_x contains NaN values!")
            # if np.isnan(v_y).any():
            #     print("Warning: v_y contains NaN values!")
            # self._render_frame(_save=_terminated)

            # _flow_mean_x, _flow_mean_y = self.get_filtered_avg_velocity(r_large=r_sensor)
            # speed_info = np.array([_flow_mean_x,_flow_mean_y])
            # state_1 = np.concatenate([state_1, speed_info])
            # self.speed = sqrt(state_1[0]**2 + state_1[1]**2)
            # self.flow_speed = sqrt(v_x[int(self.agent_pos[0])][int(self.agent_pos[1])]**2 +v_y[int(self.agent_pos[0])][int(self.agent_pos[1])]**2)

        # if self.include_flow:
        #     self.u_flow = v_x
        #     self.v_flow = v_y
        #     self.speed = sqrt(state_1[0]**2 + state_1[1]**2)
        #     self.flow_speed = sqrt(v_x[int(self.agent_pos[0])][int(self.agent_pos[1])]**2 +v_y[int(self.agent_pos[0])][int(self.agent_pos[1])]**2)
        #     _save = _terminated or _truncated
        #     self._render_frame(_save=_save)
        self.state = state_1
        return self.state, self.reward, _terminated, _truncated, _info


    # @staticmethod
    # def _rand_in_circle(origin, r):
    #     d = sqrt(np.random.uniform(0, r ** 2))
    #     theta = np.random.uniform(0, 2 * pi)
    #     return origin + np.array([d * cos(theta), d * sin(theta)])

    def _rand_in_circle(self, origin, r):
        d = sqrt(np.random.uniform(0, r ** 2))
        theta = np.random.uniform(0, 2 * pi)
        return origin + np.array([d * cos(theta), d * sin(theta)])

    def _is_in_circles(self, point):
        for circle in self.circles:
            center = np.array(circle["center"])
            radius = 1.2 * circle["radius"]
            if np.linalg.norm(point - center) < radius:
                return True
        return False
    
    def generate_point_outside_circles(self, origin, r):
        while True:
            point = self._rand_in_circle(origin, r)
            if not self._is_in_circles(point):  # 如果点不在任何一个圆内
                return point

    def reset(self):
        # Step1: 起点和终点选取
        self._roll_count = 0
        self._collision_count = 0
        if not self._is_fixed_target:
            print("随机选择目标终点")


            # TODO:20250221扩大终点范围进行训练
            self.target_position = self.generate_point_outside_circles(self.target_default, 56)

            
            # TODO:20250209扩大终点范围进行训练
            # self.target_position = self._rand_in_circle(self.target_default, self.body_r)



            # self.target_position = self._rand_in_circle(self.target_default, self.body_r/2)
            # self.target_position = self.target_default
        
        if not self._is_fixed_start:
            print("随机选择目标起点")

            # TODO:20250221扩大终点范围进行训练
            self.start_position = self.generate_point_outside_circles(self.start_default, 56)

            # self.start_position = self._rand_in_circle(self.start_default,  2 * self.body_r)

            # self.start_position = self._rand_in_circle(self.start_default, self.body_r/2)
            # self.start_position = self.start_default
        

        # 以50%的概率交换起点和终点
        # if random.random() < 0.25:
        #     print("交换起点和终点")
        #     self.start_position, self.target_position = self.target_position, self.start_position
        
        print("target_pos:", self.target_position)
        print("start_pos:", self.start_position)
        self.step_counter=0


        # Step2:Reset true environment
        # action_json = {"v1": 0, "v2": 0, "v3": 0}
        agent_init_x = self.start_position[0]
        agent_init_y = self.start_position[1]
        action_json = {"v1": 0, "v2": 0, "v3": 0, "init_x": agent_init_x, "init_y": agent_init_y} # 传入一个坐标信息给Sever
        res_str = self.proxy.connect.reset(json.dumps(action_json))  # 调用 reset 获取新状态
        if self.include_flow:
            step_state, vflow_x, vflow_y = self.parseStep(res_str, self.include_flow)  # 解析返回的状态
            # 展示时返回流场内容，训练时不返回
            v_x = np.array(vflow_x, dtype=np.float32)  # 将状态转换为 NumPy 数组
            v_y = np.array(vflow_y, dtype=np.float32)  # 将状态转换为 NumPy 数组
        else:
            step_state = self.parseStep(res_str, self.include_flow)  # 解析返回的状态
        state_1 = np.array(step_state, dtype=np.float32)  # 将状态转换为 NumPy 数组
        _info = {"vel_x": state_1[0], "vel_y": state_1[1], "vel_angle": state_1[2],
                 "pos_x": state_1[3], "pos_y": state_1[4], "angle": state_1[5],
                 "pressure_1": state_1[6], "pressure_2": state_1[7], "pressure_3": state_1[8],
                 "pressure_4": state_1[9], "pressure_5": state_1[10], "pressure_6": state_1[11],
                 "pressure_7": state_1[12], "pressure_8": state_1[13]}
        # step agent position
        self.angle = state_1[5]
        self.agent_pos = [state_1[3], state_1[4]]
        d = np.linalg.norm(self.agent_pos - self.target_position)
        self.d_init = d
        self.d_2_target = d
        state_1[3], state_1[4] = self.target_position[0] - state_1[3], self.target_position[1] - state_1[4]
        self.pre_pressure = state_1[6:14]
        

        # TODO:对比没有压力的情况
        # self.state = state_1[0:6]
        
        if self.include_flow:
            self.u_flow = v_x
            self.v_flow = v_y
            r_sensor = 8
            self.flow_info = self.get_flow_velocity(r_large=r_sensor)
            # _flow_mean_x, _flow_mean_y = self.get_filtered_avg_velocity(r_large=r_sensor)
            # speed_info = np.array([_flow_mean_x,_flow_mean_y])
            # state_1 = np.concatenate([state_1, speed_info])
            self.speed = sqrt(state_1[0]**2 + state_1[1]**2)
            # self.flow_speed = sqrt(v_x[int(self.agent_pos[0])][int(self.agent_pos[1])]**2 +v_y[int(self.agent_pos[0])][int(self.agent_pos[1])]**2)
            # # self._render_frame()
            
            # print("Speed Shape:", len(self.flow_info))
            # print("Flow_speed_x:", v_x[int(self.agent_pos[0])][int(self.agent_pos[1])])
            # print("Agent_speed_x:", state_1[0])
            # print("Flow_speed_y:", v_y[int(self.agent_pos[0])][int(self.agent_pos[1])])
            # # (53, 32)
            # print("Flow_speed_circle:", sqrt(v_x[109][64]**2 +v_x[109][64]**2 ))
            # print("Agent_speed_y:", state_1[1])
            
            # # 打印 u_flow 和 v_flow 的尺寸
            # # print("u_flow shape:", np.shape(self.u_flow))
            # # print("v_flow shape:", np.shape(self.v_flow))
            # if np.isnan(v_x).any():
            #     print("Warning: v_x contains NaN values!")
            # if np.isnan(v_y).any():
            #     print("Warning: v_y contains NaN values!")
            # self._render_frame(_save=_terminated)
        
        self.state = state_1
        return self.state, _info
    
    def clip_action(self, action):
        # 使用 np.clip 裁剪动作，确保动作值在给定的范围内
        action = np.clip(action, self.low, self.high)
        return action

    def parseStep(self, info, _is_flow):  # 对lilypad返回信息解码
        all_info = json.loads(info)
        state = json.loads(all_info['state'][0])
        state_ls = [state['vel_x'], state['vel_y'], state['vel_angle'], state['pos_x'], state['pos_y'], state['angle'],
                    state['surfacePressures_1'], state['surfacePressures_2'], state['surfacePressures_3'],
                    state['surfacePressures_4'], state['surfacePressures_5'], state['surfacePressures_6'],
                    state['surfacePressures_7'], state['surfacePressures_8']]  # state
        if _is_flow:
            # TODO:是不是速度取反了
            flow_u = all_info.get('flow_u')  # 假设 flow_u 是一个二维数组
            flow_v = all_info.get('flow_v')  # 假设 flow_v 是一个二维数组
            return state_ls, flow_u, flow_v
        return state_ls


    def parseState(self, state):
        state = json.loads(json.loads(state)['state'][0])
        state['SparsePressure'] = list(map(float, state['SparsePressure'].split('_')))
        state_ls = [state['delta_y'], state['y_velocity'], state['eta'], state['delta_theta'], state['x_velocity'],
                    state['theta_velocity']] + state['SparsePressure']
        state_ls = list(np.nan_to_num(np.array(state_ls), nan=0))
        return state_ls
    
    def is_in_circle(self, circles):
        # 不严格的撞圆柱判断
        for circle in circles:
            center = np.array(circle["center"])
            radius = circle["radius"]
            
            a = self.ellipse_a  # 椭圆的半长轴
            b = self.ellipse_b
            theta = -self.state[5]

            distance = np.linalg.norm(self.agent_pos - center)

            # 距离小于长的半轴+半径,等效于设置了一个安全碰撞半径
            if distance <= (b + radius):
                return True
        return False
    
    def is_out_of_bounds(self):
        # 增加一个伪边界，防止仿真数据无法获取的情况
        _delta = 2
        # 检测椭圆是否完全出界。
        x_c, y_c = self.agent_pos  # 椭圆中心
        a = self.ellipse_a  # 椭圆的半长轴
        b = self.ellipse_b  # 椭圆的半短轴
        theta = - self.state[5]  # 椭圆的旋转角度（弧度）

        # 椭圆顶点（未旋转前）
        vertices = [(a, 0), (-a, 0),  # 长轴方向的两个点
                    (0, b), (0, -b)   # 短轴方向的两个点
                    ]

        # 旋转后的顶点坐标
        rotated_vertices = []
        for vx, vy in vertices:
            x_rot = x_c + vx * np.cos(theta) - vy * np.sin(theta)
            y_rot = y_c + vx * np.sin(theta) + vy * np.cos(theta)
            rotated_vertices.append((x_rot, y_rot))

        for x, y in rotated_vertices:
            if not (_delta <= x <= self.x_range-_delta and _delta  <= y <= self.y_range-_delta):
                print(f"Out of bounds:Agent_pos{[x_c,y_c]};Angle{theta}")
                print(f"Steps:",self.step_counter)
                return True
        return False

    
    def is_reach_target(self):
        # 检测当前椭圆是否覆盖目标点。
        x_c, y_c = self.agent_pos  # 椭圆的中心位置
        x_o, y_o = self.target_position  # 目标点位置

        # 获取椭圆参数
        theta = - self.state[5]  # 椭圆的旋转角度，单位：弧度
        a = self.ellipse_a  # 椭圆的半长轴
        b = self.ellipse_b  # 椭圆的半短轴

        # 将目标点变换到椭圆坐标系中
        dx = x_o - x_c
        dy = y_o - y_c
        x_prime = dx * np.cos(-theta) - dy * np.sin(-theta)
        y_prime = dx * np.sin(-theta) + dy * np.cos(-theta)

        # 判断目标点是否在椭圆内
        if (x_prime**2 / a**2 + y_prime**2 / b**2) <= 1:
            return True
        return False

    def show_info(self, info):
        # 显示state中的内容
        all_info = json.loads(info)
        state_json_str = all_info['state'][0]

        # 如果 state_json_str 是字符串，才进行解析
        if isinstance(state_json_str, str):
            state_dict = json.loads(state_json_str)
        elif isinstance(state_json_str, dict):
            state_dict = state_json_str  # 如果已经是字典，就直接使用
        else:
            raise TypeError(f"Expected string or dict, got {type(state_json_str)}")

        # 打印所有标签（字典的键）
        for label in state_dict.keys():
            print(label)

    def terminate(self):
        pid = os.getpgid(self.server.pid)
        self.server.terminate()
        os.killpg(pid, signal.SIGTERM)  # Send the signal to all the process groups.

    def close(self):
        if self.local_port == None:
            self.server.terminate()
    
    def _render_frame(self, _save=False):
        if len(self.agent_pos_history) == 0:
            return
        # 获取当前坐标轴
        ax = self.ax
        ax.clear()  # 清除当前坐标轴的内容

        # 调用 plot_env 绘制图形
        self.plot_env(ax)  # 传递 ax 给 plot_env 用于绘图
        # 刷新
        plt.draw()  
        # 暂停
        plt.pause(self.frame_pause)  # 暂停以更新图形
        if _save:
            save_path = './model_output/path.png'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保路径存在
            plt.savefig(save_path, bbox_inches='tight', dpi=300)  # 保存为高分辨率图像
            print(f"Image saved to {save_path}")


    def plot_env(self, ax, sample_rate=10):
        history = self.agent_pos_history
        start_point = history[0]
        target = self.target_position
        circles = self.circles
        agent_pos = self.agent_pos
        agent_angle = self.angle
        flow_x = self.v_flow
        flow_y = self.u_flow
        speed = self.speed
        flow_speed = self.flow_speed

        ax.clear()
        # 起点
        # flow_x中保存x的流速网格
        # flow_y中保存y的流速网格
        ax.scatter(*start_point, color='blue', label='Start Point', zorder=5)
        # 目标点
        ax.scatter(*target, color='green', label='Target Point', zorder=5)
        # 障碍物
        for circle in circles:
            ax.add_patch(
                plt.Circle(circle["center"], circle["radius"], color='red', alpha=0.3)
            )
        # 历史轨迹
        if history:
            hx, hy = zip(*history)
            ax.plot(hx, hy, linestyle='--', color='orange', label='Path')

        # 流场绘制（下采样）
        x = np.linspace(0, self.y_range, flow_x.shape[1])[::sample_rate]
        y = np.linspace(0, self.x_range, flow_x.shape[0])[::sample_rate]
        X, Y = np.meshgrid(x, y)
        sampled_flow_x = flow_x[::sample_rate, ::sample_rate]
        sampled_flow_y = flow_y[::sample_rate, ::sample_rate]
        # 调换 X 和 Y 坐标，交换流场分量
        ax.quiver(
            Y, X, sampled_flow_y * 2, sampled_flow_x * 2,  # 放大流场并交换流场方向
            scale=80, scale_units='width', color='black', alpha=0.5, zorder=2,
            width=0.002, pivot='middle',
            headwidth=3, headaxislength=3
        )


        # 当前智能体椭圆形状
        # TODO:确定角度的方向，是顺时针为正还是逆时针为正
        # TODO：确定椭圆的摆放模式
        ellipse_height = circle["radius"]  # 2a = 8
        ellipse_width = ellipse_height / 1.5  # a/b = 1.5/1
        # ellipse = Ellipse(
        #     xy=agent_pos, width=ellipse_width, height=ellipse_height, 
        #     angle=np.degrees(agent_angle), color='orange', alpha=0.7, zorder=4
        # )
        ellipse = Ellipse(
            xy=agent_pos, width=ellipse_height, height=ellipse_width, 
            angle=np.degrees(agent_angle), color='orange', alpha=0.7, zorder=4
        )
        ax.add_patch(ellipse)

        # 绘制局部坐标系（随智能体移动）
        # 坐标轴的长度
        axis_length = 10.0
        cos_angle = np.cos(agent_angle)
        sin_angle = np.sin(agent_angle)

        # 绘制x轴（红色）和y轴（绿色）
        ax.quiver(
            agent_pos[0], agent_pos[1], axis_length * cos_angle, axis_length * sin_angle,
            angles='xy', scale_units='xy', scale=1, width=0.002, headwidth=2, color='red', label="Agent_X"
        )
        ax.quiver(
            agent_pos[0], agent_pos[1], -axis_length * sin_angle, axis_length * cos_angle,
            angles='xy', scale_units='xy', scale=1, width=0.002, headwidth=2, color='green', label="Agent_Y"
        )
        # 设置坐标轴比例为相等
        # 确保坐标轴的比例相同，避免椭圆变形
        ax.set_aspect('equal', 'box')  
        
        # 在右下角显示智能体的坐标、速度和流场速度
        ax.text(
            0.95, 0.05, 
            (f"Agent (x, y): ({agent_pos[0]:.2f}, {agent_pos[1]:.2f})\n"
                f"Angle/°: {np.degrees(agent_angle):.2f}\n"
                f"Speed/m/s: {speed:.2f}\n"
                # f"Flow_Speed/m/s: {flow_speed:.2f}\n"
            ),
            transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
        )

        # 设置图形属性
        ax.set_xlim(0, self.x_range)
        ax.set_ylim(0, self.y_range)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        ax.set_title("Moving Trajectory")
