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


def is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


class foil_env:
    def __init__(self, config=None, info='', local_port=None, network_port=None, target_position=None,
                 start_position=None, max_step=None, include_flow=False):
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
        self.ellipse_b = self.body_r / 4  # 竖直摆放的椭圆 b = 1.5a
        self.ellipse_a = self.ellipse_b / 1.5
        self.circles = [
            {"center": (53, 32), "radius": self.body_r / 2},
            {"center": (53, 96), "radius": self.body_r / 2},
            {"center": (109, 64), "radius": self.body_r / 2}
        ]
        # self.observation_dim = 14
        self.observation_dim = 12  # 删除xy
        # self.observation_dim = 13  # θ改为sincos

        self.action_dim = 3
        self.step_counter = 0
        self.success_counter = 0
        self.steps_default = 300
        if max_step:
            self.t_step_max = max_step
        else:
            self.t_step_max = self.steps_default
        self.state = np.zeros(self.observation_dim)
        # TODO:change action range
        # 参考:[15, 15, 10]
        # low = np.array([-15.0, -15, -10], dtype=np.float32)  # 每个维度的下界
        # high = np.array([15.0, 15, 10], dtype=np.float32)  # 每个维度的上界
        low = np.array([-0.00001, -0.00001, -10], dtype=np.float32)  # 每个维度的下界
        high = np.array([0.00001, 0.00001, 10], dtype=np.float32)  # 每个维度的上界
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
        self.target_default = np.array([self.n / 6 + self.r * cos(pi / 6) / 2, self.m / 2])
        self.start_default = np.array([6 * self.n / 8, self.m / 2])
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
        ##############################################################

        # TODO:目标速度和角度
        self.tvx = 0
        self.tvy = 0
        self.tva = 0
        self.target_vx_range = [-2, 2]
        self.target_vy_range = [-2, 2]
        self.target_angle_range = [-pi / 4, pi / 4]
        self.reward = 0
        self.d_2_target = 0
        self.a_2_target = 0
        self.angle = 0
        self.u_flow = []
        self.v_flow = []
        self.speed = 0
        self.flow_speed = 0
        self.include_flow = include_flow

        # TODO:几个奖励的值的大小，用于测量和调整对应的系数k
        self.time_reward = 0
        self.angle_reward = 0
        self.vel_angle_reward = 0
        self.roll_punish = 0

        # 绘图
        self.frame_pause = 0.03  # 渲染间隔时间
        self.fig, self.ax = plt.subplots(figsize=(16, 9))  # 创建子图

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

    def step(self, action):
        # 低层网络测试
        # 动作裁剪
        action = self.clip_action(action)
        _truncated = False
        _terminated = False
        _reward = 0
        state_1 = self.state
        self.step_counter += 1
        target_vel_x = self.tvx
        target_vel_y = self.tvy
        target_angle = self.tva

        # 当前动作分解
        # step_1 = float(action[0])  # x 加速度
        # step_2 = float(action[1])  # y 加速度
        step_1 = float(0)  # 强制为0
        step_2 = float(0)  # 强制为0
        step_3 = float(action[2])  # 角加速度
        action_json = {"v1": step_1, "v2": step_2, "v3": step_3}

        # 环境步进
        for i in range(self.action_interval):  # 10步
            res_str = self.proxy.connect.Step(json.dumps(action_json))
            if self.include_flow:
                step_state, vflow_x, vflow_y = self.parseStep(res_str, self.include_flow)  # 解析返回的状态
                # 展示时返回流场内容，训练时不返回
                v_x = np.array(vflow_x, dtype=np.float32)  # 将状态转换为 NumPy 数组
                v_y = np.array(vflow_y, dtype=np.float32)  # 将状态转换为 NumPy 数组
            else:
                step_state = self.parseStep(res_str, self.include_flow)  # 解析返回的状态
            state_1 = np.array(step_state, dtype=np.float32)
        # step:info
        _info = {"vel_x": state_1[0], "vel_y": state_1[1], "vel_angle": state_1[2],
                 "pos_x": state_1[3], "pos_y": state_1[4], "angle": state_1[5],
                 "pressure_1": state_1[6], "pressure_2": state_1[7], "pressure_3": state_1[8],
                 "pressure_4": state_1[9], "pressure_5": state_1[10], "pressure_6": state_1[11],
                 "pressure_7": state_1[12], "pressure_8": state_1[13]}

        normal_angle = self.normalize_angle(state_1[5])
        # print("----------------", "step:", self.step_counter, "----------------------------------")
        # print(f'pos_x: {state_1[3]:.4f}, pos_y: {state_1[4]:.4f}')
        # print(f'vel_x: {state_1[0]:.4f}, vel_y: {state_1[1]:.4f}')
        # print('        ----')
        # print(f'angle: {normal_angle:.4f} , {np.degrees(normal_angle):.4f} in °')
        # print(f'vel_ang: {state_1[2]:.4f} , {np.degrees(state_1[2]):.4f} in °')
        # print('        ----')
        # print(f'Action given: ax {step_1:.4f}, ay {step_2:.4f}, aw {step_3:.4f} in rad.')

        # 更新状态
        self.agent_pos = [state_1[3], state_1[4]]

        self.agent_pos_history.append(self.agent_pos)

        # TODO: state_1[5]需要规范化到[-pi, pi]
        # self.angle = state_1[5]
        self.angle = normal_angle
        state_1[3], state_1[4] = self.target_position[0] - state_1[3], self.target_position[1] - state_1[4]
        delta_angle = self.normalize_angle(target_angle - state_1[5])  # 保证delta_angel有界

        # TODO: dθ改为了sincos，记得去前面 L46 更改状态空间维度，也记得去后面reset里 L406更改
        # TODO: 后续证实必要用sincos。
        # state_2 = np.concatenate((state_1[:3], [sin(delta_angle), cos(delta_angle)], state_1[6:]))
        state_2 = np.concatenate((state_1[:3], [delta_angle], state_1[6:]))

        self.state = state_2

        # # 奖励函数计算
        # current_vel_x, current_vel_y, current_angle = state_1[0], state_1[1], state_1[5]
        # current_accel = np.linalg.norm([step_1, step_2]) # [15, 15]
        # current_accel_normalized = current_accel/np.linalg.norm([15, 15]) # [15, 15]
        # current_angular_accel = abs(step_3) # 10
        # current_angular_accel_normalized = current_angular_accel/10

        # # 理想速度和角度的偏差
        # vel_x_error = abs(target_vel_x - current_vel_x)
        # vel_y_error = abs(target_vel_y - current_vel_y)

        # current_angle = state_1[5]
        # self.roll_punish = 0
        # _roll_punish = 0
        # if abs(current_angle) > 2 * pi:
        #     _roll_punish = 10 * (abs(current_angle) - 2 * pi)  # 旋转超过一圈的惩罚
        #     self.roll_punish = _roll_punish

        old_angle_2_target = self.a_2_target
        old_angle_error = abs(self.a_2_target)  # error是加了绝对值的
        angle_error = abs(delta_angle)
        self.a_2_target = delta_angle  # 这个里存的是带符号的
        vel_angle = abs(state_1[2])

        # 检查是否达到目标速度和角度
        # vel_threshold = 0.1  # 速度误差阈值
        angle_threshold = 0.02  # 角度误差阈值
        success_threshold = 10  # 目标保持成功计数阈值
        vel_angle_threshold = 0.01  # 角速度误差阈值

        # TODO:保证时间惩罚和距离奖励在同一个量级
        kt = 10  # dt = 0.075 150 * 0.75 = 112.5
        ka_1 = 200  # [1e-2, 1e-1] pi/4 * 200 = 157
        kv_1 = 1  # [1e-2, 1e-1],真实的速度量级应该再乘一个16
        ka_2 = 100
        kv_2 = 10
        self.time_reward = self.dt
        self.angle_reward = (old_angle_error - angle_error)
        self.vel_angle_reward = vel_angle

        _reward = 0

        if self.is_out_of_bounds():
            _truncated = True
            _reward = 0  # 边界惩罚
            print(colored(f"**** Episode Finished At Step {self.step_counter} **** Hit Boundary.", 'red'))
        elif self.step_counter >= self.t_step_max:
            _truncated = True
            _reward = 0  # 时间限制惩罚
            print(colored("**** Episode Finished **** Reaches Env Time limit.", 'blue'))

        # 未结束
        else:
            # 根据角度误差计算的基本奖励
            # _reward_basic = - 10 * self.dt + 200 * (old_angle_error - angle_error)
            _reward_basic = - 10 * self.dt - 5 * angle_error
            # _reward_basic = - 10 * self.dt - 1 * angle_error
            # _reward_basic = - 1 * angle_error
            # _reward_basic = - 10 * self.dt - 2 * sqrt(angle_error)
            # _reward_basic = - 2 * sqrt(angle_error)
            # _reward_basic = - 10 * self.dt - 2.5 * (angle_error ** 0.3)

            # 角度达到要求的奖励
            _reward_angel_good = 10
            # 角速度进一步达到要求的奖励
            _reward_vel_good = 0
            # 转一整圈的惩罚
            _reward_full_circle = -10

            # 角度到达
            if angle_error < angle_threshold:
                if vel_angle < vel_angle_threshold:
                    _reward += _reward_angel_good
                    _reward += _reward_basic  # 额外加基础reward
                    print(colored(f"----Stabled.", 'green'))
                else:
                    _reward += _reward_angel_good
                    _reward += _reward_basic  # 额外加基础reward
                    print(colored(f"----Only angle matched.", 'yellow'))
            # 角度没到
            else:
                # 两步角度插值就算转成陀螺也不会超过1，30°也就是0.5rad
                # self.dt = 0.075，什么都不干150步冲出去，慢的角度跟踪40步跟踪上
                _reward = _reward_basic

                # 转大圈圈了
                if ((old_angle_2_target > 0 > self.a_2_target and state_1[2] < 0)
                        or (old_angle_2_target < 0 < self.a_2_target and state_1[2] > 0)):
                    _reward += _reward_full_circle
                    print(colored(f"----reach full circle.", 'red'))

        print(f'---step{self.step_counter}: ang {normal_angle:.4f}, ang2target {delta_angle:.4f}, '
              f'vel_ang {state_1[2]:.4f}, aw {step_3:.4f}, rwd {_reward:.2f}')

        # else:
        #     if angle_error < angle_threshold and vel_angle < vel_angle_threshold:
        #         # 达到目标并保持稳定
        #         _terminated = True
        #         _reward = 200
        #         print(colored(f"**** Reach Target **** SUCCESS.", 'green'))
        #     elif angle_error < angle_threshold:
        #         # 接近目标角度但角速度不稳定
        #         # _reward = 10 + ka_2 * self.angle_reward - kv_2 * self.vel_angle_reward
        #         # _reward = - kt * self.dt + ka_1 * (old_angle_error - self.a_2_target) - kv_1 * vel_angle - _roll_punish
        #         _reward = - kt * self.dt + ka_1 * (old_angle_error - self.a_2_target) - kv_1 * vel_angle
        #         print(colored(f"**** Near Target At Step {self.step_counter} **** Adjusting Velocity.", 'yellow'))
        #     else:
        #         # 未达到目标角度
        #         # 角度误差用两次的差值
        #         # _reward = - kt * self.dt + ka_1 * (old_angle_error - self.a_2_target) - kv_1 * vel_angle - _roll_punish
        #         _reward = - kt * self.dt + ka_1 * (old_angle_error - self.a_2_target) - kv_1 * vel_angle
        #         # print(colored(f"****dt:{self.dt} angle_loss:{old_angle_error - self.a_2_target} vel_angle:{vel_angle} ****", 'blue'))

        # 环境记录更新
        self.reward = _reward
        if self.include_flow:
            self.u_flow = v_x
            self.v_flow = v_y
            self.speed = sqrt(state_1[0] ** 2 + state_1[1] ** 2)
            self.flow_speed = sqrt(
                v_x[int(self.agent_pos[0])][int(self.agent_pos[1])] ** 2 + v_y[int(self.agent_pos[0])][
                    int(self.agent_pos[1])] ** 2)
            # print(f"Agent_Angle{self.angle}")
            self._render_frame(_save=_terminated)
        return self.state, self.reward, _terminated, _truncated, _info

    @staticmethod
    def _rand_in_circle(origin, r):
        d = sqrt(np.random.uniform(0, r ** 2))
        theta = np.random.uniform(0, 2 * pi)
        return origin + np.array([d * cos(theta), d * sin(theta)])

    def reset(self):
        # 目标点和起点是否随机
        # if not self._is_fixed_target:
        #     print("随机选择目标终点")
        #     self.target_position = self._rand_in_circle(self.target_default, self.body_r / 2)  # 随机目标点
        # else:
        #     self.target_position = self.target_default  # 固定目标点
        # print("target_pos:", self.target_position)

        # if not self._is_fixed_start:
        #     print("随机选择目标起点")
        #     self.start_position = self._rand_in_circle(self.start_default, self.body_r / 2)  # 随机起点
        # else:
        #     self.start_position = self.start_default  # 固定起点
        print("start_pos:", self.start_position)

        self.tvx = np.random.uniform(self.target_vx_range[0], self.target_vx_range[1])
        self.tvy = np.random.uniform(self.target_vy_range[0], self.target_vy_range[1])
        # self.tva = np.random.uniform(self.target_angle_range[0], self.target_angle_range[1])
        self.tva = pi / 6
        target_angle = self.tva

        # print(f"目标速度：x{self.tvx},y{self.tvy}")
        print(f"目标角度：{self.tva}")

        self.step_counter = 0

        # 初始化智能体位置并重置环境
        agent_init_x = self.start_position[0]
        agent_init_y = self.start_position[1]
        action_json = {"v1": 0, "v2": 0, "v3": 0, "init_x": agent_init_x, "init_y": agent_init_y}  # 包含坐标信息
        res_str = self.proxy.connect.reset(json.dumps(action_json))  # 调用 reset 获取新状态

        # 解析返回的环境状态
        if self.include_flow:
            step_state, vflow_x, vflow_y = self.parseStep(res_str, self.include_flow)  # 解析流场状态
            v_x = np.array(vflow_x, dtype=np.float32)
            v_y = np.array(vflow_y, dtype=np.float32)
        else:
            step_state = self.parseStep(res_str, self.include_flow)

        state_1 = np.array(step_state, dtype=np.float32)  # 转换为 NumPy 数组

        # 提取关键信息
        _info = {
            "vel_x": state_1[0], "vel_y": state_1[1], "vel_angle": state_1[2],
            "pos_x": state_1[3], "pos_y": state_1[4], "angle": state_1[5],
            "pressure_1": state_1[6], "pressure_2": state_1[7], "pressure_3": state_1[8],
            "pressure_4": state_1[9], "pressure_5": state_1[10], "pressure_6": state_1[11],
            "pressure_7": state_1[12], "pressure_8": state_1[13]
        }

        # 更新智能体位置和与目标点的距离
        self.agent_pos = [state_1[3], state_1[4]]
        self.d_2_target = np.linalg.norm(self.agent_pos - self.target_position)

        # 更新相对目标点的位置
        self.agent_pos_history.append(self.agent_pos)
        self.angle = state_1[5]
        state_1[3], state_1[4] = self.target_position[0] - state_1[3], self.target_position[1] - state_1[4]
        delta_angle = self.normalize_angle(target_angle - state_1[5])  # 保证delta_angel有界
        # TODO:角度差值
        self.a_2_target = delta_angle

        # TODO: 记得去step里和最前面L46改
        # state_2 = np.concatenate((state_1[:3], [sin(delta_angle), cos(delta_angle)], state_1[6:]))
        state_2 = np.concatenate((state_1[:3], [delta_angle], state_1[6:]))

        self.state = state_2

        # 流场信息处理
        if self.include_flow:
            self.u_flow = v_x
            self.v_flow = v_y
            self.speed = sqrt(state_1[0] ** 2 + state_1[1] ** 2)
            self.flow_speed = sqrt(
                v_x[int(self.agent_pos[0])][int(self.agent_pos[1])] ** 2 + v_y[int(self.agent_pos[0])][
                    int(self.agent_pos[1])] ** 2)
            self._render_frame()

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
        # TODO: ?
        state_ls = [state['delta_y'], state['y_velocity'], state['eta'], state['delta_theta'], state['x_velocity'],
                    state['theta_velocity']] + state['SparsePressure']
        state_ls = list(np.nan_to_num(np.array(state_ls), nan=0))
        return state_ls

    def is_in_circle(self, circles):
        # 不严格的撞涡判断
        for circle in circles:
            center = np.array(circle["center"])
            radius = circle["radius"]

            a = self.ellipse_a  # 椭圆的半长轴
            b = self.ellipse_b
            theta = -self.state[5]

            distance = np.linalg.norm(self.agent_pos - center)

            # 距离小于长的半轴+半径，及认为发生碰撞
            # TODO:把撞涡的判定范围改宽一点
            if distance <= (b + radius):
                return True
        return False

    def is_out_of_bounds(self):
        # 检测椭圆是否完全出界。
        x_c, y_c = self.agent_pos  # 椭圆中心
        a = self.ellipse_a  # 椭圆的半长轴
        b = self.ellipse_b  # 椭圆的半短轴
        theta = - self.state[5]  # 椭圆的旋转角度（弧度）

        # 椭圆顶点（未旋转前）
        vertices = [(a, 0), (-a, 0),  # 长轴方向的两个点
                    (0, b), (0, -b)  # 短轴方向的两个点
                    ]

        # 旋转后的顶点坐标
        rotated_vertices = []
        for vx, vy in vertices:
            x_rot = x_c + vx * np.cos(theta) - vy * np.sin(theta)
            y_rot = y_c + vx * np.sin(theta) + vy * np.cos(theta)
            rotated_vertices.append((x_rot, y_rot))

        for x, y in rotated_vertices:
            if not (0 <= x <= self.x_range and 0 <= y <= self.y_range):
                print(f"Out of bounds:Agent_pos{[x_c, y_c]};Angle{theta}")
                print(f"Steps:", self.step_counter)
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
        if (x_prime ** 2 / a ** 2 + y_prime ** 2 / b ** 2) <= 1:
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
        # os.killpg(pid, signal.SIGTERM)  # Send the signal to all the process groups.

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
        ellipse = Ellipse(
            xy=agent_pos, width=ellipse_width, height=ellipse_height,
            angle=np.degrees(agent_angle), color='orange', alpha=0.7, zorder=4
        )
        ax.add_patch(ellipse)

        # 绘制局部坐标系（随智能体移动）
        # TODO：确定角度的方向，是顺时针为正还是逆时针为正
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
