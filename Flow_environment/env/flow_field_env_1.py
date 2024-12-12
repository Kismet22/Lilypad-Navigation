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
from math import pi, sin, cos


def is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

class foil_env:
    def __init__(self, config=None, info='', local_port=None, network_port=None, target_position=None, max_step=None, include_flow=False):
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
        self.observation_dim = 14
        self.action_dim = 3
        self.step_counter = 0
        self.steps_default = 300
        self.target_default = np.array([self.n/6 + self.r*cos(pi/6)/2, self.m/2])
        if max_step:
            self.t_step_max = max_step
        else:
            self.t_step_max = self.steps_default
        self.state = np.zeros(self.observation_dim)
        # TODO:change action range
        low = np.array([-15.0, -8, -0.0001], dtype=np.float32)  # 每个维度的下界
        high = np.array([15.0, 8, 0.0001], dtype=np.float32)  # 每个维度的上界
        # low = np.array([-20.0, -0.0001, -0.0001], dtype=np.float32)  # 每个维度的下界
        # high = np.array([20.0, 0.0001, 0.0001], dtype=np.float32)  # 每个维度的上界
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
        if target_position is not None:
            self.target_position = target_position
        else:
            self.target_position = self.target_default
        self.target_r = 0.5  # 到达终点判定的范围
        self.reward = 0
        self.done = False
        self.d_2_target = 0
        # 速度相关记录
        self.angle = 0
        self.u_flow = []
        self.v_flow = []
        self.speed = 0
        self.flow_speed = 0
        self.include_flow = include_flow

        # 绘图
        self.frame_pause = 0.03 # 渲染间隔时间
        self.fig, self.ax = plt.subplots(figsize=(16, 9)) # 创建子图

        # TODO:start the server
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


    def step(self, action):
        # 动作裁剪
        action = self.clip_action(action)
        _truncated = False
        _terminated = False
        _reward = 0
        state_1 = self.state
        self.step_counter += 1
        # print("#########################################################################################")
        # print(f"steps:{self.step_counter}")
        pos_cur = self.agent_pos
        step_1 = float(action[0])  # x 加速度
        step_2 = float(action[1])  # y 加速度
        step_3 = float(action[2])  # 旋转加速度
        action_json = {"v1": step_1, "v2": step_2, "v3": step_3}
        for _ in range(self.action_interval):  # only 1
            res_str = self.proxy.connect.Step(json.dumps(action_json))
            if self.include_flow:
                step_state, vflow_x, vflow_y = self.parseStep(res_str, self.include_flow)  # 解析返回的状态
                # TODO:尝试调整Sever中的代码内容，让展示时返回流场内容，训练时不返回
                v_x = np.array(vflow_x, dtype=np.float32)  # 将状态转换为 NumPy 数组
                v_y = np.array(vflow_y, dtype=np.float32)  # 将状态转换为 NumPy 数组
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
        self.angle = state_1[5]
        self.agent_pos_history.append(self.agent_pos)
        new_pos = self.agent_pos
        d = np.linalg.norm(self.agent_pos - self.target_position)
        # step state:[pos_x, pos_y] -> [dx, dy]
        state_1[3], state_1[4] = self.target_position[0] - state_1[3], self.target_position[1] - state_1[4]
        state_2 = state_1[:6]
        # 完整的state
        self.state = state_1
        # 二自由度运动时,仅保留位置信息的状态
        # self.state_pos = [state_1[0], state_1[1], state_1[3], state_1[4]]
        # step reward
        # 检查出界
        if self.is_out_of_bounds():
            _truncated = True
            # TODO:出界损失的选择
            # _reward = -200
            _reward = -0
            print(colored("**** Episode Finished **** Hit Boundary.", 'red'))
        else:
            # 检查撞涡
            if self.is_in_circle(self.circles):
                _truncated = True
                # TODO:撞涡损失的选择
                _reward = -1200
                print(colored("**** Episode Finished **** Hit Circles.", 'red'))
            # 检查超时
            elif self.step_counter >= self.t_step_max or self.done:
                _truncated = True
                # TODO:超时损失的选择
                _reward = -0
                print(colored("**** Episode Finished **** Reaches Env Time limit.", 'blue'))
            # 正常情况
            else:
                # 检查终点
                if self.is_reach_target():
                    _terminated = True
                    _reward = 5000
                    print(colored("**** Episode Finished **** SUCCESS.", 'green'))
                # 运行途中
                else:
                    # 4000 times_step r = -300
                    # d = 150到200之间
                    # 需要给两个参数
                    # TODO:删除时间惩罚
                    # _reward = -self.dt - 10 * (d - self.d_2_target)
                    _reward = -10 * (d - self.d_2_target)

        # print(f"Check_reward:t{self.dt * 10};distance{-d + self.d_2_target}")
        # 环境记录以及更新
        self.d_2_target = d
        self.reward = _reward
        if self.include_flow:
            self.u_flow = v_x
            self.v_flow = v_y
            self.speed = np.sqrt(state_1[0]**2 + state_1[1]**2)
            self.flow_speed = np.sqrt(v_x[int(self.agent_pos[0])][int(self.agent_pos[1])]**2 +v_y[int(self.agent_pos[0])][int(self.agent_pos[1])]**2)
            self._render_frame()

        return self.state, self.reward, _terminated, _truncated, _info

    def reset(self):
        print("target_pos:", self.target_position)
        self.step_counter=0
        # reset true environment
        action_json = {"v1": 0, "v2": 0, "v3": 0}
        res_str = self.proxy.connect.reset(json.dumps(action_json))  # 调用 reset 获取新状态
        # self.show_info(res_str)  # 显示返回的状态信息
        if self.include_flow:
            step_state, vflow_x, vflow_y = self.parseStep(res_str, self.include_flow)  # 解析返回的状态
            # TODO:尝试调整Sever中的代码内容，让展示时返回流场内容，训练时不返回
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
        self.agent_pos = [state_1[3], state_1[4]]
        d = np.linalg.norm(self.agent_pos - self.target_position)
        self.d_2_target = d
        # step state:[pos_x, pos_y] -> [dx, dy]
        state_1[3], state_1[4] = self.target_position[0] - state_1[3], self.target_position[1] - state_1[4]
        self.state = state_1
        if self.include_flow:
            self.u_flow = v_x
            self.v_flow = v_y
            self.speed = np.sqrt(state_1[0]**2 + state_1[1]**2)
            self.flow_speed = np.sqrt(v_x[int(self.agent_pos[0])][int(self.agent_pos[1])]**2 +v_y[int(self.agent_pos[0])][int(self.agent_pos[1])]**2)
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
                    (0, b), (0, -b)   # 短轴方向的两个点
                    ]

        # 旋转后的顶点坐标
        rotated_vertices = []
        for vx, vy in vertices:
            x_rot = x_c + vx * np.cos(theta) - vy * np.sin(theta)
            y_rot = y_c + vx * np.sin(theta) + vy * np.cos(theta)
            rotated_vertices.append((x_rot, y_rot))

        for x, y in rotated_vertices:
            if not (0 <= x <= self.x_range and 0 <= y <= self.y_range):
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
    
    def _render_frame(self):
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
            angle=-np.degrees(agent_angle), color='orange', alpha=0.7, zorder=4
        )
        ax.add_patch(ellipse)

        # 绘制局部坐标系（随智能体移动）
        # TODO：确定角度的方向，是顺时针为正还是逆时针为正
        # 坐标轴的长度
        axis_length = 10.0
        cos_angle = np.cos(-agent_angle)
        sin_angle = np.sin(-agent_angle)

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

