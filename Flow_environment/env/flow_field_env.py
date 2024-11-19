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
from termcolor import colored
from math import pi, sin, cos


def is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

class foil_env:
    def __init__(self, config=None, info='', local_port=None, network_port=None, target_position=None, max_step=None):
        # n = 20 * 16  m = 8 * 16 r = 3 * 16 每个圆柱的半径是1*16
        # m 和 n 对应模拟窗格的大小
        # 三角形排布 [n/6, n/6, n/6+r*cos(theta)] [m/2 + r/2, m/2 - r/2, m/2]
        # 智能体位置：[6*n/8 , m/2](240, 64)
        self.x_range = 20 * 16
        self.y_range = 8 * 16
        self.body_r = 1 * 16
        self.n = 20 * 16
        self.m = 8 * 16
        self.r = 3 * 16
        self.circles = [
            {"center": (self.n/6, self.m/2 + self.r/2), "radius": self.body_r},
            {"center": (self.n/6, self.m/2 - self.r/2), "radius": self.body_r},
            {"center": (self.n/6 + self.r*cos(pi/6), self.m/2), "radius": self.body_r}
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
        self.state = [0, 0, 0, 0, 0, 0]
        low = np.array([-10.0, -10, -2.0], dtype=np.float32)  # 每个维度的下界
        high = np.array([10.0, 10, 2.0], dtype=np.float32)  # 每个维度的上界
        self.agent_pos = np.array([0.0, 0.0])
        self.action_interval = config.action_interval
        self.dt = 0.075
        self.unwrapped = self
        self.unwrapped.spec = None
        self.observation_space = Box(low=-1e6, high=1e6, shape=[self.observation_dim])
        self.action_space = Box(low=low, high=high, shape=(self.action_dim,), dtype=np.float32)
        self.local_port = local_port
        if target_position:
            self.target_position = target_position
        else:
            self.target_position = self.target_default
        self.target_r = 0.5  # 到达终点判定的范围
        self.reward = 0
        self.done = False
        self.d_2_target = 0

        # TODO:start the server
        while True:
            port = random.randint(6000, 8000)
            if not is_port_in_use(port):
                break
        port = port if network_port == None else network_port
        # print("Foil_env start!")
        if local_port == None:
            self.server = subprocess.Popen(
                f'xvfb-run -a /home/zhengshizhan/workspace/processing-4.3/processing-java '
                f'--sketch=/home/zhengshizhan/project1/Navigation_understand-master/PinballCFD_server --run {port} '
                f'{info}',
                shell=True)
            # wait the server to start
            time.sleep(20)
            print("server start")
            self.proxy = ServerProxy(f"http://localhost:{port}/")
        else:
            self.proxy = ServerProxy(f"http://localhost:{local_port}/")

    def step(self, action):
        _truncated = False
        _terminated = False
        _reward = 0
        state_1 = self.state
        self.step_counter += 1
        print("#########################################################################################")
        print(f"steps:{self.step_counter}")
        pos_cur = self.agent_pos
        step_1 = float(action[0])  # x 加速度
        step_2 = float(action[1])  # y 加速度
        step_3 = float(action[2])  # 旋转加速度
        action_json = {"v1": step_1, "v2": step_2, "v3": step_3}
        for _ in range(self.action_interval):  # only 1
            res_str = self.proxy.connect.Step(json.dumps(action_json))
            step_state = self.parseStep(res_str)  # 解析返回的状态
            state_1 = np.array(step_state, dtype=np.float32)  # 将状态转换为 NumPy 数组
            """"
            res_str = self.proxy.connect.Step(json.dumps(action_json))
            # 将action传入lilypad并返回state, reward, 将一个Python数据结构转换为JSON
            [step_state, step_reward, step_done] = self.parseStep(res_str)
            reward_1, state_1, done_1 = (np.array(step_reward, dtype=np.float32), np.array(step_state, np.float32),
                                         np.array(step_done, np.float32))
            """
        # step:info
        _info = {"vel_x": state_1[0], "vel_y": state_1[1], "vel_angle": state_1[2],
                 "pos_x": state_1[3], "pos_y": state_1[4], "angle": state_1[5],
                 "pressure_1": state_1[6], "pressure_2": state_1[7], "pressure_3": state_1[8],
                 "pressure_4": state_1[9], "pressure_5": state_1[10], "pressure_6": state_1[11],
                 "pressure_7": state_1[12], "pressure_8": state_1[13]}
        # step agent position
        # self.done = done_1
        # print("simulation dones:",self.done)
        self.agent_pos = [state_1[3], state_1[4]]
        new_pos = self.agent_pos
        d = np.linalg.norm(self.agent_pos - self.target_position)
        # step state:[pos_x, pos_y] -> [dx, dy]
        state_1[3], state_1[4] = self.target_position[0] - state_1[3], self.target_position[1] - state_1[4]
        state_2 = state_1[:6]
        self.state = state_1
        # step reward
        # 环境反馈的出界标识
        if not (0 <= new_pos[0] <= self.x_range and 0 <= new_pos[1] <= self.y_range):
            _truncated = True
            _reward = -0
            print(colored("**** Episode Finished **** Hit Boundary.", 'red'))
        else:
            if self.is_in_circle(self.circles):
                _truncated = True
                _reward = -0
                print(colored("**** Episode Finished **** Hit Circles.", 'red'))
            # 检查超时
            elif self.step_counter >= self.t_step_max or self.done:
                _truncated = True
                _reward = -0
                print(colored("**** Episode Finished **** Reaches Env Time limit.", 'blue'))
            # 正常情况
            else:
                # 检查终点
                if d <= self.target_r * 2.5:
                    mid = 0.5 * (self.agent_pos + pos_cur)
                    d_mid = np.linalg.norm(mid - self.target_position)
                    if d_mid <= self.target_r or d <= self.target_r:
                        _terminated = True
                        _reward = 200
                        print(colored("**** Episode Finished **** SUCCESS.", 'green'))
                # 运行途中
                else:
                    _reward = -self.dt * 10 - (d - self.d_2_target)
        self.d_2_target = d
        self.reward = _reward
        
        return self.state, self.reward, _terminated, _truncated, _info

    def reset(self):
        self.step_counter=0
        # TODO: reset true environment
        action_json = {"v1": 0, "v2": 0, "v3": 0}
        res_str = self.proxy.connect.reset(json.dumps(action_json))  # 调用 reset 获取新状态
        self.show_info(res_str)  # 显示返回的状态信息
        step_state = self.parseStep(res_str)  # 解析返回的状态
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
        # step terminate
        # self.done = done_1
        return self.state, _info
    """"
    def parseStep(self, info):  # 对lilypad返回信息解码
        all_info = json.loads(info)
        state = json.loads(all_info['state'][0])
        reward = all_info['reward']
        done = all_info['done']
        state_ls = [state['vel_x'], state['vel_y'], state['vel_angle'], state['pos_x'], state['pos_y'], state['angle'],
                    state['surfacePressures_1'], state['surfacePressures_2'], state['surfacePressures_3'],
                    state['surfacePressures_4'], state['surfacePressures_5'], state['surfacePressures_6'],
                    state['surfacePressures_7'], state['surfacePressures_8']]  # state
        return state_ls, reward, done
    """

    def parseStep(self, info):  # 对lilypad返回信息解码
        all_info = json.loads(info)
        state = json.loads(all_info['state'][0])
        # TODO: ?
        state_ls = [state['vel_x'], state['vel_y'], state['vel_angle'], state['pos_x'], state['pos_y'], state['angle'],
                    state['surfacePressures_1'], state['surfacePressures_2'], state['surfacePressures_3'],
                    state['surfacePressures_4'], state['surfacePressures_5'], state['surfacePressures_6'],
                    state['surfacePressures_7'], state['surfacePressures_8']]  # state
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
        for circle in circles:
            center = np.array(circle["center"])
            radius = circle["radius"]
            distance = np.linalg.norm(self.agent_pos - center)
            if distance < radius:
                return True
        return False

    def show_info(self, info):
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

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    args.action_interval = 10

    env = foil_env(args)
    env.reset()
    print("target_init:", env.target_position)
    print("position_init:", env.agent_pos)

    for _ in range(5):
        # 执行动作并获取状态
        state, reward, terminate, truncate, info = env.step([5, 5, 1])
        print("position:", env.agent_pos)
        print(info)
        env.reset()
        print("position_re_init:", env.agent_pos)
    env.terminate()
"""