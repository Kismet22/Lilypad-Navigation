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
import matplotlib.colors as mcolors


# Port Check
def is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

class foil_env:
    def __init__(self, config=None, info='', local_port=None, network_port=None,
             target_center=None, target_position=None, start_center=None, 
             start_position=None, max_step=None, _include_flow=False, 
             _plot_flow=False, _proccess_flow=False, _random_range = 56, _flow_range=32, _init_flow_num=0, _init_ax=-8):
        """
        Basic Info:
        - Expansion Factor: 16
        - Window Range: 320 * 128
        - Real Range: 20 * 8
        - Ellipse Agent: a = 1.5 * b; a = 4
        - Cylinder(3): r = 8
        """
        #################################################
        # Basic Info
        self.window_r = 1 * 16
        self.x_range = 20 * self.window_r
        self.y_range = 8 * self.window_r
        self.ellipse_a = self.window_r/4
        self.ellipse_b = self.ellipse_a / 1.5
        self.circles = [
            {"center": (53, 32), "radius": self.window_r/2},
            {"center": (53, 96), "radius": self.window_r/2},
            {"center": (109, 64), "radius": self.window_r/2}
        ]
        self.flow_num = _init_flow_num
        #################################################

        ###################################################
        # State Space
        self.include_flow = _include_flow
        self.observation_dim = 14
        self.state = np.zeros(self.observation_dim)
        self.observation_space = Box(low=-1e6, high=1e6, shape=[self.observation_dim])

        # Action Space
        self.action_dim = 1
        low = np.array([-10.0], dtype=np.float32)
        high = np.array([10.0], dtype=np.float32)
        self.low = low
        self.high = high
        self.action_space = Box(low=low, high=high, shape=(self.action_dim,), dtype=np.float32)
        self.init_ax = _init_ax

        # Sim Time
        self.step_counter = 0
        self.steps_default = 300
        self.dt = 0.075
        if max_step:
            self.t_step_max = max_step
        else:
            self.t_step_max = self.steps_default
        
        # Reward
        self.reward = 0

        # Done
        self.done = False
        ###################################################

        #####################################################
        # Port Settings
        self.action_interval = config.action_interval
        self.unwrapped = self
        self.unwrapped.spec = None
        self.local_port = local_port
        #####################################################
        

        #######################################################
        # Sim Settings
        self.agent_pos = np.array([0.0, 0.0])
        # Reach Range
        self.target_r = 0.5
        self.random_range = _random_range
        self.target_default = np.array([240, 104])
        # Target Range Center
        if target_center is not None:
            self.target_default = target_center
        self.start_default = np.array([240, 24])
        # Start Range Center
        if start_center is not None:
            self.start_default = start_center
        self.target_position = self.target_default
        self.start_position = self.start_default
        self._is_fixed_target = True
        self._is_fixed_start = True
        # Input Target
        if target_position is not None:
            self.target_default = target_position
            self.target_position = target_position
        else:
            # Random Select
            self._is_fixed_target = False
        # Input Start
        if start_position is not None:
            self.start_default = start_position
            self.start_position = start_position
        else:
            # Random Select
            self._is_fixed_start = False
        # Distance to Target
        self.d_2_target = 0
        self._roll_count = 0
        self._collision_count = 0
        self._roll_punish = -20
        # self.d_init = 10
        # self._old_dis_2_barricade = 24
        #######################################################

        

        #########################################################
        # State Record
        # Agent Info
        self.agent_pos_history = []
        self.angle = 0
        self.vel_angle = 0
        self.speed = 0
        self.pressure = []
        self.pre_pressure = []
        # Flow Info
        self.u_flow = []
        self.v_flow = []
        self._proccess_flow = _proccess_flow
        self.flow_speed = 0
        if self._proccess_flow:
            self.flow_range = flow_range
            self.last_flow_x = np.zeros((flow_range, flow_range))
            self.last_flow_y = np.zeros((flow_range, flow_range))
        #########################################################

        
        ###########################################################
        # Plot Settings
        self._plot_flow = _plot_flow
        self.frame_pause = 0.03
        self.fig, self.ax = plt.subplots(figsize=(16, 9))
        ###########################################################

        #############################################################
        # Port Settings
        flow_flag = 'true' if self.include_flow else 'false'
        while True:
            port = random.randint(6000, 8000)
            if not is_port_in_use(port):
                break
        port = port if network_port == None else network_port
        if local_port == None:
            command = (
                f'xvfb-run -a /home/zhengshizhan/workspace/processing-4.3/processing-java '
                f'--sketch=/home/zhengshizhan/project1/Navigation_understand-master/PinballCFD_server --run '
                f'{port} {flow_flag} {info}'
            )
            # Using subprocess to start
            self.server = subprocess.Popen(command, shell=True)
            # wait the server to start
            time.sleep(20)
            print("server start")
            self.proxy = ServerProxy(f"http://localhost:{port}/")
        else:
            self.proxy = ServerProxy(f"http://localhost:{local_port}/")
        #############################################################


    def get_flow_velocity(self):
        """
        Compute the velocity field around the agent and store it in 16x16 grids.

        Parameters:
        r_large : Half-length of the larger square sensing range

        Returns:
        v_x_grid, v_y_grid : Two numpy arrays where the small square region is filled with 0.
        """
        v_x = self.u_flow
        v_y = self.v_flow
        x, y = self.agent_pos
        # Mask Range
        a = int(self.ellipse_a)
        r_large = int(self.flow_range)
        _cut_range = 2 * r_large
        # Flow Range
        H, W = v_x.shape

        # Checking NaN
        if np.isnan(x) or np.isnan(y):
            print(f"Warning: Agent position contains NaN (x={x}, y={y}).")
            return self.last_flow_x, self.last_flow_y

        # Checking Abnormal Position
        if x < -r_large or y < -r_large or x >= H + r_large or y >= W + r_large:
            print(f"Warning: Agent position out of bounds (x={x}, y={y}).")
            return self.last_flow_x, self.last_flow_y

        # Safety_margin
        safety_margin = 4
        pad_width = r_large + safety_margin

        # Mirror Padding
        v_x_padded = np.pad(v_x, pad_width=pad_width, mode='reflect')
        v_y_padded = np.pad(v_y, pad_width=pad_width, mode='reflect')
        x_padded = int(x + pad_width)
        y_padded = int(y + pad_width)

        v_x_region = v_x_padded[x_padded - r_large:x_padded + r_large + 1, 
                                y_padded - r_large:y_padded + r_large + 1]
        v_y_region = v_y_padded[x_padded - r_large:x_padded + r_large + 1, 
                                y_padded - r_large:y_padded + r_large + 1]
        mask = np.ones_like(v_x_region, dtype=bool)
        mask[r_large - a:r_large + a + 1, r_large - a:r_large + a + 1] = False
        v_x_region[~mask] = 0
        v_y_region[~mask] = 0
        return v_x_region[:_cut_range, :_cut_range], v_y_region[:_cut_range, :_cut_range]


    def clip_action(self, action):
        action = np.clip(action, self.low, self.high)
        return action

    def step(self, action):
        # Clip action
        action = self.clip_action(action)
        _truncated = False
        _terminated = False
        _reward = 0
        state_1 = self.state
        self.step_counter += 1
        pos_cur = self.agent_pos
        # action dim=1
        step_1 = float(action[0])

        # action_json = {"v1": -10, "v2": 0, "v3": step_1}

        action_json = {"v1": self.init_ax, "v2": 0, "v3": step_1}

        action_json_0 = {"v1": 0, "v2": 0, "v3": 0}
        # action_interbal = 10
        for i in range(self.action_interval):
            res_str = self.proxy.connect.Step(json.dumps(action_json))
            if self.include_flow:
                step_state, vflow_x, vflow_y = self.parseStep(res_str, self.include_flow)
                v_x = np.array(vflow_x, dtype=np.float32)
                v_y = np.array(vflow_y, dtype=np.float32)
            else:
                step_state = self.parseStep(res_str, self.include_flow)
            state_1 = np.array(step_state, dtype=np.float32)
        _info = {"vel_x": state_1[0], "vel_y": state_1[1], "vel_angle": state_1[2],
                 "pos_x": state_1[3], "pos_y": state_1[4], "angle": state_1[5],
                 "pressure_1": state_1[6], "pressure_2": state_1[7], "pressure_3": state_1[8],
                 "pressure_4": state_1[9], "pressure_5": state_1[10], "pressure_6": state_1[11],
                 "pressure_7": state_1[12], "pressure_8": state_1[13]}
        self.agent_pos = [state_1[3], state_1[4]]

        ########################################################################################
        # # Distance to Obstacle
        # distances = [np.linalg.norm(np.array(self.agent_pos) - np.array(circle["center"])) for circle in self.circles]
        # # Nearest Obstacle
        # nearest_idx = np.argmin(distances)
        # nearest_circle = self.circles[nearest_idx]
        # dx, dy = np.array(nearest_circle["center"]) - np.array(self.agent_pos)
        # _dis_2_barricade = distances[nearest_idx]
        # self._old_dis_2_barricade = _dis_2_barricade
        
        # # All distance
        # relative_positions = []
        # min_distance = float('inf')
        # for circle in self.circles[:3]:
        #     dx, dy = np.array(circle["center"]) - np.array(self.agent_pos)
        #     relative_positions.extend([dx, dy])
        #     distance = np.linalg.norm([dx, dy])
        #     min_distance = min(min_distance, distance)
        # _dis_2_barricade = min_distance 
        # relative_positions = np.array(relative_positions, dtype=np.float32)
        # self._old_dis_2_barricade = _dis_2_barricade
        ########################################################################################

        d = np.linalg.norm(self.agent_pos - self.target_position)
        self.angle = state_1[5]
        self.vel_angle = abs(state_1[2]) # [1e-2, 1e-1]
        state_1[3], state_1[4] = self.target_position[0] - state_1[3], self.target_position[1] - state_1[4]
        self.pressure = state_1[6:14]

        # TODO:dim=14
        self.state = state_1
        # TODO:dim=16
        # self.state = np.hstack([state_1, np.array([dx, dy])]).astype(np.float32)
        # TODO:dim=20
        # self.state = np.hstack([state_1, relative_positions]).astype(np.float32)

        # Step Reward
        if self.is_out_of_bounds():
            _truncated = True
            # _reward = -500
            _reward = 0
            print(colored("**** Episode Finished **** Hit Boundary.", 'red'))
        else:
            if self.is_in_circle(self.circles):
                self._collision_count += 1
                # _reward = -2 * self.dt - 10 * (d - self.d_2_target) - 55 * self._collision_count
                if self._collision_count > 2:
                    _truncated = True
                    # _reward = -100
                    _reward = 0
                    print(colored("**** Episode Finished **** Too Many Collisions.", 'red'))
                else:
                    _reward = -55 * self._collision_count
                print(colored(f"**** Forbidden Move **** Hit Circles{int(self._collision_count)}.", 'yellow'))
            elif self.step_counter >= self.t_step_max or self.done:
                _truncated = True
                # _reward = -50
                _reward = 0
                print(colored("**** Episode Finished **** Reaches Env Time limit.", 'blue'))
            elif self.is_reach_target():
                _terminated = True
                _reward = 500
                print(colored("**** Episode Finished **** SUCCESS.", 'green'))
            else:
                _reward = -10 * self.dt - 10 * (d - self.d_2_target)

                #####################################################################################################                
                # # Condition 1:_dis_2_barricade = 12 avoid_penalty=-60 此时碰撞圆柱
                # # Condition 2:_dis_2_barricade = 24 avoid_penalty=-8 此时进入碰撞安全范围内,关注避障
                # if _dis_2_barricade > 24:
                #     avoid_penalty = 0
                # else:
                #     avoid_penalty = -60 * np.exp(-(_dis_2_barricade-12) / 6)
                # _reward = -2 * self.dt - 10 * (d - self.d_2_target) + avoid_penalty
                #####################################################################################################

                # Roll Count
                current_roll_count = self.angle / (2 * pi)
                full_turns = int(current_roll_count) - int(self._roll_count)
                if full_turns != 0:
                    print(colored(f"**** Over Roll:{int(self._roll_count)} to {int(current_roll_count)} ****", 'blue'))
                    _reward += abs(full_turns) * self._roll_punish
                self._roll_count = current_roll_count
        
        # Update
        self.d_2_target = d
        self.reward = _reward
        # self._old_dis_2_barricade = _dis_2_barricade
        # self.pre_pressure = self.pressure
        self.state = state_1

        if self.include_flow:
            self.u_flow = v_x
            self.v_flow = v_y
            # Flow Plot
            if self._plot_flow:
                self.agent_pos_history.append(self.agent_pos)
                self.speed = sqrt(state_1[0]**2 + state_1[1]**2)
                _save = _terminated or _truncated
                self._render_frame(_save=_save)
            if self._proccess_flow:
                # Get Flow Speed
                v_x_grid, v_y_grid = self.get_flow_velocity()
                self.last_flow_x = v_x_grid
                self.last_flow_y = v_y_grid
                # Switch to PyTorch Tensor, Add batch dim
                flow_input = np.stack([v_x_grid, v_y_grid], axis=0) # Stack at batch_size dim
                return self.state, self.reward, _terminated, _truncated, _info, flow_input
        return self.state, self.reward, _terminated, _truncated, _info


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
            if not self._is_in_circles(point):
                return point

    def reset(self):
        ########################################################################################
        # Step1: Task Reset
        self.step_counter=0
        self.agent_pos_history = []
        self._roll_count = 0
        self._collision_count = 0

        # 1.1 Reset Flow ID
        if self.flow_num:
            flow_init = self.flow_num
            print(f"Flow Reset Fixed:{flow_init}")
        else:
            flow_init = random.randint(1, 30)
            print(f"Flow Reset Random:{flow_init}")

        # 1.2 Reset Obstacle
        obstacle_path = f'./PinballCFD_server/saved/init/init_{str(flow_init)}.txt'
        centers = np.loadtxt(obstacle_path)
        # print(f"centers:{centers}")
        self.circles = [
            {"center": (centers[0], centers[1]), "radius": self.window_r/2},
            {"center": (centers[2], centers[3]), "radius": self.window_r/2},
            {"center": (centers[4], centers[5]), "radius": self.window_r/2}
        ]
        
        # 1.3 Reset Agent & Target
        if not self._is_fixed_target:
            print("Random Target")
            self.target_position = self.generate_point_outside_circles(self.target_default, self.random_range)
        if not self._is_fixed_start:
            print("Random Staet")
            self.start_position = self.generate_point_outside_circles(self.start_default, self.random_range)
        # if random.random() < 0.25:
        #     print("Switch")
        #     self.start_position, self.target_position = self.target_position, self.start_position
        print("target_pos:", self.target_position)
        print("start_pos:", self.start_position)
        agent_init_x = self.start_position[0]
        agent_init_y = self.start_position[1]
        ########################################################################################

        ########################################################################################
        # Step2:Reset true environment
        # action_json = {"v1": 0, "v2": 0, "v3": 0}
        # 2.1 Send Info To Sever
        action_json = {"v1": 0, "v2": 0, "v3": 0, "init_x": agent_init_x, "init_y": agent_init_y, "flow_num":float(flow_init)}

        # 2.2 Server Reset
        res_str = self.proxy.connect.reset(json.dumps(action_json))
        if self.include_flow:
            # Unpacking
            step_state, vflow_x, vflow_y = self.parseStep(res_str, self.include_flow)
            v_x = np.array(vflow_x, dtype=np.float32)
            v_y = np.array(vflow_y, dtype=np.float32)
        else:
            step_state = self.parseStep(res_str, self.include_flow)
        state_1 = np.array(step_state, dtype=np.float32)
        _info = {"vel_x": state_1[0], "vel_y": state_1[1], "vel_angle": state_1[2],
                 "pos_x": state_1[3], "pos_y": state_1[4], "angle": state_1[5],
                 "pressure_1": state_1[6], "pressure_2": state_1[7], "pressure_3": state_1[8],
                 "pressure_4": state_1[9], "pressure_5": state_1[10], "pressure_6": state_1[11],
                 "pressure_7": state_1[12], "pressure_8": state_1[13]}

        # 2.3 Agent Info
        self.agent_pos = [state_1[3], state_1[4]]
        ########################################################################################

        ########################################################################################
        # # Distance to Obstacle
        # distances = [np.linalg.norm(np.array(self.agent_pos) - np.array(circle["center"])) for circle in self.circles]
        # # Nearest Obstacle
        # nearest_idx = np.argmin(distances)
        # nearest_circle = self.circles[nearest_idx]
        # dx, dy = np.array(nearest_circle["center"]) - np.array(self.agent_pos)
        # _dis_2_barricade = distances[nearest_idx]
        # self._old_dis_2_barricade = _dis_2_barricade
        
        # # All distance
        # relative_positions = []
        # min_distance = float('inf')
        # for circle in self.circles[:3]:
        #     dx, dy = np.array(circle["center"]) - np.array(self.agent_pos)
        #     relative_positions.extend([dx, dy])
        #     distance = np.linalg.norm([dx, dy])
        #     min_distance = min(min_distance, distance)
        # _dis_2_barricade = min_distance 
        # relative_positions = np.array(relative_positions, dtype=np.float32)
        # self._old_dis_2_barricade = _dis_2_barricade
        ########################################################################################

        d = np.linalg.norm(self.agent_pos - self.target_position)
        self.d_2_target = d
        state_1[3], state_1[4] = self.target_position[0] - state_1[3], self.target_position[1] - state_1[4]
        self.angle = state_1[5]
        # self.pre_pressure = state_1[6:14]
        # self.d_init = d

        # TODO:dim=14
        self.state = state_1
        # TODO:dim=16
        # self.state = np.hstack([state_1, np.array([dx, dy])]).astype(np.float32)
        # TODO:dim=20
        # self.state = np.hstack([state_1, relative_positions]).astype(np.float32)
        
        if self.include_flow:
            self.u_flow = v_x
            self.v_flow = v_y
            # Plot Flow
            if self._plot_flow:
                self.agent_pos_history.append(self.agent_pos)
                self.speed = sqrt(state_1[0]**2 + state_1[1]**2)
                self._render_frame()
            if self._proccess_flow:
                # Get Flow Speed
                v_x_grid, v_y_grid = self.get_flow_velocity()
                self.last_flow_x = v_x_grid
                self.last_flow_y = v_y_grid
                # Switch to PyTorch Tensor, Add batch dim
                flow_input = np.stack([v_x_grid, v_y_grid], axis=0) # Stack at batch_size dim
                return self.state, _info, flow_input
        return self.state, _info
    

    def parseStep(self, info, _is_flow):
        all_info = json.loads(info)
        state = json.loads(all_info['state'][0])
        state_ls = [state['vel_x'], state['vel_y'], state['vel_angle'], state['pos_x'], state['pos_y'], state['angle'],
                    state['surfacePressures_1'], state['surfacePressures_2'], state['surfacePressures_3'],
                    state['surfacePressures_4'], state['surfacePressures_5'], state['surfacePressures_6'],
                    state['surfacePressures_7'], state['surfacePressures_8']]
        if _is_flow:
            flow_u = all_info.get('flow_u')
            flow_v = all_info.get('flow_v')
            return state_ls, flow_u, flow_v
        return state_ls


    def is_in_circle(self, circles):
        # Hitting obstacle
        for circle in circles:
            center = np.array(circle["center"])
            radius = circle["radius"] 
            a = self.ellipse_a
            b = self.ellipse_b
            distance = np.linalg.norm(self.agent_pos - center)
            if distance <= (b + radius):
                return True
        return False
    
    def is_out_of_bounds(self):
        # pseudo-boundary
        _delta = 2
        x_c, y_c = self.agent_pos
        a = self.ellipse_a
        b = self.ellipse_b
        # anticlockwise
        theta = self.state[5]
        # agent coordinate
        vertices = [(a, 0), (-a, 0), (0, b), (0, -b)]
        # world coordinate
        rotated_vertices = []
        for vx, vy in vertices:
            x_rot = x_c + vx * np.cos(theta) - vy * np.sin(theta)
            y_rot = y_c + vx * np.sin(theta) + vy * np.cos(theta)
            rotated_vertices.append((x_rot, y_rot))
        for x, y in rotated_vertices:
            if not (_delta <= x <= self.x_range - _delta and _delta <= y <= self.y_range - _delta):
                print(f"Out of bounds: Agent_pos {x_c, y_c}; Angle {theta}")
                print(f"Steps: {self.step_counter}")
                return True
        return False

    
    def is_reach_target(self):
        x_c, y_c = self.agent_pos
        x_o, y_o = self.target_position
        # anticlockwise
        theta = self.state[5]
        a = self.ellipse_a
        b = self.ellipse_b
        # world coordinate to agent coordinate
        dx = x_o - x_c
        dy = y_o - y_c
        x_prime = dx * np.cos(theta) + dy * np.sin(theta)
        y_prime = -dx * np.sin(theta) + dy * np.cos(theta)
        # dx < a and dy < b
        return (x_prime**2 / a**2 + y_prime**2 / b**2) <= 1

    def show_info(self, info):
        all_info = json.loads(info)
        state_json_str = all_info['state'][0]
        # state_json_str must be str or dic
        if isinstance(state_json_str, str):
            state_dict = json.loads(state_json_str)
        elif isinstance(state_json_str, dict):
            state_dict = state_json_str
        else:
            raise TypeError(f"Expected string or dict, got {type(state_json_str)}")
        # print labels
        for label in state_dict.keys():
            print(label)

    def terminate(self):
        pid = os.getpgid(self.server.pid)
        self.server.terminate()
        # Send the signal to all the process groups
        os.killpg(pid, signal.SIGTERM)  

    def close(self):
        if self.local_port == None:
            self.server.terminate()
    
    def _render_frame(self, _save=False):
        if len(self.agent_pos_history) == 0:
            return
        # clear coordinate
        ax = self.ax
        ax.clear()
        # draw
        self.plot_env(ax)
        plt.draw()  
        plt.pause(self.frame_pause)
        if _save:
            save_path = './model_output/path.png'
            # make sure path is legal
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Image saved to {save_path}")



    def plot_env(self, ax, sample_rate=10):
        history = self.agent_pos_history
        start_default = [220, 64]
        target_default = [64, 64]
        start_point = history[0]
        target = self.target_position
        circles = self.circles
        agent_pos = self.agent_pos
        agent_angle = self.angle
        x_range = self.x_range
        y_range = self.y_range
        flow_x = self.u_flow
        flow_y = self.v_flow
        speed = self.speed
        flow_speed = self.flow_speed

        ax.clear()
        # Plot Flow Speed
        if flow_x is not None and flow_y is not None:
            speed_field = np.sqrt(flow_x**2 + flow_y**2)

            # heat map of flow speed
            cmap = mcolors.LinearSegmentedColormap.from_list("red_white_blue", ["blue", "white", "red"])
            # normalize
            norm = mcolors.Normalize(vmin=np.min(speed_field), vmax=np.max(speed_field))
            # grid
            x = np.linspace(0, x_range, flow_x.shape[0])
            y = np.linspace(0, y_range, flow_x.shape[1])
            X, Y = np.meshgrid(x, y)
            # using pcolormesh replace imshow
            im = ax.pcolormesh(X, Y, speed_field.T, cmap=cmap, norm=norm, shading='auto', alpha=0.5)
            # colorbar info
            if not hasattr(ax, "colorbar"):
                ax.colorbar = plt.colorbar(im, ax=ax, fraction=0.05, label="Speed (m/s)")

            # speed quiver
            sample_rate = max(10, sample_rate)
            x_sample = x[::sample_rate]
            y_sample = y[::sample_rate]
            X_sample, Y_sample = np.meshgrid(x_sample, y_sample)
            sampled_flow_x = flow_x[::sample_rate, ::sample_rate]
            sampled_flow_y = flow_y[::sample_rate, ::sample_rate]
            ax.quiver(X_sample, Y_sample, sampled_flow_x * 1.5, sampled_flow_y * 1.5, 
                    scale=100, scale_units='width', color='black', alpha=0.6, zorder=2, 
                    width=0.002, pivot='middle', headwidth=3, headaxislength=3)

        # Start
        ax.scatter(*start_point, color='orange', label='Start Point', zorder=5)
        # Target
        ax.scatter(*target, color='green', label='Target Point', zorder=5)
        # Cylinder
        for circle in circles:
            ax.add_patch(plt.Circle(circle["center"], circle["radius"], color='red', alpha=0.3))

        # History Trajectory
        if history:
            hx, hy = zip(*history)
            ax.plot(hx, hy, linestyle='--', color='orange', label='Path')

        # Random Range
        ax.add_patch(plt.Circle(target_default, 56, color='green', fill=False, linestyle='--'))
        ax.add_patch(plt.Circle(start_default, 56, color='orange', fill=False, linestyle='--'))

        # Agent(Ellipse)
        ellipse_height = circle["radius"]
        ellipse_width = ellipse_height / 1.5
        ellipse = Ellipse(xy=agent_pos, width=ellipse_height, height=ellipse_width, 
                            angle= np.degrees(agent_angle), color='orange', alpha=0.7, zorder=4)
        ax.add_patch(ellipse)

        # Agent Coordinate
        axis_length = 10.0
        cos_angle = np.cos(agent_angle)
        sin_angle = np.sin(agent_angle)

        ax.quiver(agent_pos[0], agent_pos[1], axis_length * cos_angle, axis_length * sin_angle,
                    angles='xy', scale_units='xy', scale=1, width=0.002, headwidth=2, color='red', label="Agent_X")
        ax.quiver(agent_pos[0], agent_pos[1], -axis_length * sin_angle, axis_length * cos_angle,
                    angles='xy', scale_units='xy', scale=1, width=0.002, headwidth=2, color='green', label="Agent_Y")

        # Axis
        ax.set_xlim(0, x_range)
        ax.set_ylim(0, y_range)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        ax.set_title("Moving Trajectory")
