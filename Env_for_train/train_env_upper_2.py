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
import matplotlib.patches as patches
from matplotlib.patches import Arc


# Port Check
def is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

class foil_env:
    def __init__(self, config=None, info='', local_port=None, network_port=None,
             target_center=None, target_position=None, start_center=None, 
             start_position=None, max_step=None, _include_flow=False, 
             _plot_flow=False, _proccess_flow=False, _random_range = 56, _flow_range=16, _init_flow_num=0, _pos_normalize=True, _is_test=False, _is_random = True, _set_random=0,
             _state_dim=18, u_range=20.0, v_range=20.0, w_range=5.0, _forward_only = True, _obstacle_avoid=False, _is_switch=False, _theta_mode=True, u_clip=20.0, v_clip=20.0, w_clip=5.0):
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
        self._obstacle_avoid = _obstacle_avoid
        self._is_switch = _is_switch
        self._theta_mode = _theta_mode
        #################################################

        ###################################################
        # State Space
        self.include_flow = _include_flow
        self._pos_normalize = _pos_normalize
        # select mode: train or test
        self._is_test = _is_test 
        self._is_random = _is_random
        self._set_random = _set_random
        self.observation_dim = _state_dim
        self.state = np.zeros(self.observation_dim)
        self.observation_space = Box(low=-1e6, high=1e6, shape=[self.observation_dim])
        #####################################################

        #####################################################
        # Action Space
        self.forward_mode = _forward_only
        if _forward_only:
            # self.action_dim = 2
            # low = np.array([0, 6], dtype=np.float32)
            # high = np.array([np.pi, 12], dtype=np.float32)
            self.action_dim = 1
            low = np.array([0], dtype=np.float32)
            high = np.array([np.pi], dtype=np.float32)
        else:
            if self._theta_mode:
                self.action_dim = 1
                low = np.array([-np.pi], dtype=np.float32)
                high = np.array([np.pi], dtype=np.float32)
            else:
                self.action_dim = 2
                low = np.array([-1.0, -1.0], dtype=np.float32)
                high = np.array([1.0, 1.0], dtype=np.float32)
        self.low = low
        self.high = high
        self.action_space = Box(low=low, high=high, shape=(self.action_dim,), dtype=np.float32)
        #####################################################

        #####################################################
        # Low-Level Controller
        self.lc_action_dim = 3
        lc_low = np.array([-float(u_range), -float(v_range), -float(w_range)], dtype=np.float32)
        lc_high = np.array([float(u_range), float(v_range), float(w_range)], dtype=np.float32)
        self.lc_low = lc_low
        self.lc_high = lc_high
        self.lc_action_space = Box(low=lc_low, high=lc_high, shape=(self.lc_action_dim,), dtype=np.float32)
        #####################################################


        #####################################################
        # Max Speed Limit Controller
        clip_low = np.array([-float(u_clip), -float(v_clip), -float(w_clip)], dtype=np.float32)
        clip_high = np.array([float(u_clip), float(v_clip), float(w_clip)], dtype=np.float32)
        self.clip_low = clip_low
        self.clip_high = clip_high
        #####################################################

        #####################################################
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
        self.target_default = np.array([64, 64])
        # Target Range Center
        if target_center is not None:
            self.target_default = target_center
        self.start_default = np.array([220, 64])
        # Start Range Center
        if start_center is not None:
            self.start_default = start_center
        self.target_position = self.target_default
        self.target_position_lc = self.target_position
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
        self.d_2_target_min = np.inf
        self._rolls = 0
        self._roll_count = 0
        self._collision_count = 0
        self._roll_punish = -50

        ############################
        # self.max_detect_dis = 24
        # self._old_dis_2_barricade = 24
        self.max_detect_dis = 24 - self.window_r/2
        self._old_dis_2_barricade = 24 - self.window_r/2
        self.safe_range_flag = False
        ############################

        self.old_direction = 0
        self.old_angle = 1
        #######################################################

        

        #########################################################
        # State Record
        # Agent Info
        self.agent_pos_history = []
        self.angle = 0
        self.vel_angle = 0
        self.speed = 0
        self.pressure = []
        # Flow Info
        self.u_flow = []
        self.v_flow = []
        self._proccess_flow = _proccess_flow
        self.flow_speed = 0
        if self._proccess_flow:
            self.flow_range = _flow_range
            self.last_flow_x = np.zeros((2 * self.flow_range, 2 * self.flow_range))
            self.last_flow_y = np.zeros((2 * self.flow_range, 2 * self.flow_range))
        # Action Info
        self.action = []
        # Low-Level Controller Info
        self.state_14 = []
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
        # port = int(6000)
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
        

        if self._is_random:
            random.seed(None)
        else:
            random.seed(self._set_random)
        #############################################################


    def get_flow_velocity(self):
        """
        Compute the velocity field around the agent and store it in 16x16 grids.
        All returned values will be clipped to [-2.5, 2.5].

        Returns:
        v_x_grid, v_y_grid : Two numpy arrays where the small square region is filled with 0,
                            and values clipped to [-2.5, 2.5].
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

        # Clip values to [-2.5, 2.5]
        v_x_region = np.clip(v_x_region, -2.5, 2.5)
        v_y_region = np.clip(v_y_region, -2.5, 2.5)

        return v_x_region[:_cut_range, :_cut_range], v_y_region[:_cut_range, :_cut_range]


    def clip_action(self, action):
        # action = np.clip(action, self.lc_low, self.lc_high)
        action = np.clip(action, self.clip_low, self.clip_high)
        return action
    
    def sense(self,dist):
        sense_thresh = self.max_detect_dis+self.window_r/2
        if dist > sense_thresh:
            return 1.0
        return dist/ sense_thresh
    
    def set_lc_target(self, _lc_target):
        self.target_position_lc = _lc_target

    def step(self, action):
        self.safe_range_flag = False
        ########################
        # 1.1 Step Settings
        self.step_counter += 1
        action = self.clip_action(action)
        _truncated = False
        _terminated = False
        _reward = 0
        state_1 = self.state
        _dis_2_barricade = self._old_dis_2_barricade
        angle_norm = self.old_angle
        direction = self.old_direction
        self.action = action
        ########################

        ########################
        # 1.2 Simulator Act
        step_1 = float(action[0])
        step_2 = float(action[1])
        step_3 = float(action[2])
        action_json = {"v1": step_1, "v2": step_2, "v3": step_3}
        # action_json_0 = {"v1": 0, "v2": 0, "v3": 0}
        # Sim_Steps:10
        for i in range(self.action_interval):
            res_str = self.proxy.connect.Step(json.dumps(action_json))
            if self.include_flow:
                step_state, vflow_x, vflow_y = self.parseStep(res_str, self.include_flow)
                v_x = np.array(vflow_x, dtype=np.float32)
                v_y = np.array(vflow_y, dtype=np.float32)
            else:
                step_state = self.parseStep(res_str, self.include_flow)
            state_1 = np.array(step_state, dtype=np.float32)
            state_2 = np.array(step_state, dtype=np.float32)
        _info = {"vel_x": state_1[0], "vel_y": state_1[1], "vel_angle": state_1[2],
                 "pos_x": state_1[3], "pos_y": state_1[4], "angle": state_1[5],
                 "pressure_1": state_1[6], "pressure_2": state_1[7], "pressure_3": state_1[8],
                 "pressure_4": state_1[9], "pressure_5": state_1[10], "pressure_6": state_1[11],
                 "pressure_7": state_1[12], "pressure_8": state_1[13]}
        self.agent_pos = [state_1[3], state_1[4]]
        self.angle = state_1[5]
        self.speed = sqrt(state_1[0]**2 + state_1[1]**2)
        self.vel_angle = abs(state_1[2]) # [1e-2, 1e-1]
        self.pressure = state_1[6:14]
        ########################

        ########################################################################################
        # Distance to Target
        d = np.linalg.norm(self.agent_pos - self.target_position)
        ########################################################################################

        ########################################################################################
        # obstacle distance (filtered by ±90° within ship heading direction, local ship coords)

        agent_pos = np.array(self.agent_pos, dtype=np.float32)
        agent_angle = self.angle  # agent heading(rad);+:anticlockwise

        # Rotation matrix: rotate -agent_angle around Z-axis
        cos_a = np.cos(-agent_angle)
        sin_a = np.sin(-agent_angle)
        R = np.array([[cos_a, -sin_a],
                    [sin_a,  cos_a]])

        # Field of view ±90°
        _fov = np.pi

        _barricade_states = []

        for circle in self.circles[:3]:
            obstacle_pos = np.array(circle["center"], dtype=np.float32)
            obstacle_radius = circle.get("radius", 1.0)  # Use default radius if not provided

            # Obstacle vector in world coordinates (center to agent)
            vec_world = obstacle_pos - agent_pos
            center_distance = np.linalg.norm(vec_world)

            # Compute distance to edge of the obstacle
            edge_distance = center_distance - obstacle_radius
            if edge_distance < 1e-5 or edge_distance > self.max_detect_dis:
                continue

            # Unit vector pointing from center to agent
            unit_vec_world = vec_world / (center_distance + 1e-8)

            # Compute the edge point in world coordinates (closest point on the circle)
            edge_point_world = obstacle_pos - unit_vec_world * obstacle_radius

            # Transform edge point to agent's local coordinates
            vec_edge_local = R @ (edge_point_world - agent_pos)

            # Check if within ±90° field of view
            forward_local = np.array([-1.0, 0.0], dtype=np.float32)
            unit_vec_local = vec_edge_local / (np.linalg.norm(vec_edge_local) + 1e-8)
            dot = np.dot(forward_local, unit_vec_local)
            angle = np.arccos(np.clip(dot, -1.0, 1.0))

            if angle <= (_fov / 2):
                # Append: (distance to edge, local x, local y, world coordinates of edge)
                _barricade_states.append((
                    edge_distance,
                    vec_edge_local[0],
                    vec_edge_local[1],
                    edge_point_world
                ))

        # obstacle state(default)
        if len(_barricade_states) == 0:
            closest_distance = self.max_detect_dis
            if self._pos_normalize:
                closest_dx, closest_dy = -1.0, 1.0
            else:
                closest_dx, closest_dy = self.max_detect_dis, self.max_detect_dis
            direction = 0
            angle_rad = np.pi
        else:
            self.safe_range_flag = True
            # sort by distance
            _barricade_states.sort(key=lambda x: x[0])
            closest_distance = _barricade_states[0][0]

            if self._pos_normalize:
                raw_dx = _barricade_states[0][1]
                raw_dy = _barricade_states[0][2]
                closest_dx = raw_dx / self.max_detect_dis
                closest_dx = np.clip(closest_dx, -1.0, 0.0)
                closest_dy = raw_dy / self.max_detect_dis
                closest_dy = np.clip(closest_dy, -1.0, 1.0)
            else:
                closest_dx = _barricade_states[0][1]
                closest_dy = _barricade_states[0][2]

            obstacle_pos = np.array(_barricade_states[0][3], dtype=np.float32)
            agent_pos = np.array(self.agent_pos, dtype=np.float32)
            target_pos = np.array(self.target_position, dtype=np.float32)

            vec1 = agent_pos - obstacle_pos
            vec2 = target_pos - obstacle_pos

            # normalize
            unit_vec1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
            unit_vec2 = vec2 / (np.linalg.norm(vec2) + 1e-8)

            dot = np.dot(unit_vec1, unit_vec2)
            cross = np.cross(unit_vec1, unit_vec2)
            angle_rad = np.arccos(np.clip(dot, -1.0, 1.0))
            direction = np.sign(cross)

            # Stability filter for near-edge cases
            # angle_thresh = np.deg2rad(30)
            # if angle_rad < angle_thresh:
            #     direction = 0  
        _dis_2_barricade = min(closest_distance, self.max_detect_dis)
        barricade_state = np.array([closest_dx, closest_dy], dtype=np.float32)
        angle_norm = angle_rad / np.pi
        ########################################################################################

        ########################################################################################
        x_min, x_max = 4, self.x_range - 4
        y_min, y_max = 4, self.y_range - 4
        x, y = agent_pos
        dist_left = x - x_min
        dist_right = x_max - x
        dist_bottom = y - y_min
        dist_top = y_max - y

        state_border = np.array([
            self.sense(dist_left),
            self.sense(dist_right),
            self.sense(dist_bottom),
            self.sense(dist_top)
        ], dtype=np.float32)
        ########################################################################################

        ########################
        # Low-Level Controller State
        state_2[3], state_2[4] = self.target_position[0] - state_2[3], self.target_position[1] - state_2[4]
        self.state_14 = state_2
        ########################
        ########################
        # Relative Positon(Target & World Coordinate)
        if self._pos_normalize:
            near_threshold = self.max_detect_dis + self.window_r/2
            # === Extra fine-grained relative position ===
            if d < near_threshold:
                rel_x = (self.target_position[0] - state_1[3])
                rel_y = (self.target_position[1] - state_1[4])
                near_flag = 1.0
            else:
                rel_x = 0.0
                rel_y = 0.0
                near_flag = 0.0
            # === Normalized relative position for base state ===
            state_1[3], state_1[4] = (self.target_position[0] - state_1[3])/near_threshold, (self.target_position[1] - state_1[4])/near_threshold
        else:
            # === Unnormalized relative position for base state ===
            state_1[3], state_1[4] = (self.target_position[0] - state_1[3]), (self.target_position[1] - state_1[4])
        ########################
 
        # Step Reward
        #####################################################################################################
        if self.is_out_of_bounds():
            _truncated = True
            _reward = -500
            # _reward = 0
            print(colored(f"**** Episode Finished At: {self.step_counter} **** Hit Boundary.", 'red'))
        #####################################################################################################
        else:
            #####################################################################################################
            if self.is_in_circle(self.circles):
                self._collision_count += 1
                # end episode
                if self._collision_count > 2:
                    _truncated = True
                    _reward = -250
                    print(colored(f"**** Episode Finished At: {self.step_counter} **** Too Many Collisions.", 'red'))
                else:
                    _reward = -250
                    print(colored(f"**** Forbidden Move **** Hit Circles{int(self._collision_count)}.", 'yellow'))
            ##################################################################################################### 

            #####################################################################################################
            elif self.step_counter >= self.t_step_max or self.done:
                _truncated = True
                # _reward = -50
                _reward = 0
                print(colored("**** Episode Finished **** Reaches Env Time limit.", 'blue'))
            #####################################################################################################

            #####################################################################################################
            elif self.is_reach_target():
                _terminated = True
                _reward = 500
                print(colored(f"**** Episode Finished At: {self.step_counter} **** SUCCESS.", 'green'))
            #####################################################################################################

            #####################################################################################################
            else:
                ########################
                # _reward = -10 * self.dt - 10 * (d - self.d_2_target)
                ########################

                ########################
                if self.old_direction == 0 and direction !=0:
                    print(colored(f"**** Direction Confirm: {direction} ****", 'magenta'))   
                if self.old_direction != 0 and direction !=0 and self.old_direction != direction:
                    print(colored(f"**** Direction Switch: {self.old_direction} -> {direction} ****", 'magenta'))
                ########################

                ########################
                avoid_penalty = 25 * (_dis_2_barricade - self._old_dis_2_barricade)
                target_reward = 10 * (self.d_2_target - d)
                angle_reward = 5 * (self.old_angle - angle_norm) * abs(self.old_direction) * abs(direction)
                roll_swich_punish = -100 * abs(direction - self.old_direction) * abs(self.old_direction) * abs(direction)
                border_penalty = -50 * np.sum(1.0 - state_border)
                ########################

                ########################
                # mixed reward
                if self.observation_dim == 14:
                    _reward = -50 * self.dt + target_reward
                if self.observation_dim == 16:
                    _reward = -50 * self.dt + target_reward + avoid_penalty
                if self.observation_dim == 18:
                    if self._obstacle_avoid:
                        _reward = -50 * self.dt + target_reward + avoid_penalty + angle_reward + roll_swich_punish
                    else:
                        _reward = -50 * self.dt + target_reward + border_penalty
                if self.observation_dim == 20:
                    _reward = -50 * self.dt + target_reward + avoid_penalty + border_penalty  
                if self.observation_dim == 22:
                    _reward = -50 * self.dt + target_reward + 2 * avoid_penalty + angle_reward + roll_swich_punish + border_penalty 
                if self.observation_dim == 25:
                    _reward = -50 * self.dt + target_reward + avoid_penalty + angle_reward + border_penalty + roll_swich_punish 
                    if d < near_threshold:
                        _reward = -10 * self.dt + 5 * target_reward
                ########################

                ########################
                # Roll Count & Terminate
                current_roll_count = self.angle / (2 * pi)
                full_turns = int(current_roll_count) - int(self._roll_count)

                if full_turns != 0:
                    self._rolls += 1
                    print(colored(f"**** Over Roll: {int(self._roll_count)} to {int(current_roll_count)} ****", 'blue'))
                    _reward += abs(full_turns) * self._roll_punish

                    # === too many rolls ===
                    max_roll_threshold = 5
                    if self._rolls > max_roll_threshold:
                        print(colored(f"**** Episode terminated: Over-Roll At: {self.step_counter} ****", 'red'))
                        _truncated = True
                        _reward = -500

                self._roll_count = current_roll_count
                ########################
            #####################################################################################################
        
        # Update
        self.d_2_target = d
        self.d_2_target_min = min(self.d_2_target, self.d_2_target_min)
        self.reward = _reward
        self._old_dis_2_barricade = _dis_2_barricade
        self.old_angle = angle_norm
        self.old_direction = direction
        if self.observation_dim == 14:
            # only position
            self.state = state_1
        if self.observation_dim == 16:
            # barricate sensor
            self.state = np.hstack([state_1, barricade_state]).astype(np.float32)
        if self.observation_dim == 18:
            # roll direction guidance
            if self._obstacle_avoid:
                self.state = np.hstack([state_1, barricade_state, direction, angle_norm]).astype(np.float32)
            else:
                self.state = np.hstack([state_1, state_border]).astype(np.float32)
        if self.observation_dim == 20:
            # edge sensor + barricate sensor
            self.state = np.hstack([state_1, barricade_state, state_border]).astype(np.float32)
        if self.observation_dim == 22:
            # edge sensor
            self.state = np.hstack([state_1, barricade_state, direction, angle_norm, state_border]).astype(np.float32)
        if self.observation_dim == 25:
            # near target sensor
            self.state = np.hstack([state_1, barricade_state, direction, angle_norm, state_border, rel_x, rel_y, near_flag]).astype(np.float32)

        if self.include_flow:
            self.u_flow = v_x
            self.v_flow = v_y
            # Flow Plot
            if self._plot_flow:
                self.agent_pos_history.append(self.agent_pos)
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

    def _rand_in_half_circle(self, origin, r, _half_flag):
        d = sqrt(np.random.uniform(0, r ** 2))
        if _half_flag == 1:
            theta = np.random.uniform(0, pi)
        else:
            theta = np.random.uniform(pi, 2 * pi)
        return origin + np.array([d * cos(theta), d * sin(theta)])

    def _rand_in_circle(self, origin, r):
        d = sqrt(np.random.uniform(0, r ** 2))
        theta = np.random.uniform(0, 2 * pi)
        return origin + np.array([d * cos(theta), d * sin(theta)])

    def _is_in_circles(self, point):
        for circle in self.circles:
            center = np.array(circle["center"])
            #############################################
            radius = self.max_detect_dis + 2
            #############################################
            if np.linalg.norm(point - center) < radius:
                return True
        return False
    
    def half_generate_point_outside_circles(self, origin, r, _half_flag):
        while True:
            point = self._rand_in_half_circle(origin, r, _half_flag)
            if not self._is_in_circles(point):
                return point

    def generate_point_outside_circles(self, origin, r):
        while True:
            point = self._rand_in_circle(origin, r)
            if not self._is_in_circles(point):
                return point

    def reset(self):
        self.safe_range_flag = False
        ########################################################################################
        # Step1: Task Reset
        angle_norm = 1
        direction = 0
        self.step_counter=0
        self.agent_pos_history = []
        self._roll_count = 0
        self._rolls = 0
        self._collision_count = 0
        self._old_dis_2_barricade = self.max_detect_dis
        _dis_2_barricade = self.max_detect_dis
        self.old_direction = 0
        self.old_angle = 1
        self.action = np.array([0, 0, 0], dtype=np.float32)

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
        ########################
        if not self._is_fixed_target:
            print("Random Target")
            self.target_position = self.generate_point_outside_circles(self.target_default, self.random_range)
            self.target_position_lc = self.target_position
        if not self._is_fixed_start:
            print("Random Start")
            self.start_position = self.generate_point_outside_circles(self.start_default, self.random_range)
        ########################

        # --- Switch Start and Target ---
        if self._is_switch:
            if np.random.rand() < 0.5:
                print("Swapping start and target")
                tmp = self.start_position.copy()
                self.start_position = self.target_position.copy()
                self.target_position = tmp.copy()
                self.target_position_lc = self.target_position

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
        state_2 = np.array(step_state, dtype=np.float32)
        _info = {"vel_x": state_1[0], "vel_y": state_1[1], "vel_angle": state_1[2],
                 "pos_x": state_1[3], "pos_y": state_1[4], "angle": state_1[5],
                 "pressure_1": state_1[6], "pressure_2": state_1[7], "pressure_3": state_1[8],
                 "pressure_4": state_1[9], "pressure_5": state_1[10], "pressure_6": state_1[11],
                 "pressure_7": state_1[12], "pressure_8": state_1[13]}

        # 2.3 Agent Info
        self.agent_pos = [state_1[3], state_1[4]]
        self.angle = state_1[5]
        self.speed = sqrt(state_1[0]**2 + state_1[1]**2)
        self.pressure = state_1[6:14]
        ########################################################################################

        ########################################################################################
        # Distance to Target
        d = np.linalg.norm(self.agent_pos - self.target_position)
        ########################################################################################

        ########################################################################################
        # obstacle distance (filtered by ±90° within ship heading direction, local ship coords)

        agent_pos = np.array(self.agent_pos, dtype=np.float32)
        agent_angle = self.angle  # agent heading(rad);+:anticlockwise

        # Rotation matrix: rotate -agent_angle around Z-axis
        cos_a = np.cos(-agent_angle)
        sin_a = np.sin(-agent_angle)
        R = np.array([[cos_a, -sin_a],
                    [sin_a,  cos_a]])

        # Field of view ±90°
        _fov = np.pi

        _barricade_states = []

        for circle in self.circles[:3]:
            obstacle_pos = np.array(circle["center"], dtype=np.float32)
            obstacle_radius = circle.get("radius", 1.0)  # Use default radius if not provided

            # Obstacle vector in world coordinates (center to agent)
            vec_world = obstacle_pos - agent_pos
            center_distance = np.linalg.norm(vec_world)

            # Compute distance to edge of the obstacle
            edge_distance = center_distance - obstacle_radius
            if edge_distance < 1e-5 or edge_distance > self.max_detect_dis:
                continue

            # Unit vector pointing from center to agent
            unit_vec_world = vec_world / (center_distance + 1e-8)

            # Compute the edge point in world coordinates (closest point on the circle)
            edge_point_world = obstacle_pos - unit_vec_world * obstacle_radius

            # Transform edge point to agent's local coordinates
            vec_edge_local = R @ (edge_point_world - agent_pos)

            # Check if within ±90° field of view
            forward_local = np.array([-1.0, 0.0], dtype=np.float32)
            unit_vec_local = vec_edge_local / (np.linalg.norm(vec_edge_local) + 1e-8)
            dot = np.dot(forward_local, unit_vec_local)
            angle = np.arccos(np.clip(dot, -1.0, 1.0))

            if angle <= (_fov / 2):
                # Append: (distance to edge, local x, local y, world coordinates of edge)
                _barricade_states.append((
                    edge_distance,
                    vec_edge_local[0],
                    vec_edge_local[1],
                    edge_point_world
                ))

        # obstacle state(default)
        if len(_barricade_states) == 0:
            closest_distance = self.max_detect_dis
            if self._pos_normalize:
                closest_dx, closest_dy = -1.0, 1.0
            else:
                closest_dx, closest_dy = self.max_detect_dis, self.max_detect_dis
            direction = 0
            angle_rad = np.pi
        else:
            # sort by distance
            _barricade_states.sort(key=lambda x: x[0])
            closest_distance = _barricade_states[0][0]

            if self._pos_normalize:
                raw_dx = _barricade_states[0][1]
                raw_dy = _barricade_states[0][2]
                closest_dx = raw_dx / self.max_detect_dis
                closest_dx = np.clip(closest_dx, -1.0, 0.0)
                closest_dy = raw_dy / self.max_detect_dis
                closest_dy = np.clip(closest_dy, -1.0, 1.0)
            else:
                closest_dx = _barricade_states[0][1]
                closest_dy = _barricade_states[0][2]

            obstacle_pos = np.array(_barricade_states[0][3], dtype=np.float32)
            agent_pos = np.array(self.agent_pos, dtype=np.float32)
            target_pos = np.array(self.target_position, dtype=np.float32)

            vec1 = agent_pos - obstacle_pos
            vec2 = target_pos - obstacle_pos

            # normalize
            unit_vec1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
            unit_vec2 = vec2 / (np.linalg.norm(vec2) + 1e-8)

            dot = np.dot(unit_vec1, unit_vec2)
            cross = np.cross(unit_vec1, unit_vec2)
            angle_rad = np.arccos(np.clip(dot, -1.0, 1.0))
            direction = np.sign(cross)

            # Stability filter for near-edge cases
            # angle_thresh = np.deg2rad(30)
            # if angle_rad < angle_thresh:
            #     direction = 0  
        _dis_2_barricade = min(closest_distance, self.max_detect_dis)
        barricade_state = np.array([closest_dx, closest_dy], dtype=np.float32)
        angle_norm = angle_rad / np.pi
        ########################################################################################

        ########################################################################################
        x_min, x_max = 4, self.x_range - 4
        y_min, y_max = 4, self.y_range - 4
        x, y = agent_pos
        dist_left = x - x_min
        dist_right = x_max - x
        dist_bottom = y - y_min
        dist_top = y_max - y

        state_border = np.array([
            self.sense(dist_left),
            self.sense(dist_right),
            self.sense(dist_bottom),
            self.sense(dist_top)
        ], dtype=np.float32)
        ########################################################################################


        ########################
        # Low-Level Controller State
        state_2[3], state_2[4] = self.target_position[0] - state_2[3], self.target_position[1] - state_2[4]
        self.state_14 = state_2
        ########################

        ########################
        # Relative Positon(Target & World Coordinate)
        if self._pos_normalize:
            near_threshold = self.max_detect_dis + self.window_r/2
            # === Extra fine-grained relative position ===
            if d < near_threshold:
                rel_x = (self.target_position[0] - state_1[3])
                rel_y = (self.target_position[1] - state_1[4])
                near_flag = 1.0
            else:
                rel_x = 0.0
                rel_y = 0.0
                near_flag = 0.0
            state_1[3], state_1[4] = (self.target_position[0] - state_1[3])/near_threshold, (self.target_position[1] - state_1[4])/near_threshold
        else:
            state_1[3], state_1[4] = (self.target_position[0] - state_1[3]), (self.target_position[1] - state_1[4])
        ########################

        ########################
        if self._pos_normalize:
            near_threshold = self.max_detect_dis + self.window_r/2
            # === Extra fine-grained relative position ===
            if d < near_threshold:
                rel_x = (self.target_position[0] - state_1[3])
                rel_y = (self.target_position[1] - state_1[4])
                near_flag = 1
            else:
                rel_x = 0.0
                rel_y = 0.0
                near_flag = 0
        ########################

        self.d_2_target = d
        self.d_2_target_min = d
        self.old_angle = angle_norm
        self.old_direction = direction
        self._old_dis_2_barricade = _dis_2_barricade

        if self.observation_dim == 14:
            # only position
            self.state = state_1
        if self.observation_dim == 16:
            # barricate sensor
            self.state = np.hstack([state_1, barricade_state]).astype(np.float32)
        if self.observation_dim == 18:
            # roll direction guidance
            if self._obstacle_avoid:
                self.state = np.hstack([state_1, barricade_state, direction, angle_norm]).astype(np.float32)
            else:
                self.state = np.hstack([state_1, state_border]).astype(np.float32)
        if self.observation_dim == 20:
            # edge sensor + barricate sensor
            self.state = np.hstack([state_1, barricade_state, state_border]).astype(np.float32)
        if self.observation_dim == 22:
            # edge sensor
            self.state = np.hstack([state_1, barricade_state, direction, angle_norm, state_border]).astype(np.float32)
        if self.observation_dim == 25:
            # near target sensor
            self.state = np.hstack([state_1, barricade_state, direction, angle_norm, state_border, rel_x, rel_y, near_flag]).astype(np.float32)
        
        if self.include_flow:
            self.u_flow = v_x
            self.v_flow = v_y
            # Plot Flow
            if self._plot_flow:
                self.agent_pos_history.append(self.agent_pos)
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
            if distance <= (a + radius):
                return True
        return False
    
    # def is_out_of_bounds(self):
    #     # pseudo-boundary
    #     _delta = 4
    #     x_c, y_c = self.agent_pos
    #     a = self.ellipse_a
    #     b = self.ellipse_b
    #     # anticlockwise
    #     theta = self.state[5]
    #     # agent coordinate
    #     vertices = [(a, 0), (-a, 0), (0, b), (0, -b)]
    #     # world coordinate
    #     rotated_vertices = []
    #     for vx, vy in vertices:
    #         x_rot = x_c + vx * np.cos(theta) - vy * np.sin(theta)
    #         y_rot = y_c + vx * np.sin(theta) + vy * np.cos(theta)
    #         rotated_vertices.append((x_rot, y_rot))
    #     for x, y in rotated_vertices:
    #         if not (_delta <= x <= self.x_range - _delta and _delta <= y <= self.y_range - _delta):
    #             print(f"Out of bounds: Agent_pos {x_c, y_c}; Angle {theta}")
    #             print(f"Steps: {self.step_counter}")
    #             return True
    #     return False

    def is_out_of_bounds(self):
        _delta = 4
        x, y = self.agent_pos
        if not (_delta <= x <= self.x_range - _delta and _delta <= y <= self.y_range - _delta):
            print(f"Out of bounds: Agent_pos ({x:.2f}, {y:.2f})")
            # print(f"Steps: {self.step_counter}")
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
        r = 12
        if self._is_test:
            if (x_prime**2 / a**2 + y_prime**2 / b**2) <= 1:
                print(f"Success Steps: {self.step_counter}")
            # dx < a and dy < b
            return (x_prime**2 / a**2 + y_prime**2 / b**2) <= 1
        return dx**2 + dy**2 <= r**2

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
        current_target = self.target_position_lc
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
        # Current Target
        ax.scatter(*current_target, color='red', label='Low Level Target Point', zorder=5)
        # Cylinder
        for circle in circles:
            ax.add_patch(plt.Circle(circle["center"], circle["radius"], color='red', alpha=0.3))
            # ax.add_patch(plt.Circle(circle["center"], self.max_detect_dis + self.window_r/2, color='red', fill=False, linestyle='--'))

        agent_x, agent_y = self.agent_pos
        radius = self.max_detect_dis
        fov_angle = 180

        angle_deg = np.degrees(self.angle)
        center_angle = (angle_deg + 180) % 360
        start_angle = (center_angle - fov_angle / 2) % 360
        end_angle = (center_angle + fov_angle / 2) % 360
        wedge = patches.Wedge(
            center=(agent_x, agent_y),
            r=radius,
            theta1=start_angle,
            theta2=end_angle,
            facecolor='orange',
            alpha=0.2
        )
        ax.add_patch(wedge)
        
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

        ax_local, ay_local = self.action[0], self.action[1]
        # combine
        global_action_x = ax_local * cos_angle + ay_local * (-sin_angle)
        global_action_y = ax_local * sin_angle + ay_local * cos_angle
        ax.quiver(agent_pos[0], agent_pos[1], global_action_x/2, global_action_y/2,
                angles='xy', scale_units='xy', scale=1, width=0.002, headwidth=2, color='blue', label="Action")

        # Axis
        ax.set_xlim(0, x_range)
        ax.set_ylim(0, y_range)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        ax.set_title("Moving Trajectory")
