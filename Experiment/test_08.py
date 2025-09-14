import gym
import csv
import pandas as pd
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from termcolor import colored
import time
import torch.nn as nn
from math import *
from collections import deque


import my_ppo_net_1
from my_ppo_net_1 import Classic_PPO
import hjbppo
from hjbppo import HJB_PPO
import icm_ppo
from icm_ppo import ICM_PPO


# Env
from env_new import train_env_upper_2
from env_new import train_env_upper_2_2

save_dir = './SuccessTest'
save_dir_1 = './model_output'
# plot_save_dir = './model_output/success_plot_withp.png'


####### DEVICE #######
device = my_ppo_net_1.device
# device = hjbppo.device
# device = icm_ppo.device

####### PREDICTOR #######
class Seq2SeqPredictor(nn.Module):
    def __init__(self, input_dim=17, condition_dim=3, output_dim=14, hidden_size=128, lstm_layers=2, pred_steps=5, teacher_forcing=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.pred_steps = pred_steps
        self.output_dim = output_dim
        self.teacher_forcing = teacher_forcing

        # Encoder: 处理 (pressure + vel + acc)
        self.encoder_input_fc = nn.Linear(input_dim, hidden_size)
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Decoder: 输入 (pred_pressure_prev + future_acc)
        self.decoder_input_fc = nn.Linear(output_dim + condition_dim, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.decoder_output_fc = nn.Linear(hidden_size, output_dim)

        self._initialize_weights()

    def forward(self, past_pressure, past_agent, future_acc, target_state=None):
        """
        past_pressure: (B, T_in, 8)
        past_agent(pos speed acc): (B, T_in, 9)
        future_acc: (B, T_pred, 3)
        target_state(pos speed pressure): (B, T_pred, 14) for teacher forcing
        """
        B, T_in, _ = past_pressure.shape
        T_pred = future_acc.shape[1]

        # ===== Encoder =====
        enc_input = torch.cat([past_agent, past_pressure], dim=-1)  # (B, T_in, 17)
        enc_input = self.encoder_input_fc(enc_input)                # (B, T_in, hidden_size)
        h0 = torch.zeros(self.lstm_layers, B, self.hidden_size, device=past_pressure.device)
        c0 = torch.zeros(self.lstm_layers, B, self.hidden_size, device=past_pressure.device)
        _, (h_n, c_n) = self.encoder_lstm(enc_input, (h0, c0))

        # ===== Decoder =====
        preds = []
        decoder_input = torch.zeros(B, self.output_dim, device=past_pressure.device)  # 初始输入为零向量

        h_dec, c_dec = h_n, c_n
        for t in range(T_pred):
            cond_acc = future_acc[:, t, :]                                        # (B, 3)
            dec_input = torch.cat([decoder_input, cond_acc], dim=-1)              # (B, 14+3)
            dec_input = self.decoder_input_fc(dec_input).unsqueeze(1)             # (B, 1, hidden_size)

            dec_output, (h_dec, c_dec) = self.decoder_lstm(dec_input, (h_dec, c_dec))
            pred_p = self.decoder_output_fc(dec_output.squeeze(1))                # (B, 14)

            preds.append(pred_p.unsqueeze(1))                                     # [(B, 1, 14), ...]

            if self.teacher_forcing and self.training and target_state is not None:
                decoder_input = target_state[:, t, :]
            else:
                decoder_input = pred_p

        preds = torch.cat(preds, dim=1)                                           # (B, T_pred, 14)
        return preds

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)


def load_model(model_path, hidden_size, num_layers, output_dim, pred_steps, device):
    model = Seq2SeqPredictor(input_dim=17, condition_dim=3, output_dim=14, hidden_size=hidden_size, lstm_layers=num_layers, pred_steps=pred_steps, teacher_forcing=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

def predict_pressure(p_model, input_window, other_window, future_action_seq, device):
    input_tensor = torch.from_numpy(np.stack(input_window, axis=0)).unsqueeze(0).to(dtype=torch.float32, device=device)  # (1, 5, 8)
    other_tensor = torch.from_numpy(np.stack(other_window, axis=0)).unsqueeze(0).to(dtype=torch.float32, device=device)  # (1, 5, 9)
    future_tensor = torch.from_numpy(future_action_seq).unsqueeze(0).to(dtype=torch.float32, device=device)             # (1, 5, 3)
    with torch.no_grad():
        outputs = p_model(input_tensor, other_tensor, future_tensor).cpu().numpy() # (1, 5, 14)
    return outputs

def get_future_pressure_batch(p_model, input_window, other_window, action_window, device, target, norm_range, _pos_norm=True):
    """
    Generate future pressure information using the flow prediction model.

    Args:
        p_model: The pressure prediction model.
        input_window: deque or list of len 10, each with shape (8,) (pressure).
        other_window: deque or list of len 10, each with shape (9,) (pos + speed + acc).
        action_window: list of len 10, each with shape (8, 10, 3) (acc).
        device: torch.device
        target: np.ndarray, shape (2,) or (3,), target position (x, y[, angle])
        norm_range: float, normalization denominator for delta computation.
        _pos_norm: whether predicted pos is normalized (True means we need to denormalize before computing delta).

    Returns:
        _future_state: np.ndarray of shape (8, 14), each row: [pos + speed + pressure] with normalized (target - pos)
    """
    _future_state = []
    for i in range(len(action_window)):
        future_action_seq = action_window[i]  # shape: (8, 10, 3)
        outputs = predict_pressure(p_model, input_window, other_window, future_action_seq, device)  # shape: (1, T_pred, out_dim)

        last_state = outputs[0, -1, :].copy()  # avoid in-place modification if outputs reused

        if _pos_norm:
            pos_x = last_state[0] * 160.0
            pos_y = last_state[1] * 60.0
        else:
            pos_x = last_state[0]
            pos_y = last_state[1]

        last_state[0] = (target[0] - pos_x) / norm_range
        last_state[1] = (target[1] - pos_y) / norm_range

        _future_state.append(last_state)

    _future_state = np.stack(_future_state, axis=0)  # (8, 14)
    return _future_state

def normalize_position(pos, scale=(160.0, 60.0)):
    pos = pos.copy()
    pos[:2] /= scale
    return pos

def generate_fixed_action_window(acc_norm, length=10, num_dirs=8):
    acc = 1.0 if acc_norm else 15.0
    angles = [i * np.pi / num_dirs * 2 for i in range(num_dirs)]
    window = []
    for theta in angles:
        a_x = acc * np.cos(theta)
        a_y = acc * np.sin(theta)
        a_w = 0.0
        action_seq = [[a_x, a_y, a_w] for _ in range(length)]
        window.append(action_seq)
    return np.array(window)  # Shape: (num_dirs, length, 3)

def build_current_other(position, speed, action):
    return np.concatenate([position, speed, action], axis=-1)

####### HRL LOW #######
def compute_subgoal_forward(pos, angle, r):
    # print(f"target angle:{angle.item()}")
    x, y = pos[0], pos[1]
    angle_1 = angle.item() + np.pi/2
    dx = r * np.cos(angle_1)
    dy = r * np.sin(angle_1)
    return x + dx, y + dy

def compute_subgoal(pos, angle, r):
    x, y = pos[0], pos[1]
    angle_1 = angle.item()
    dx = r * np.cos(angle_1)
    dy = r * np.sin(angle_1)
    return x + dx, y + dy

def subgoal_reached(pos, target, threshold=4.0):
    dx = pos[0] - target[0]
    dy = pos[1] - target[1]
    return np.sqrt(dx**2 + dy**2) < threshold

# def compute_relative_subgoal(angle, r):
#     angle_1 = angle + np.pi/2
#     dx = r * np.cos(angle_1)
#     dy = r * np.sin(angle_1)
#     return dx.item(), dy.item()


def test():
    ########################### Mode 0: Single Test ###########################
    print("============================================================================================")
    ########################### Environmenrt ###########################

    ################ Predict Model ################
    _pos_norm = True
    _acc_norm = False
    _expand_pressure = False
    # p_model_path = "./p_predict/0731_id15.pth"
    p_model_path = "./p_predict/0801_id16.pth"
    p_hidden_size = 256
    p_num_layers = 2
    p_output_dim = 14
    p_pred_steps = 10
    p_model = load_model(p_model_path, p_hidden_size, p_num_layers, p_output_dim, p_pred_steps, device)
    #####################################################  

    ################ Lilypad Settings ################
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10
    #####################################################  

    ################ RL Hyperparameters ################
    # action mode
    has_continuous_action_space = True
    # update epochs
    K_epochs = 40
    # clip rate for PPO
    eps_clip = 0.2
    # discount factor γ
    gamma = 0.99  
    # learning rate for actor network
    lr_actor = 0.0001
    # learning rate for critic network  
    lr_critic = 0.0002
    # set std for action distribution when testing
    action_std_4_test = 0.04  
    #####################################################  

    ################ Env Settings ################
    ### Avoid Mode ###
    _is_avoid = True
    # _is_avoid = False

    ### State Dim ###
    # state_dim_mode = 22
    state_dim_mode = 18
    # state_dim_mode = 16
    # state_dim_mode = 14
    # state_dim_mode = 6
    lc_state_dim = 14
    # lc_state_dim = 6

    ### Steps ###
    max_steps = 500
    max_ep_len = max_steps + 20
    max_ll_steps = 5
    #####################################################  

    
    ################ Other Settings ################
    ### set flow seed if required (0 : random flow; else : fixed flow) ###
    flow_seed = 0
    # flow_seed = 1

    ### set heading mode(True : forward only; else : all direction) ###
    # _is_head = True
    _is_head = False


    ### predict & pre-trained high-level controller ###
    # _is_predict = True
    # ppo_path = './models/08/pre_pos_dim14_test_50.pth'
    # ppo_path = './models/08/pre_pos_dim16_10.pth'

    _is_predict = False
    ppo_path = './models/exp/hrl_dim18_guide_51025.pth'
    # ppo_path = './models/09/8315_max.pth'
    # ppo_path = './models/09/hrl_dim2_a15_noobs_notest.pth'
    # ppo_path = './models/09/hrl_dim2_a15_normal_test.pth'
    ### pre-trained low-level controller ###
    # if _is_head:
    #     ppo_path_1 = './models/easy_task/0704_seed1.pth' # forward
    # else:
    #     ppo_path_1 = './models/easy_task/0705_lc.pth' # all direction
    # ppo_path_1 = './models/easy_task/0705_lc.pth' # all direction
    ppo_path_1 = './models/exp/lc_dim14.pth'
    # ppo_path_1 = './models/exp/lc_dim6.pth'

    ### set random mode(True : real random mode; else : random.seed(random_seed)) ###
    true_random = True
    if true_random:
        random_seed = 0
    else:
        random_seed = 1
    
    test_mode = True
    switch_range = 1
    # test_mode = False
    # switch_range = 12

    # theta_mode = True
    theta_mode = False

    # switch_mode = True
    switch_mode = False

    _is_normalize = True
    # _is_normalize = False

    ### Obstacle Mode ###
    _is_near = True
    # _is_near = False

    video_dir = "./Video_roll/video_frames"
    # video_dir = "./Video_normal/video_frames"

    speed_range = 15.0
    speed_clip = 15.0
    #####################################################


    # random point
    # _start = np.array([float(160), float(64)])
    _start = np.array([float(220), float(64)])
    _target = np.array([float(64), float(64)])
    random_range = 56
    # _start = np.array([float(240), float(64)])
    # _target = np.array([float(160), float(64)])
    # random_range = 32

    # _start = np.array([float(200), float(36)])
    # _target = np.array([float(200), float(88)])
    # random_range = 28
    # env = train_env_upper_2.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, _include_flow=True, _plot_flow=True, _proccess_flow=False, _random_range=random_range, _init_flow_num=flow_seed, 
    # _pos_normalize=_is_normalize, _is_test = test_mode, _state_dim=state_dim_mode, 
    # _is_random = true_random, _set_random=random_seed, _swich_range = switch_range, u_range=20.0, v_range=20.0, 
    # _forward_only=_is_head, _obstacle_avoid=_is_avoid, _theta_mode = theta_mode, _is_switch=switch_mode, u_clip=15.0, v_clip=15.0, _test_info=False, video_dir="./Video_a20_dim1/video_frames", _obstacle_mode=_is_near)

    env = train_env_upper_2.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, _include_flow=True, _plot_flow=True, _proccess_flow=False, _random_range=random_range, _init_flow_num=flow_seed, 
    _pos_normalize=_is_normalize, _is_test = test_mode, _state_dim=state_dim_mode, 
    _is_random = true_random, _set_random=random_seed, _switch_range = switch_range, u_range=speed_range, v_range=speed_range, 
    _forward_only=_is_head, _obstacle_avoid=_is_avoid, _theta_mode = theta_mode, _is_switch=switch_mode, u_clip=speed_clip, v_clip=speed_clip, _test_info=False, video_dir=video_dir, _obstacle_mode=_is_near)


    # fixed point
    # _start = np.array([float(173), float(81)])
    # _target = np.array([float(19.3), float(33.3)])
    # _start = np.array([float(220), float(64)])
    # _target = np.array([float(190), float(56)])
    # random_range = 56
    # _start = np.array([float(236.41341788), float(77.02539591)])
    # _target = np.array([float(154.23059141), float(77.74279773)])
    # random_range = 32
    # env = train_env_upper_2.foil_env(args_1, max_step=max_steps, start_position=_start, target_position=_target, _include_flow=True, _plot_flow=True, _proccess_flow=False, _random_range=random_range, _init_flow_num=flow_seed, 
    # _pos_normalize=_is_normalize, _is_test = test_mode, _state_dim=state_dim_mode, 
    # _is_random = true_random, _set_random=random_seed, _swich_range = switch_range, u_range=20.0, v_range=20.0, 
    # _forward_only=_is_head, _obstacle_avoid=_is_avoid, _theta_mode = theta_mode, _is_switch=switch_mode, u_clip=20.0, v_clip=20.0, _test_info=False, video_dir=video_dir, _obstacle_mode=_is_near)


    ################ Action Space Settings ################
    ### High-level Controller ###
    state_dim = env.observation_space.shape[0]
    expand_dim = state_dim
    if _is_predict:
        if _expand_pressure:
            expand_dim += 8 * 14
        else:
            expand_dim += 8 * 6
    action_output_scale = np.array([])
    action_output_bias = np.array([])
    if has_continuous_action_space:
        # action space dimension
        action_dim = env.action_space.shape[0]
        action_output_scale = (env.action_space.high - env.action_space.low) / 2.0
        action_output_bias = (env.action_space.high + env.action_space.low) / 2.0
    else:
        action_dim = env.action_space.n
    
    ### Low-level Controller ###
    lc_action_output_scale = np.array([])
    lc_action_output_bias = np.array([])
    if has_continuous_action_space:
        # low-level controller action space dimension
        lc_action_dim = env.lc_action_space.shape[0]
        lc_action_output_scale = (env.lc_action_space.high - env.lc_action_space.low) / 2.0
        lc_action_output_bias = (env.lc_action_space.high + env.lc_action_space.low) / 2.0
    else:
        lc_action_dim = env.lc_action_space.n
    ######################################################

    ########################### PPO ###########################
    # initialize RL agent
    # ppo_agent = ICM_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
    #                     has_continuous_action_space,
    #                     action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
    #                     continuous_action_output_bias=action_output_bias, icm_alpha=200)
    # ppo_agent.load_full_icm(ppo_path)
    ppo_agent = Classic_PPO(expand_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                            action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
                            continuous_action_output_bias=action_output_bias)
    ppo_agent.load_full(ppo_path)
    ppo_agent_lc = Classic_PPO(lc_state_dim, lc_action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
            action_std_init=action_std_4_test, continuous_action_output_scale=lc_action_output_scale,
            continuous_action_output_bias=lc_action_output_bias)
    ppo_agent_lc.load_full(ppo_path_1)
    ######################################################

    ########################### Test ###########################
    # Predictor Info
    if _is_predict:
        input_window = deque(maxlen=10)
        other_window = deque(maxlen=10)
        action_window = generate_fixed_action_window(_acc_norm)

    print(colored("**** Forward Mode **** " if _is_head else "**** Normal Mode **** ", 'green'))

    subgoal_threshold = 1
    state, info = env.reset()

    # Initial observation
    if _is_predict:
        pressure_now = np.array([info[f"pressure_{i}"] for i in range(1, 9)], dtype=np.float32)
        speed_now = np.array([info["vel_x"], info["vel_y"], info["vel_angle"]], dtype=np.float32)
        position_now = np.array([info["pos_x"], info["pos_y"], info["angle"]], dtype=np.float32)
        if _pos_norm:
            position_now = normalize_position(position_now)
        initial_action = np.zeros(3, dtype=np.float32)
        current_other = build_current_other(position_now, speed_now, initial_action)
        for _ in range(10):
            input_window.append(pressure_now.copy())
            other_window.append(current_other.copy())

        # Predict future pressure and construct high-level state
        _future_pressure = get_future_pressure_batch(
            p_model, input_window, other_window, action_window,
            device, env.target_position, env.max_detect_dis + env.window_r / 2,
            _pos_norm=_pos_norm
        )
        if not _expand_pressure:
            _future_pressure = _future_pressure[:, :6]
        _expand_state = np.concatenate([state, _future_pressure.flatten()], axis=0)
    else:
        _expand_state = state

    # Store initial states
    state_14 = env.state_14
    agent_position = env.agent_pos
    prev_state = state.copy()
    prev_state_14 = state_14.copy()
    prev_agent_position = agent_position.copy()
    print("position_reset:", agent_position)

    # Main episode loop
    t = 0
    hl_time_step = 0
    done = False

    while not done and t < max_ep_len:
        # === Get current high-level state ===
        high_state = _expand_state

        # === Decide whether to use fallback or normal high-level policy ===
        use_fallback = env.d_2_target < switch_range
        if use_fallback:
            # Compute angle from agent to final target
            vec_to_target = env.target_position - env.agent_pos
            target_angle = np.arctan2(vec_to_target[1], vec_to_target[0])
            print(f"=== Fallback to target mode | d2target = {env.d_2_target:.2f} ===")
        else:
            high_action = ppo_agent.select_action(high_state)
            high_action = np.clip(high_action, env.low, env.high)
            hl_time_step += 1
            print(f"=== High Level Step {hl_time_step}, Action: {high_action} ===")

        # === Compute Subgoal ===
        agent_position = env.agent_pos
        # r = min(np.linalg.norm(env.target_position - agent_position), 3)
        r = min(np.linalg.norm(env.target_position - agent_position), 5)
        if _is_head:
            angle_to_use = target_angle if use_fallback else high_action
            lc_target_x, lc_target_y = compute_subgoal_forward(pos=agent_position, angle=angle_to_use, r=r)
        else:
            if use_fallback:
                angle_to_use = target_angle
            else:
                if not theta_mode:
                    vec = high_action
                    vec = vec / (np.linalg.norm(vec) + 1e-6)
                    angle_to_use = np.arctan2(vec[1], vec[0])
                else:
                    angle_to_use = high_action

            lc_target_x, lc_target_y = compute_subgoal(pos=agent_position, angle=angle_to_use, r=r)
        env.set_lc_target([lc_target_x, lc_target_y])
        print(f"=== Subgoal: ({lc_target_x}, {lc_target_y}) | Angle used: {angle_to_use} ===")
        # print(f"=== Subgoal: ({lc_target_x:.2f}, {lc_target_y:.2f}) | Angle used: {angle_to_use:.2f} ===")

        # === Low-level loop ===
        ll_step = 0
        done_lc = False
        while ll_step < max_ll_steps:
            agent_angle_old = env.angle

            # Low-level state update
            low_state = state_14.copy()
            low_state[3] = lc_target_x - agent_position[0]
            low_state[4] = lc_target_y - agent_position[1]

            if lc_state_dim == 3:
                low_state = low_state[3:6]
            elif lc_state_dim == 6:
                low_state = low_state[:6]
            elif lc_state_dim == 11:
                low_state = np.hstack([low_state[3:6], low_state[6:14]]).astype(np.float32)

            # Low-level action
            low_action = ppo_agent_lc.select_action(low_state)
            cliped_low_action = env.clip_action(low_action)
            print(f"=== Low Level Step {t}, Action: {low_action}")
            print(f"===               -> Clip_Action: {cliped_low_action} ===")

            # Sliding window update (for pressure prediction)
            if _is_predict:
                action_input = cliped_low_action / 15.0 if _acc_norm else cliped_low_action
                current_other = build_current_other(position_now, speed_now, action_input)
                input_window.append(pressure_now.copy())
                other_window.append(current_other.copy())

            # Step env
            state, reward, terminated, truncated, info = env.step(low_action)
            done = terminated or truncated

            if _is_predict:
                pressure_now = np.array([info[f"pressure_{i}"] for i in range(1, 9)], dtype=np.float32)
                speed_now = np.array([info["vel_x"], info["vel_y"], info["vel_angle"]], dtype=np.float32)
                position_now = np.array([info["pos_x"], info["pos_y"], info["angle"]], dtype=np.float32)
                if _pos_norm:
                    position_now = normalize_position(position_now)

            agent_position = env.agent_pos
            state_14 = env.state_14

            # Check subgoal success
            dist_to_subgoal = np.linalg.norm(agent_position - np.array([lc_target_x, lc_target_y]))
            if dist_to_subgoal < subgoal_threshold or done or ll_step + 1 >= max_ll_steps:
                done_lc = True

            t += 1
            ll_step += 1

            if done_lc:
                if _is_predict:
                    _future_pressure = get_future_pressure_batch(
                        p_model, input_window, other_window, action_window,
                        device, env.target_position, env.max_detect_dis + env.window_r / 2,
                        _pos_norm=_pos_norm
                    )
                    if not _expand_pressure:
                        _future_pressure = _future_pressure[:, :6]
                    _expand_state = np.concatenate([state, _future_pressure.flatten()], axis=0)
                else:
                    _expand_state = state
                ppo_agent_lc.buffer.clear()
                break

        ppo_agent.buffer.clear()

    env.close()
    print("Total_Steps:", t)


    # while not done and t < max_ep_len:
    #     high_state = _expand_state
    #     high_action = ppo_agent.select_action(high_state)
    #     high_action = np.clip(high_action, env.low, env.high)
    #     hl_time_step += 1
    #     print(f"=== High Level Step {hl_time_step}, Action: {high_action} ===")

    #     # Compute subgoal
    #     agent_position = env.agent_pos
    #     if _is_head:
    #         lc_target_x, lc_target_y = compute_subgoal_forward(pos=agent_position, angle=high_action, r=5)
    #     else:
    #         if not theta_mode:
    #             vec = high_action
    #             vec = vec / (np.linalg.norm(vec) + 1e-6)
    #             high_action = np.arctan2(vec[1], vec[0])
    #         lc_target_x, lc_target_y = compute_subgoal(pos=agent_position, angle=high_action, r=5)
    #     env.set_lc_target([lc_target_x, lc_target_y])
    #     # Low-level control loop
    #     ll_step = 0
    #     done_lc = False
    #     while ll_step < max_ll_steps:
    #         agent_angle_old = env.angle

    #         # Build low-level state
    #         low_state = state_14.copy()
    #         low_state[3] = lc_target_x - agent_position[0]
    #         low_state[4] = lc_target_y - agent_position[1]

    #         # Select low-level action
    #         low_action = ppo_agent_lc.select_action(low_state)
    #         cliped_low_action = env.clip_action(low_action)
    #         print(f"=== Low Level Step {t}, Action: {low_action}")
    #         print(f"===               ->  Clip_Action: {cliped_low_action} ===")

    #         # Update sliding window
    #         if _is_predict:
    #             action_input = cliped_low_action / 15.0 if _acc_norm else cliped_low_action
    #             current_other = build_current_other(position_now, speed_now, action_input)
    #             input_window.append(pressure_now.copy())
    #             other_window.append(current_other.copy())

    #         # Step the environment
    #         state, reward, terminated, truncated, info = env.step(low_action)
    #         done = terminated or truncated

    #         if _is_predict:
    #             pressure_now = np.array([info[f"pressure_{i}"] for i in range(1, 9)], dtype=np.float32)
    #             speed_now = np.array([info["vel_x"], info["vel_y"], info["vel_angle"]], dtype=np.float32)
    #             position_now = np.array([info["pos_x"], info["pos_y"], info["angle"]], dtype=np.float32)
    #             if _pos_norm:
    #                 position_now = normalize_position(position_now)

    #         agent_position = env.agent_pos
    #         state_14 = env.state_14

    #         # Check subgoal success
    #         dist_to_subgoal = np.linalg.norm(agent_position - np.array([lc_target_x, lc_target_y]))
    #         if dist_to_subgoal < subgoal_threshold or done or ll_step + 1 >= max_ll_steps:
    #             done_lc = True

    #         t += 1
    #         ll_step += 1

    #         if done_lc:
    #             if _is_predict:
    #                 _future_pressure = get_future_pressure_batch(
    #                     p_model, input_window, other_window, action_window,
    #                     device, env.target_position, env.max_detect_dis + env.window_r / 2,
    #                     _pos_norm=_pos_norm
    #                 )
    #                 if not _expand_pressure:
    #                     _future_pressure = _future_pressure[:, :6]
    #                 _expand_state = np.concatenate([state, _future_pressure.flatten()], axis=0)
    #             else:
    #                 _expand_state = state
    #             ppo_agent_lc.buffer.clear()
    #             break

    #     ppo_agent.buffer.clear()

    # env.close()
    # print("Total_Steps:", t)
    ##############################################################################



def success_rate_test(_id_in=1):
    ########################### Mode 1: Multi-Test ###########################
    print("============================================================================================")
    ################ Predict Model ################
    _pos_norm = True
    _acc_norm = False
    _expand_pressure = False
    # p_model_path = "./p_predict/0731_id15.pth"
    p_model_path = "./p_predict/0801_id16.pth"
    p_hidden_size = 256
    p_num_layers = 2
    p_output_dim = 14
    p_pred_steps = 10
    p_model = load_model(p_model_path, p_hidden_size, p_num_layers, p_output_dim, p_pred_steps, device)
    #####################################################  

    ################ Lilypad Settings ################
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10
    #####################################################  

    ################ RL Hyperparameters ################
    # action mode
    has_continuous_action_space = True
    # update epochs
    K_epochs = 40
    # clip rate for PPO
    eps_clip = 0.2
    # discount factor γ
    gamma = 0.99  
    # learning rate for actor network
    lr_actor = 0.0001
    # learning rate for critic network  
    lr_critic = 0.0002
    # set std for action distribution when testing
    action_std_4_test = 0.04  
    #####################################################  

    ################ Env Settings ################
    ### Obstacles Mode ###
    # _is_avoid = True
    _is_avoid = False

    ### State Dim ###
    # state_dim_mode = 22
    # state_dim_mode = 16
    state_dim_mode = 14
    # state_dim_mode = 6

    ### Steps ###
    max_steps = 300
    max_ep_len = max_steps + 20
    max_ll_steps = 5
    #####################################################  

    
    ################ Other Settings ################
    ### set flow seed if required (0 : random flow; else : fixed flow) ###
    # flow_seed = 0
    flow_seed = 20

    ### set heading mode(True : forward only; else : all direction) ###
    # _is_head = True
    _is_head = False


    ### predict & pre-trained high-level controller ###
    # _is_predict = True
    # ppo_path = './models/08/pre_pos_dim14_test_50.pth'
    # ppo_path = './models/08/pre_pos_dim16_10.pth'

    _is_predict = False
    # ppo_path = './models/08/dim6_test_0.pth'
    ppo_path = './models/08/dim14_test_50.pth'
    # ppo_path = './models/08/dim16_3.pth'

    ### pre-trained low-level controller ###
    if _is_head:
        ppo_path_1 = './models/easy_task/0704_seed1.pth' # forward
    else:
        ppo_path_1 = './models/easy_task/0705_lc.pth' # all direction

    ### set random mode(True : real random mode; else : random.seed(random_seed)) ###
    true_random = False
    if true_random:
        random_seed = 0
    else:
        random_seed = 1
    
    # test_mode = True
    # switch_range = 1
    test_mode = False
    switch_range = 12

    # theta_mode = True
    theta_mode = False

    # switch_mode = True
    switch_mode = False

    _is_normalize = True
    # _is_normalize = False
    #####################################################


    # random point
    # _start = np.array([float(220), float(64)])
    # _target = np.array([float(64), float(64)])
    # random_range = 56
    _start = np.array([float(240), float(64)])
    _target = np.array([float(160), float(64)])
    random_range = 32
    env = train_env_upper_2.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, _include_flow=False, _plot_flow=False, _proccess_flow=False, _random_range=random_range, _init_flow_num=flow_seed, 
    _pos_normalize=_is_normalize, _is_test = test_mode, _state_dim=state_dim_mode, 
    _is_random = true_random, _set_random=random_seed, _swich_range = switch_range, u_range=20.0, v_range=20.0, 
    _forward_only=_is_head, _obstacle_avoid=_is_avoid, _theta_mode = theta_mode, _is_switch=switch_mode, u_clip=15.0, v_clip=15.0, _test_info=True)


    # fixed point
    # _start = np.array([float(256.71), float(85.69)])
    # _target = np.array([float(131.76), float(67.41)])
    # random_range = 32
    # env = train_env_upper_2.foil_env(args_1, max_step=max_steps, start_position=_start, target_position=_target, _include_flow=True, _plot_flow=True, _proccess_flow=False, _random_range=random_range, _init_flow_num=flow_seed, 
    # _pos_normalize=_is_normalize, _is_test = test_mode, _state_dim=state_dim_mode, 
    # _is_random = true_random, _set_random=random_seed, _swich_range = switch_range, u_range=20.0, v_range=20.0, 
    # _forward_only=_is_head, _obstacle_avoid=_is_avoid, _theta_mode = theta_mode, _is_switch=switch_mode, u_clip=15.0, v_clip=15.0)


    ################ Action Space Settings ################
    ### High-level Controller ###
    state_dim = env.observation_space.shape[0]
    expand_dim = state_dim
    if _is_predict:
        if _expand_pressure:
            expand_dim += 8 * 14
        else:
            expand_dim += 8 * 6
    action_output_scale = np.array([])
    action_output_bias = np.array([])
    if has_continuous_action_space:
        # action space dimension
        action_dim = env.action_space.shape[0]
        action_output_scale = (env.action_space.high - env.action_space.low) / 2.0
        action_output_bias = (env.action_space.high + env.action_space.low) / 2.0
    else:
        action_dim = env.action_space.n
    
    ### Low-level Controller ###
    lc_state_dim = 14
    lc_action_output_scale = np.array([])
    lc_action_output_bias = np.array([])
    if has_continuous_action_space:
        # low-level controller action space dimension
        lc_action_dim = env.lc_action_space.shape[0]
        lc_action_output_scale = (env.lc_action_space.high - env.lc_action_space.low) / 2.0
        lc_action_output_bias = (env.lc_action_space.high + env.lc_action_space.low) / 2.0
    else:
        lc_action_dim = env.lc_action_space.n
    ######################################################

    ########################### PPO ###########################
    # initialize RL agent
    # ppo_agent = ICM_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
    #                     has_continuous_action_space,
    #                     action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
    #                     continuous_action_output_bias=action_output_bias, icm_alpha=200)
    # ppo_agent.load_full_icm(ppo_path)
    ppo_agent = Classic_PPO(expand_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                            action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
                            continuous_action_output_bias=action_output_bias)
    ppo_agent.load_full(ppo_path)
    ppo_agent_lc = Classic_PPO(lc_state_dim, lc_action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
            action_std_init=action_std_4_test, continuous_action_output_scale=lc_action_output_scale,
            continuous_action_output_bias=lc_action_output_bias)
    ppo_agent_lc.load_full(ppo_path_1)
    ######################################################


    ########################### Test ###########################
    print(colored("**** Forward Mode **** " if _is_head else "**** Normal Mode **** ", 'green'))
    success_count = 0
    success_steps = []

    fail_out_bound = 0
    fail_timeout = 0
    fail_collide = 0

    record_rows = []
    num_trials = 1000

    for trial in range(num_trials):
        print(colored(f"\n================ Trial {trial + 1} ================\n", 'cyan'))
        if _is_predict:
            input_window = deque(maxlen=10)
            other_window = deque(maxlen=10)
            action_window = generate_fixed_action_window(_acc_norm)
        subgoal_threshold = 1
        state, info = env.reset()
        start_x, start_y = env.agent_pos
        target_x, target_y = env.target_position

        if _is_predict:
            pressure_now = np.array([info[f"pressure_{i}"] for i in range(1, 9)], dtype=np.float32)
            speed_now = np.array([info["vel_x"], info["vel_y"], info["vel_angle"]], dtype=np.float32)
            position_now = np.array([info["pos_x"], info["pos_y"], info["angle"]], dtype=np.float32)
            if _pos_norm:
                position_now = normalize_position(position_now)
            initial_action = np.zeros(3, dtype=np.float32)
            current_other = build_current_other(position_now, speed_now, initial_action)
            for _ in range(10):
                input_window.append(pressure_now.copy())
                other_window.append(current_other.copy())

            _future_pressure = get_future_pressure_batch( p_model, input_window, other_window, action_window, device, env.target_position, env.max_detect_dis + env.window_r / 2, _pos_norm=_pos_norm)
            if not _expand_pressure:
                _future_pressure = _future_pressure[:, :6]
            _expand_state = np.concatenate([state, _future_pressure.flatten()], axis=0)
        else:
            _expand_state = state

        state_14 = env.state_14
        agent_position = env.agent_pos
        t = 0
        hl_time_step = 0
        done = False
        _terminated = _out_bound = _time_out = _collide = False  # init

        while not done and t < max_ep_len:
            high_state = _expand_state
            high_action = ppo_agent.select_action(high_state)
            high_action = np.clip(high_action, env.low, env.high)
            hl_time_step += 1
            # print(f"=== High Level Step {hl_time_step}, Action: {high_action} ===")
            agent_position = env.agent_pos
            if _is_head:
                lc_target_x, lc_target_y = compute_subgoal_forward(pos=agent_position, angle=high_action, r=5)
            else:
                if not theta_mode:
                    vec = high_action
                    vec = vec / (np.linalg.norm(vec) + 1e-6)
                    high_action = np.arctan2(vec[1], vec[0])
                lc_target_x, lc_target_y = compute_subgoal(pos=agent_position, angle=high_action, r=5)
            env.set_lc_target([lc_target_x, lc_target_y])

            ll_step = 0
            done_lc = False
            while ll_step < max_ll_steps:
                low_state = state_14.copy()
                low_state[3] = lc_target_x - agent_position[0]
                low_state[4] = lc_target_y - agent_position[1]

                low_action = ppo_agent_lc.select_action(low_state)
                cliped_low_action = env.clip_action(low_action)
                # print(f"=== Low Level Step {t}, Action: {low_action}")
                # print(f"===               ->  Clip_Action: {cliped_low_action} ===")

                if _is_predict:
                    action_input = cliped_low_action / 15.0 if _acc_norm else cliped_low_action
                    current_other = build_current_other(position_now, speed_now, action_input)
                    input_window.append(pressure_now.copy())
                    other_window.append(current_other.copy())

                # Step the environment
                state, reward, _terminated, _out_bound, _time_out, _collide, info = env.step(low_action)
                done = _terminated or _out_bound or _time_out or _collide

                if _is_predict:
                    pressure_now = np.array([info[f"pressure_{i}"] for i in range(1, 9)], dtype=np.float32)
                    speed_now = np.array([info["vel_x"], info["vel_y"], info["vel_angle"]], dtype=np.float32)
                    position_now = np.array([info["pos_x"], info["pos_y"], info["angle"]], dtype=np.float32)
                    if _pos_norm:
                        position_now = normalize_position(position_now)

                agent_position = env.agent_pos
                state_14 = env.state_14

                dist_to_subgoal = np.linalg.norm(agent_position - np.array([lc_target_x, lc_target_y]))
                if dist_to_subgoal < subgoal_threshold or done or ll_step + 1 >= max_ll_steps:
                    done_lc = True

                t += 1
                ll_step += 1

                if done_lc:
                    if _is_predict:
                        _future_pressure = get_future_pressure_batch(p_model, input_window, other_window, action_window, device, env.target_position, env.max_detect_dis + env.window_r / 2, _pos_norm=_pos_norm)
                        if not _expand_pressure:
                            _future_pressure = _future_pressure[:, :6]
                        _expand_state = np.concatenate([state, _future_pressure.flatten()], axis=0)
                    else:
                        _expand_state = state
                    ppo_agent_lc.buffer.clear()
                    break

            ppo_agent.buffer.clear()

        end_x, end_y = env.agent_pos
        env.close()
        print("Total_Steps:", t)
        # 记录终止类型和数据
        if _terminated:
            print(colored(f"Episode{trial + 1} success", 'green'))
            result = "success"
            success_count += 1
            success_steps.append(t)
        elif _out_bound:
            print(colored(f"Episode{trial + 1} outbound", 'red'))
            result = "out_bound"
            fail_out_bound += 1
        elif _time_out:
            print(colored(f"Episode{trial + 1} timeout", 'blue'))
            result = "timeout"
            fail_timeout += 1
        elif _collide:
            print(colored(f"Episode{trial + 1} collide", 'yellow'))
            result = "collision"
            fail_collide += 1
        else:
            result = "unknown"

        record_rows.append({
            "trial_id": trial + 1,
            "result_type": result,
            "start_x": start_x,
            "start_y": start_y,
            "target_x": target_x,
            "target_y": target_y,
            "end_x": end_x,
            "end_y": end_y,
            "total_steps": t
        })

    # === Final summary ===
    print("\n" + colored("========== Evaluation Results ==========", "cyan"))
    print(f"Total Trials: {num_trials}")
    print(colored(f"Success Count: {success_count}", "green"))
    print(colored(f"Success Rate: {success_count / num_trials * 100:.2f}%", "green"))

    if success_steps:
        avg_success_time = sum(success_steps) / len(success_steps)
        print(colored(f"Average Navigation Time (Successful Episodes): {avg_success_time:.2f} steps", "green"))
    else:
        print(colored("No successful trials.", "red"))

    print(colored(f"Failure due to Out of Bounds: {fail_out_bound}", "red"))
    print(colored(f"Failure due to Timeout: {fail_timeout}", "blue"))
    print(colored(f"Failure due to Collision: {fail_collide}", "yellow"))

    # === Ensure directory exists ===
    csv_dir = "./test_output_08"
    os.makedirs(csv_dir, exist_ok=True)

    # === Construct file path ===
    if _is_predict:
        csv_file = os.path.join(csv_dir, f"Predict_seed_{random_seed}_dim_{state_dim}_flow_{flow_seed}_range_{switch_range}.csv")
    else:
        csv_file = os.path.join(csv_dir, f"Normal_seed_{random_seed}_dim_{state_dim}_flow_{flow_seed}_range_{switch_range}.csv")

    # === Write to CSV ===
    if record_rows:  # 确保有数据才写
        with open(csv_file, mode="w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=record_rows[0].keys())
            writer.writeheader()
            writer.writerows(record_rows)
    else:
        print("No data to write to CSV.")

    print(f"\nEvaluation details saved to {csv_file}")


def test_1():
    ########################### Mode 2: Heading Test ###########################
    print("============================================================================================")
    ########################### Environmenrt ###########################

    ################ Predict Model ################
    _pos_norm = True
    _acc_norm = False
    _expand_pressure = False
    # p_model_path = "./p_predict/0731_id15.pth"
    p_model_path = "./p_predict/0801_id16.pth"
    p_hidden_size = 256
    p_num_layers = 2
    p_output_dim = 14
    p_pred_steps = 10
    p_model = load_model(p_model_path, p_hidden_size, p_num_layers, p_output_dim, p_pred_steps, device)
    #####################################################  

    ################ Lilypad Settings ################
    parser_1 = argparse.ArgumentParser()
    args_1, unknown = parser_1.parse_known_args()
    args_1.action_interval = 10
    #####################################################  

    ################ RL Hyperparameters ################
    # action mode
    has_continuous_action_space = True
    # update epochs
    K_epochs = 40
    # clip rate for PPO
    eps_clip = 0.2
    # discount factor γ
    gamma = 0.99  
    # learning rate for actor network
    lr_actor = 0.0001
    # learning rate for critic network  
    lr_critic = 0.0002
    # set std for action distribution when testing
    action_std_4_test = 0.04  
    #####################################################  

    ################ Env Settings ################
    ### Obstacles Mode ###
    # _is_avoid = True
    _is_avoid = False

    ### State Dim ###
    # state_dim_mode = 22
    # state_dim_mode = 16
    state_dim_mode = 14
    # state_dim_mode = 6

    ### Steps ###
    max_steps = 300
    max_ep_len = max_steps + 20
    max_ll_steps = 5
    #####################################################  

    
    ################ Other Settings ################
    ### set flow seed if required (0 : random flow; else : fixed flow) ###
    # flow_seed = 0
    flow_seed = 24

    ### set heading mode(True : forward only; else : all direction) ###
    # _is_head = True
    _is_head = False


    ### predict & pre-trained high-level controller ###
    _is_predict = True
    ppo_path = './models/08/pre_pos_dim14_test_50.pth'
    # ppo_path = './models/08/pre_pos_dim16_10.pth'

    # _is_predict = False
    # ppo_path = './models/08/dim14_test_50.pth'
    # ppo_path = './models/08/dim16_3.pth'

    ### pre-trained low-level controller ###
    if _is_head:
        ppo_path_1 = './models/easy_task/0704_seed1.pth' # forward
    else:
        ppo_path_1 = './models/easy_task/0705_lc.pth' # all direction

    ### set random mode(True : real random mode; else : random.seed(random_seed)) ###
    true_random = True
    if true_random:
        random_seed = 0
    else:
        random_seed = 1
    
    # test_mode = True
    # switch_range = 1
    test_mode = False
    switch_range = 12

    # theta_mode = True
    theta_mode = False

    # switch_mode = True
    switch_mode = False

    _is_normalize = True
    # _is_normalize = False
    #####################################################


    # random point
    # _start = np.array([float(220), float(64)])
    # _target = np.array([float(64), float(64)])
    # random_range = 56
    # _start = np.array([float(240), float(64)])
    # _target = np.array([float(160), float(64)])
    # random_range = 32
    # env = train_env_upper_2.foil_env(args_1, max_step=max_steps, start_center=_start, target_center=_target, _include_flow=True, _plot_flow=True, _proccess_flow=False, _random_range=random_range, _init_flow_num=flow_seed, 
    # _pos_normalize=_is_normalize, _is_test = test_mode, _state_dim=state_dim_mode, 
    # _is_random = true_random, _set_random=random_seed, _swich_range = switch_range, u_range=20.0, v_range=20.0, 
    # _forward_only=_is_head, _obstacle_avoid=_is_avoid, _theta_mode = theta_mode, _is_switch=switch_mode, u_clip=15.0, v_clip=15.0, _test_info=False)


    # fixed point
    _start = np.array([float(256.71), float(85.69)])
    _target = np.array([float(131.76), float(67.41)])
    random_range = 32
    env = train_env_upper_2.foil_env(args_1, max_step=max_steps, start_position=_start, target_position=_target, _include_flow=True, _plot_flow=True, _proccess_flow=False, _random_range=random_range, _init_flow_num=flow_seed, 
    _pos_normalize=_is_normalize, _is_test = test_mode, _state_dim=state_dim_mode, 
    _is_random = true_random, _set_random=random_seed, _swich_range = switch_range, u_range=20.0, v_range=20.0, 
    _forward_only=_is_head, _obstacle_avoid=_is_avoid, _theta_mode = theta_mode, _is_switch=switch_mode, u_clip=15.0, v_clip=15.0, _test_info=False, video_dir="./Video_Heading/video_frames")


    ################ Action Space Settings ################
    ### High-level Controller ###
    state_dim = env.observation_space.shape[0]
    expand_dim = state_dim
    if _is_predict:
        if _expand_pressure:
            expand_dim += 8 * 14
        else:
            expand_dim += 8 * 6
    action_output_scale = np.array([])
    action_output_bias = np.array([])
    if has_continuous_action_space:
        # action space dimension
        action_dim = env.action_space.shape[0]
        action_output_scale = (env.action_space.high - env.action_space.low) / 2.0
        action_output_bias = (env.action_space.high + env.action_space.low) / 2.0
    else:
        action_dim = env.action_space.n
    
    ### Low-level Controller ###
    lc_state_dim = 14
    lc_action_output_scale = np.array([])
    lc_action_output_bias = np.array([])
    if has_continuous_action_space:
        # low-level controller action space dimension
        lc_action_dim = env.lc_action_space.shape[0]
        lc_action_output_scale = (env.lc_action_space.high - env.lc_action_space.low) / 2.0
        lc_action_output_bias = (env.lc_action_space.high + env.lc_action_space.low) / 2.0
    else:
        lc_action_dim = env.lc_action_space.n
    ######################################################

    ########################### PPO ###########################
    # initialize RL agent
    # ppo_agent = ICM_PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
    #                     has_continuous_action_space,
    #                     action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
    #                     continuous_action_output_bias=action_output_bias, icm_alpha=200)
    # ppo_agent.load_full_icm(ppo_path)
    ppo_agent = Classic_PPO(expand_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                            action_std_init=action_std_4_test, continuous_action_output_scale=action_output_scale,
                            continuous_action_output_bias=action_output_bias)
    ppo_agent.load_full(ppo_path)
    ppo_agent_lc = Classic_PPO(lc_state_dim, lc_action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
            action_std_init=action_std_4_test, continuous_action_output_scale=lc_action_output_scale,
            continuous_action_output_bias=lc_action_output_bias)
    ppo_agent_lc.load_full(ppo_path_1)
    ######################################################

    ########################### Test ###########################
    # Predictor Info
    if _is_predict:
        input_window = deque(maxlen=10)
        other_window = deque(maxlen=10)
        action_window = generate_fixed_action_window(_acc_norm)

    print(colored("**** Forward Mode **** " if _is_head else "**** Normal Mode **** ", 'green'))

    subgoal_threshold = 1
    state, info = env.reset()

    # Initial observation
    if _is_predict:
        pressure_now = np.array([info[f"pressure_{i}"] for i in range(1, 9)], dtype=np.float32)
        speed_now = np.array([info["vel_x"], info["vel_y"], info["vel_angle"]], dtype=np.float32)
        position_now = np.array([info["pos_x"], info["pos_y"], info["angle"]], dtype=np.float32)
        if _pos_norm:
            position_now = normalize_position(position_now)
        initial_action = np.zeros(3, dtype=np.float32)
        current_other = build_current_other(position_now, speed_now, initial_action)
        for _ in range(10):
            input_window.append(pressure_now.copy())
            other_window.append(current_other.copy())

        _future_pressure = get_future_pressure_batch(
            p_model, input_window, other_window, action_window,
            device, env.target_position, env.max_detect_dis + env.window_r / 2,
            _pos_norm=_pos_norm
        )
        if not _expand_pressure:
            _future_pressure = _future_pressure[:, :6]
        _expand_state = np.concatenate([state, _future_pressure.flatten()], axis=0)
    else:
        _expand_state = state

    state_14 = env.state_14
    agent_position = env.agent_pos
    prev_state = state.copy()
    prev_state_14 = state_14.copy()
    prev_agent_position = agent_position.copy()
    print("position_reset:", agent_position)

    # Main episode loop
    t = 0
    hl_time_step = 0
    done = False

    while not done and t < max_ep_len:
        # === Rule-based high-level direction: towards target ===
        agent_position = env.agent_pos
        target_position = env.target_position
        delta = target_position - agent_position
        angle_to_target = np.arctan2(delta[1], delta[0])  # [-π, π]

        if _is_head:
            high_action = angle_to_target
        else:
            vec = delta / (np.linalg.norm(delta) + 1e-6)
            high_action = np.arctan2(vec[1], vec[0])

        hl_time_step += 1
        print(f"[Rule] High Level Step {hl_time_step}, Angle to Target: {high_action:.2f}")

        # Compute subgoal position
        if _is_head:
            lc_target_x, lc_target_y = compute_subgoal_forward(pos=agent_position, angle=high_action, r=5)
        else:
            lc_target_x, lc_target_y = compute_subgoal(pos=agent_position, angle=high_action, r=5)
        env.set_lc_target([lc_target_x, lc_target_y])

        # Low-level control loop
        ll_step = 0
        done_lc = False
        while ll_step < max_ll_steps:
            agent_angle_old = env.angle

            # Build low-level state
            low_state = state_14.copy()
            low_state[3] = lc_target_x - agent_position[0]
            low_state[4] = lc_target_y - agent_position[1]

            # Select low-level action
            low_action = ppo_agent_lc.select_action(low_state)
            cliped_low_action = env.clip_action(low_action)
            print(f"=== Low Level Step {t}, Action: {low_action}")
            print(f"===               ->  Clip_Action: {cliped_low_action} ===")

            # Update sliding window
            if _is_predict:
                action_input = cliped_low_action / 15.0 if _acc_norm else cliped_low_action
                current_other = build_current_other(position_now, speed_now, action_input)
                input_window.append(pressure_now.copy())
                other_window.append(current_other.copy())

            # Step the environment
            state, reward, terminated, truncated, info = env.step(low_action)
            done = terminated or truncated

            if _is_predict:
                pressure_now = np.array([info[f"pressure_{i}"] for i in range(1, 9)], dtype=np.float32)
                speed_now = np.array([info["vel_x"], info["vel_y"], info["vel_angle"]], dtype=np.float32)
                position_now = np.array([info["pos_x"], info["pos_y"], info["angle"]], dtype=np.float32)
                if _pos_norm:
                    position_now = normalize_position(position_now)

            agent_position = env.agent_pos
            state_14 = env.state_14

            # Check subgoal success
            dist_to_subgoal = np.linalg.norm(agent_position - np.array([lc_target_x, lc_target_y]))
            if dist_to_subgoal < subgoal_threshold or done or ll_step + 1 >= max_ll_steps:
                done_lc = True

            t += 1
            ll_step += 1

            if done_lc:
                if _is_predict:
                    _future_pressure = get_future_pressure_batch(
                        p_model, input_window, other_window, action_window,
                        device, env.target_position, env.max_detect_dis + env.window_r / 2,
                        _pos_norm=_pos_norm
                    )
                    if not _expand_pressure:
                        _future_pressure = _future_pressure[:, :6]
                    _expand_state = np.concatenate([state, _future_pressure.flatten()], axis=0)
                else:
                    _expand_state = state
                ppo_agent_lc.buffer.clear()
                break

    env.close()
    print("Total_Steps:", t)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Choose a function to execute.")
    parser.add_argument("mode", type=int, choices=[0, 1, 2], help="Enter 0 to run test(), 1 to run success_rate_test(), 2 to run success_rate_test_1()")
    parser.add_argument("id", type=int, nargs="?", default=0, help="ID for success_rate_test() if mode is 1, Seed for success_rate_test_1() if mode is 2")
    args = parser.parse_args()

    if args.mode == 0:
        test()
    elif args.mode == 1:
        success_rate_test(args.id)
    elif args.mode == 2:
        test_1()

