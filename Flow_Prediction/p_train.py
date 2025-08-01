import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.io import loadmat
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import traceback
import json
import csv

folders = [
    f"./train_data_2/flow_data_{i}.h5"
    for i in range(1, 21)
]

class Seq2SeqPredictor(nn.Module):
    def __init__(self, input_dim=14, condition_dim=3, output_dim=8, hidden_size=128, lstm_layers=2, pred_steps=5, teacher_forcing=True):
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

    def forward(self, past_pressure, past_speed, future_acc, target_pressure=None):
        """
        past_pressure: (B, T_in, 8)
        past_speed: (B, T_in, 6)
        future_acc: (B, T_pred, 3)
        target_pressure: (B, T_pred, 8) for teacher forcing
        """
        B, T_in, _ = past_pressure.shape
        T_pred = future_acc.shape[1]

        # ===== Encoder =====
        enc_input = torch.cat([past_pressure, past_speed], dim=-1)  # (B, T_in, 14)
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
            dec_input = torch.cat([decoder_input, cond_acc], dim=-1)              # (B, 8+3)
            dec_input = self.decoder_input_fc(dec_input).unsqueeze(1)             # (B, 1, hidden_size)

            dec_output, (h_dec, c_dec) = self.decoder_lstm(dec_input, (h_dec, c_dec))
            pred_p = self.decoder_output_fc(dec_output.squeeze(1))                # (B, 8)

            preds.append(pred_p.unsqueeze(1))                                     # [(B, 1, 8), ...]

            if self.teacher_forcing and self.training and target_pressure is not None:
                decoder_input = target_pressure[:, t, :]
            else:
                decoder_input = pred_p

        preds = torch.cat(preds, dim=1)                                           # (B, T_pred, 8)
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


class LSTMPredictorIndependentDecoders(nn.Module):
    def __init__(self, hidden_size=128, lstm_layers=2, output_dim=8, pred_steps=5, acc_proj_dim=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.pred_steps = pred_steps
        self.acc_proj_dim = acc_proj_dim

        # 输入维度: 8 (pressure) + 3 (vel) + 3 (acc) = 14
        self.input_fc = nn.Linear(14, hidden_size)
        # self.input_fc = nn.Sequential(
        #                 nn.Linear(14, hidden_size), 
        #                 nn.ReLU(), 
        #                 nn.Linear(hidden_size, hidden_size))

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )

        # === 加速度投影层 ===
        self.acc_fc = nn.Linear(3, acc_proj_dim)
        self.acc_fc =nn.Sequential(
                    nn.Linear(3, acc_proj_dim*2),
                    nn.ReLU(),
                    nn.Linear(acc_proj_dim*2, acc_proj_dim)
                    )

        # 每个预测步使用独立的 Decoder MLP
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size + acc_proj_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_dim)
            ) for _ in range(pred_steps)
        ])

        self._initialize_weights()

    def forward(self, past_pressure, past_speed, future_acc):
        """
        past_pressure: (B, T, 8)
        past_speed: (B, T, 6)  # 3 (vel) + 3 (acc)
        future_acc: (B, pred_steps, 3)
        """
        B, T, _ = past_pressure.shape

        # Concatenate pressure, vel, acc -> (B, T, 14)
        x = torch.cat([past_pressure, past_speed], dim=-1)
        x = self.input_fc(x)  # (B, T, hidden_size)

        # LSTM Encoding
        h0 = torch.zeros(self.lstm_layers, B, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.lstm_layers, B, self.hidden_size, device=x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))  # (B, T, hidden_size)

        lstm_output = lstm_out[:, -1, :]  # (B, hidden_size)

        preds = []
        for t in range(self.pred_steps):
            cond = future_acc[:, t, :]  # (B, 3)

            # === 新增: 投影加速度 ===
            cond_proj = self.acc_fc(cond)  # (B, acc_proj_dim)

            dec_input = torch.cat([lstm_output, cond_proj], dim=-1)  # (B, hidden_size + acc_proj_dim)
            pred_p = self.decoders[t](dec_input)  # (B, 8)
            preds.append(pred_p.unsqueeze(1))

        preds = torch.cat(preds, dim=1)  # (B, pred_steps, 8)
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

class PressureLSTMDataset_Normalize(Dataset):
    def __init__(self, file_paths, input_len=5, pred_len=5, _is_normalize=False, normalize_h5_path="./train_data/min_max.h5"):
        self.input_len = input_len
        self.pred_len = pred_len
        self.total_len = input_len + pred_len
        self.normalize = _is_normalize

        # 归一化参数初始化为None
        self.pressure_min = None
        self.pressure_max = None
        self.speed_min = None
        self.speed_max = None
        self.acc_min = None
        self.acc_max = None

        if self.normalize:
            with h5py.File(normalize_h5_path, "r") as f:
                if "pressure/q05" in f and "pressure/q95" in f:
                    self.pressure_min = f["pressure/q05"][:]   # numpy array
                    self.pressure_max = f["pressure/q95"][:]
                else:
                    raise ValueError("Pressure quantile information not found in provided HDF5 file.")
            self.speed_min = np.array([-1.0, -1.0, -0.15], dtype=np.float32)
            self.speed_max = np.array([1.0, 1.0, 0.15], dtype=np.float32)
            self.acc_min = np.array([-15.0, -15.0, -5.0], dtype=np.float32)
            self.acc_max = np.array([15.0, 15.0, 5.0], dtype=np.float32)
            

        self.index_list = []

        for path in file_paths:
            if os.path.isdir(path):
                file_names = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.h5')]
            else:
                file_names = [path]

            for file in file_names:
                with h5py.File(file, "r") as f:
                    for ep_key in f.keys():
                        if "pressure" not in f[ep_key]:
                            continue
                        T = f[ep_key]["pressure"].shape[1]
                        if T >= self.total_len:
                            for t in range(T - self.total_len + 1):
                                self.index_list.append((file, ep_key, t))

        print(f"Loaded {len(self.index_list)} samples from {len(file_paths)} files.")

        if len(self.index_list) == 0:
            print("[Error] Dataset contains 0 samples. Check data paths and data validity.")
        else:
            try:
                first_sample = self.__getitem__(0)
                print("[Debug] Successfully fetched first sample.")
                print("[Debug] input_seq shape:", first_sample[0].shape)
                print("[Debug] other_seq shape:", first_sample[1].shape)
                print("[Debug] action_seq shape:", first_sample[2].shape)
                print("[Debug] target_seq shape:", first_sample[3].shape)
            except Exception:
                import traceback
                traceback.print_exc()

    def __len__(self):
        return len(self.index_list)

    def normalize_data(self, data, data_min, data_max):
        denom = data_max - data_min
        denom[denom == 0] = 1e-6
        normalized = (data - data_min) / denom  # [0, 1] 归一化

        normalized = normalized * 2 - 1         # [0, 1] → [-1, 1]

        normalized = np.clip(normalized, -1.0, 1.0)  # 再次严格 clip

        return normalized


    def __getitem__(self, idx):
        while True:
            file_path, ep_key, t = self.index_list[idx]
            with h5py.File(file_path, "r") as f:
                grp = f[ep_key]
                pressure = grp["pressure"][0]  # (T, 8)
                speed = grp["speed"][0]        # (T, 6)
                acc = grp["action"][0]         # (T, 8)

                input_seq = np.clip(pressure[t: t + self.input_len], -10.0, 10.0)
                target_seq = np.clip(pressure[t + self.input_len: t + self.total_len], -10.0, 10.0)
                action_seq = acc[t + self.input_len: t + self.total_len]
                speed_seq = np.clip(speed[t: t + self.input_len], -2.0, 2.0)
                acc_seq = acc[t: t + self.input_len]


                if (
                    np.isnan(input_seq).any() or np.isinf(input_seq).any() or
                    np.isnan(speed_seq).any() or np.isinf(speed_seq).any() or
                    np.isnan(action_seq).any() or np.isinf(action_seq).any() or
                    np.isnan(target_seq).any() or np.isinf(target_seq).any() or
                    np.isnan(acc_seq).any() or np.isinf(acc_seq).any()
                ):
                    idx = np.random.randint(0, len(self.index_list))
                    continue

                if self.normalize:
                    input_seq = self.normalize_data(input_seq, self.pressure_min, self.pressure_max)
                    target_seq = self.normalize_data(target_seq, self.pressure_min, self.pressure_max)
                    action_seq = self.normalize_data(action_seq, self.acc_min, self.acc_max)
                    acc_seq = self.normalize_data(acc_seq, self.acc_min, self.acc_max)
                    speed_seq = self.normalize_data(speed_seq, self.speed_min, self.speed_max)

                other_seq = np.concatenate([speed_seq, acc_seq], axis=-1)    

                return (
                    torch.tensor(input_seq, dtype=torch.float32),
                    torch.tensor(other_seq, dtype=torch.float32),
                    torch.tensor(action_seq, dtype=torch.float32),
                    torch.tensor(target_seq, dtype=torch.float32),
                )

class PressureLSTMDataset(Dataset):
    def __init__(self, file_paths, input_len=5, pred_len=5):
        """
        Return shapes:
        input_seq:   (input_len, pressure_dim)
        other_seq:   (input_len, speed_dim)
        action_seq:  (pred_len, acc_dim)
        target_seq:  (pred_len, pressure_dim)
        """
        self.input_len = input_len
        self.pred_len = pred_len
        self.total_len = input_len + pred_len

        self.index_list = []  # (file_path, ep_key, t_steps)

        for path in file_paths:
            if os.path.isdir(path):
                file_names = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.h5')]
            else:
                file_names = [path]

            for file in file_names:
                with h5py.File(file, "r") as f:
                    for ep_key in f.keys():
                        if "pressure" not in f[ep_key]:
                            continue
                        T = f[ep_key]["pressure"].shape[1]  # (1, T, ...)
                        if T >= self.total_len:
                            for t in range(T - self.total_len + 1):
                                self.index_list.append((file, ep_key, t))
        
        print(f"Loaded {len(self.index_list)} samples from {len(file_paths)} files.")
        if len(self.index_list) == 0:
            print("[Error] Dataset contains 0 samples. Check data paths and data validity.")
            for path in file_paths:
                print("Checking path:", path)
        else:
            try:
                first_sample = self.__getitem__(0)
                print("[Debug] Successfully fetched first sample.")
                print("[Debug] input_seq shape:", first_sample[0].shape)
                print("[Debug] other_seq shape:", first_sample[1].shape)
                print("[Debug] action_seq shape:", first_sample[2].shape)
                print("[Debug] target_seq shape:", first_sample[3].shape)
            except Exception as e:
                print("[Error] Failed to fetch first sample:")
                
                traceback.print_exc()

    def __len__(self):
        return len(self.index_list)
    
    def __getitem__(self, idx):
        while True:
            file_path, ep_key, t = self.index_list[idx]
            with h5py.File(file_path, "r") as f:
                grp = f[ep_key]
                pressure = grp["pressure"][0]
                speed = grp["speed"][0]
                acc = grp["action"][0]
                position = grp["position"][0]   # (T, 3)

                # === 自动过滤 angle > pi 的样本 ===
                # angle_seq = position[t: t + self.total_len, 2]  # (total_len,)
                # if np.any(angle_seq > np.pi):
                #     idx = np.random.randint(0, len(self.index_list))
                #     continue

                # === 异常值裁剪 ===
                input_seq = np.clip(pressure[t: t + self.input_len], -10.0, 10.0)  # (input_len, 8)

                # TODO:pressure or speed
                target_seq = np.clip(pressure[t + self.input_len: t + self.total_len], -10.0, 10.0)  # (pred_len, 8)
                # target_seq = np.clip(speed[t + self.input_len: t + self.total_len], -2.5, 2.5)  # (pred_len, 3)

                action_seq = acc[t + self.input_len: t + self.total_len] # (pred_len, 8)
                speed_seq = np.clip(speed[t: t + self.input_len], -2.5, 2.5) # (pred_len, 3)
                acc_seq = acc[t: t + self.input_len] # (pred_len, 3)
                other_seq = np.concatenate([speed_seq, acc_seq], axis=-1)  # (input_len, 6)

                # === 自动跳过包含 NaN / Inf 的样本 ===
                if (
                    np.isnan(input_seq).any() or np.isinf(input_seq).any() or
                    np.isnan(other_seq).any() or np.isinf(other_seq).any() or
                    np.isnan(action_seq).any() or np.isinf(action_seq).any() or
                    np.isnan(target_seq).any() or np.isinf(target_seq).any()
                ):
                    idx = np.random.randint(0, len(self.index_list))
                    continue
                # print(f"[DEBUG] input_seq shape: {input_seq.shape}, target_seq shape: {target_seq.shape}, other_seq shape: {other_seq.shape}")
                return (
                    torch.tensor(input_seq, dtype=torch.float32),
                    torch.tensor(other_seq, dtype=torch.float32),
                    torch.tensor(action_seq, dtype=torch.float32),
                    torch.tensor(target_seq, dtype=torch.float32),
                )

    # def __getitem__(self, idx):
    #     while True:
    #         file_path, ep_key, t = self.index_list[idx]
    #         with h5py.File(file_path, "r") as f:
    #             grp = f[ep_key]
    #             pressure = grp["pressure"][0]      # (T, 8)
    #             speed = grp["speed"][0]            # (T, 3)
    #             acc = grp["action"][0]             # (T, 3)
    #             position = grp["position"][0]      # (T, 3)

    #             # ========== 压力差分 ==========
    #             pressure_diff = pressure[1:] - pressure[:-1]  # shape (T-1, 8)

    #             # 使用差分作为目标预测
    #             target_seq = np.clip(
    #                 pressure_diff[t + self.input_len - 1: t + self.total_len - 1], 
    #                 -5.0, 5.0
    #             )  # (pred_len, 8)

    #             # 输入依旧使用绝对压力（也可换成差分）
    #             input_seq = np.clip(
    #                 pressure[t: t + self.input_len], 
    #                 -10.0, 10.0
    #             )  # (input_len, 8)

    #             action_seq = acc[t + self.input_len: t + self.total_len]  # (pred_len, 3)

    #             other_seq = np.concatenate([
    #                 np.clip(speed[t: t + self.input_len], -2.5, 2.5),
    #                 acc[t: t + self.input_len]
    #             ], axis=-1)  # (input_len, 6)

    #             # ========== 异常过滤 ==========
    #             if (
    #                 np.isnan(input_seq).any() or np.isinf(input_seq).any() or
    #                 np.isnan(other_seq).any() or np.isinf(other_seq).any() or
    #                 np.isnan(action_seq).any() or np.isinf(action_seq).any() or
    #                 np.isnan(target_seq).any() or np.isinf(target_seq).any()
    #             ):
    #                 idx = np.random.randint(0, len(self.index_list))
    #                 continue

    #             return (
    #                 torch.tensor(input_seq, dtype=torch.float32),    # (input_len, 8)
    #                 torch.tensor(other_seq, dtype=torch.float32),    # (input_len, 6)
    #                 torch.tensor(action_seq, dtype=torch.float32),   # (pred_len, 3)
    #                 torch.tensor(target_seq, dtype=torch.float32),   # (pred_len, 8)
    #             )


def train(
    model, train_loader, criterion, optimizer, scaler, device, epoch, num_epochs, log_dir,
    grad_clip_value=0.5, initial_loss_clip_threshold=1e3, dynamic_k=3, min_threshold=10.0, warmup_epochs=5):
    model.train()
    remaining_epochs = num_epochs - epoch - 1
    total_loss = 0
    valid_batch_count = 0

    skipped_batch_count = 0
    total_batch_count = len(train_loader)

    start_time = time.time()
    batch_losses = []
    grad_stats = []
    valid_losses_for_dynamic = []  # 用于动态调整阈值

    for batch_idx, (input_seq, other_seq, action_seq, target_seq) in enumerate(train_loader):
        input_seq = input_seq.to(device)
        other_seq = other_seq.to(device)
        action_seq = action_seq.to(device)
        target_seq = target_seq.to(device)

        optimizer.zero_grad()
        with autocast('cuda'):
            outputs = model(input_seq, other_seq, action_seq)
            loss = criterion(outputs, target_seq)

        # 计算动态阈值
        if valid_losses_for_dynamic:
            mean_loss = np.mean(valid_losses_for_dynamic)
            std_loss = np.std(valid_losses_for_dynamic)
            dynamic_threshold = max(mean_loss + dynamic_k * std_loss, min_threshold)
        else:
            dynamic_threshold = initial_loss_clip_threshold

        # 判断是否异常
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() > dynamic_threshold:
            print(f"[Warning] Skipped Batch {batch_idx} due to abnormal loss: {loss.item():.6f} > threshold {dynamic_threshold:.6f}")
            skipped_batch_count += 1
            continue
        # === 只有超过warmup_epochs才启用异常batch跳过 ===
        # if epoch >= warmup_epochs:
        #     if torch.isnan(loss) or torch.isinf(loss) or loss.item() > dynamic_threshold:
        #         print(f"[Warning] Skipped Batch {batch_idx} due to abnormal loss: {loss.item():.6f} > threshold {dynamic_threshold:.6f}")
        #         skipped_batch_count += 1
        #         continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        scaler.step(optimizer)
        scaler.update()

        loss_cpu = loss.item()
        total_loss += loss_cpu
        valid_batch_count += 1
        valid_losses_for_dynamic.append(loss_cpu)

        batch_losses.append({
            'Epoch': epoch + 1,
            'Batch': batch_idx + 1,
            'Loss': loss_cpu,
            'Dynamic_Threshold': dynamic_threshold
        })

        if batch_idx % 10 == 0:
            elapsed_time = time.time() - start_time
            remaining_batches = len(train_loader) - batch_idx
            estimated_time = elapsed_time / (batch_idx + 1) * remaining_batches
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch {batch_idx}/{len(train_loader)}, Loss: {loss_cpu:.8f}, "
                  f"Thresh: {dynamic_threshold:.2f}, "
                  f"Elapsed: {elapsed_time / 60:.2f}min, Remaining: {estimated_time / 60:.2f}min, "
                  f"Total Remain: {((elapsed_time + estimated_time) * remaining_epochs + estimated_time) / 3600:.2f}h")

            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_max = param.grad.abs().max().item()
                    grad_stats.append({
                        'Epoch': epoch + 1,
                        'Batch': batch_idx + 1,
                        'Layer': name,
                        'Grad_Max': grad_max
                    })

    # 写入训练损失日志
    if batch_losses:
        batch_loss_df = pd.DataFrame(batch_losses)
        batch_loss_df.to_csv(
            os.path.join(log_dir, 'train_batch_losses.csv'),
            mode='a',
            header=not os.path.exists(os.path.join(log_dir, 'train_batch_losses.csv')),
            index=False
        )

    # 写入梯度日志
    if grad_stats:
        grad_stats_df = pd.DataFrame(grad_stats)
        grad_stats_df.to_csv(
            os.path.join(log_dir, 'grad_stats.csv'),
            mode='a',
            header=not os.path.exists(os.path.join(log_dir, 'grad_stats.csv')),
            index=False
        )

    # 打印丢弃比例
    skip_ratio = skipped_batch_count / total_batch_count
    print(f"[Info] Epoch {epoch + 1}: Skipped {skipped_batch_count}/{total_batch_count} batches "
          f"({skip_ratio:.2%}) due to abnormal loss filtering.")

    if valid_batch_count == 0:
        print("[Warning] All batches in this epoch were skipped due to abnormal loss.")
        return float('nan')

    return total_loss / valid_batch_count


def test(model, test_loader, criterion, device, epoch, num_epochs, log_dir):
    model.eval()
    remaining_epochs = num_epochs - epoch - 1
    total_loss = 0
    start_time = time.time()
    batch_losses = []

    with torch.no_grad():
        for batch_idx, (input_seq, other_seq, action_seq, target_seq) in enumerate(test_loader):
            input_seq = input_seq.to(device)
            other_seq = other_seq.to(device)
            action_seq = action_seq.to(device)
            target_seq = target_seq.to(device)

            # with autocast():
            with autocast('cuda'):
                outputs = model(input_seq, other_seq, action_seq)  # 使用两个输入
                loss = criterion(outputs, target_seq)

            loss_cpu = loss.item()
            total_loss += loss_cpu
            batch_losses.append({'Epoch': epoch + 1, 'Batch': batch_idx + 1, 'Loss': loss_cpu})

            if batch_idx % 10 == 0:  # 每10个batch打印一次
                elapsed_time = time.time() - start_time
                remaining_batches = len(test_loader) - batch_idx
                estimated_time = elapsed_time / (batch_idx + 1) * remaining_batches

                print(
                    f"Test Epoch [{epoch + 1}/{num_epochs}], Batch {batch_idx}/{len(test_loader)}, "
                    f"Loss: {loss_cpu:.8f}, "
                    f"Epoch Elapsed: {elapsed_time / 60:.2f}min, Epoch Remains: {estimated_time / 60:.2f}min, "
                    f"Total Remaining: {((elapsed_time + estimated_time) * remaining_epochs + estimated_time) / 3600:.2f}h"
                )

    # 保存批次测试损失日志
    batch_loss_df = pd.DataFrame(batch_losses)
    batch_loss_df.to_csv(
        os.path.join(log_dir, 'test_batch_losses.csv'),
        mode='a',
        header=not os.path.exists(os.path.join(log_dir, 'test_batch_losses.csv')),
        index=False
    )

    return total_loss / len(test_loader)


def train_main(data_paths, _lr=1e-4, num_epochs=10, batch_size=64, CUDA_ID=0, model_id='001'):
    device = torch.device(f'cuda:{CUDA_ID}' if torch.cuda.is_available() else 'cpu')

    print("Using device:", device)

    # Step 1: 构建 Dataset 和 DataLoader
    # TODO:pressure to speed
    dataset = PressureLSTMDataset(data_paths, input_len=5, pred_len=5)
    # dataset = PressureLSTMDataset_Normalize(data_paths, input_len=5, pred_len=5, _is_normalize=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Step 2: 构建模型
    # TODO:pressure to speed
    # model = LSTMPredictorIndependentDecoders(hidden_size=256, lstm_layers=2, output_dim=8, pred_steps=5, acc_proj_dim=128).to(device)
    model = Seq2SeqPredictor(input_dim=14, condition_dim=3, output_dim=8, hidden_size=256, lstm_layers=2, pred_steps=5, teacher_forcing=True).to(device)

    # Step 3: 优化器、损失函数、GradScaler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=_lr)
    scaler = torch.amp.GradScaler('cuda')

    # 日志目录
    model_dir = f'pressure_prediction_models/ID_{model_id}'
    log_dir = f'pressure_prediction_logs/ID_{model_id}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    loss_log_path = os.path.join(log_dir, "epoch_loss_log.csv")
    with open(loss_log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss'])

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, scaler, device, epoch, num_epochs, log_dir)
        val_loss = test(model, val_loader, criterion, device, epoch, num_epochs, log_dir)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        torch.save(model.state_dict(), os.path.join(model_dir, f'model_epoch_{epoch+1}.pth'))
        with open(loss_log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss])

    torch.save(model.state_dict(), os.path.join(model_dir, 'model_final.pth'))

if __name__ == '__main__':
    # id 1 压力 hidden 64 acc 32
    # id 2 压力差分 hidden 64 acc 32
    # id 3 压力差分 hidden 256 acc 128
    # id 4 压力 hidden 256 acc 128
    # id 5 压力+动作 预测速度变化
    # id 6 速度 10steps
    # id 7 网络结构改(input)
    # id 8 网络结构改(action)

    # id 9 seq2seq
    # id 10 seq2seq normalize
    start_id = 12
    train_times = 1
    lrs = [1e-4]

    for i in range(train_times):
        train_main(data_paths=folders, _lr=lrs[i], num_epochs=150, batch_size=1024, model_id=f'{start_id + i}')

    # test_main(data_idx=10010)
    # test_main(data_idx=10086)
    # test_main(data_idx=12345)
    # test_main(data_idx=180000)