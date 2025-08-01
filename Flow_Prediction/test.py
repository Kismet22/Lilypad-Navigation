import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
from scipy.io import loadmat
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import h5py

# folders = [
#     "./06_data/flow_data_0.h5",
#     "./06_data/flow_data_1.h5",
#     "./06_data/flow_data_2.h5",
#     "./06_data/flow_data_3.h5",
#     "./06_data/flow_data_4.h5",
# ]

folders = [
    "./07_data/flow_data_1.h5",
    "./07_data/flow_data_2.h5",
    "./07_data/flow_data_3.h5",
    "./07_data/flow_data_4.h5",
    "./07_data/flow_data_5.h5",
    "./07_data/flow_data_6.h5"
]

model_path = "./test_models/0708.pth"
test_path = "./07_data/flow_data_1.h5"

# LSTM-TransConv Network
class LSTMConvUpsamplePredictor(nn.Module):
    def __init__(self, flow_encoder_output_size, other_size, hidden_size, num_layers, output_steps):
        super().__init__()
        self.flow_encoder_output_size = flow_encoder_output_size
        self.other_size = other_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_steps = output_steps

        # CNN Encoder for flow
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, flow_encoder_output_size),
            nn.ReLU(),
        )

        # LSTM
        lstm_input_size = flow_encoder_output_size + other_size
        self.lstm = nn.LSTM(
            lstm_input_size, hidden_size, num_layers, batch_first=True
        )

        # Linear + upsampling layers
        self.fcs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size * 8 * 8) for _ in range(output_steps)
        ])
        self.conv2 = nn.Conv2d(hidden_size, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 8, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(8, 2, kernel_size=3, padding=1)

        self._initialize_weights()

    def forward(self, flow_seq, other_seq):
        B, seq_len, C, H, W = flow_seq.shape
        flow_seq_reshaped = flow_seq.view(B * seq_len, C, H, W)
        encoded_flow = self.encoder_cnn(flow_seq_reshaped)
        encoded_flow = encoded_flow.view(B, seq_len, -1)

        lstm_input = torch.cat([encoded_flow, other_seq], dim=-1)

        h_0 = torch.zeros(self.lstm.num_layers, B, self.hidden_size, device=flow_seq.device)
        c_0 = torch.zeros(self.lstm.num_layers, B, self.hidden_size, device=flow_seq.device)

        out, _ = self.lstm(lstm_input, (h_0, c_0))
        lstm_output = out[:, -1, :]

        outputs = []
        for t in range(self.output_steps):
            fc_out = self.fcs[t](lstm_output)
            fc_out = fc_out.view(-1, self.hidden_size, 8, 8)

            # 动态匹配输入尺寸
            upsampled_out = F.interpolate(fc_out, size=(H // 2, W // 2), mode='bilinear', align_corners=True)
            conv_out = torch.relu(self.conv2(upsampled_out))

            upsampled_out = F.interpolate(conv_out, size=(H, W), mode='bilinear', align_corners=True)
            conv_out = torch.relu(self.conv3(upsampled_out))
            conv_out = torch.relu(self.conv4(conv_out))
            conv_out = self.conv5(conv_out)

            outputs.append(conv_out)

        outputs = torch.stack(outputs, dim=1)
        return outputs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

# # LSTM-TransConv Network
# class LSTMConvUpsamplePredictor(nn.Module):
#     def __init__(self, flow_encoder_output_size, hidden_size, num_layers, output_steps):
#         super().__init__()
#         self.flow_encoder_output_size = flow_encoder_output_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.output_steps = output_steps

#         # CNN Encoder for flow
#         self.encoder_cnn = nn.Sequential(
#             nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(),
#             nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
#             nn.AdaptiveAvgPool2d((8, 8)),
#             nn.Flatten(),
#             nn.Linear(32 * 8 * 8, flow_encoder_output_size),
#             nn.ReLU(),
#         )

#         # 延迟初始化 LSTM，首次 forward 时初始化
#         self.lstm = None

#         # Linear + upsampling layers
#         self.fcs = nn.ModuleList([
#             nn.Linear(hidden_size, hidden_size * 8 * 8) for _ in range(output_steps)
#         ])
#         self.conv2 = nn.Conv2d(hidden_size, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(32, 8, kernel_size=3, padding=1)
#         self.conv5 = nn.Conv2d(8, 2, kernel_size=3, padding=1)

#         self._initialize_weights()

#     def forward(self, flow_seq, other_seq):
#         B, seq_len, C, H, W = flow_seq.shape
#         flow_seq_reshaped = flow_seq.view(B * seq_len, C, H, W)
#         encoded_flow = self.encoder_cnn(flow_seq_reshaped)
#         encoded_flow = encoded_flow.view(B, seq_len, -1)

#         # 动态推断 other_size 并在首次 forward 时创建 LSTM
#         if self.lstm is None:
#             other_size = other_seq.shape[-1]
#             lstm_input_size = self.flow_encoder_output_size + other_size
#             self.lstm = nn.LSTM(
#                 lstm_input_size, self.hidden_size, self.num_layers, batch_first=True
#             ).to(flow_seq.device)
#             # 初始化权重
#             for name, param in self.lstm.named_parameters():
#                 if 'weight' in name:
#                     nn.init.xavier_normal_(param)
#                 elif 'bias' in name:
#                     nn.init.constant_(param, 0)

#         lstm_input = torch.cat([encoded_flow, other_seq], dim=-1)

#         h_0 = torch.zeros(self.lstm.num_layers, B, self.hidden_size, device=flow_seq.device)
#         c_0 = torch.zeros(self.lstm.num_layers, B, self.hidden_size, device=flow_seq.device)

#         out, _ = self.lstm(lstm_input, (h_0, c_0))
#         lstm_output = out[:, -1, :]

#         outputs = []
#         for t in range(self.output_steps):
#             fc_out = self.fcs[t](lstm_output)
#             fc_out = fc_out.view(-1, self.hidden_size, 8, 8)

#             upsampled_out = F.interpolate(fc_out, scale_factor=2, mode='bilinear', align_corners=True)
#             conv_out = torch.relu(self.conv2(upsampled_out))
#             upsampled_out = F.interpolate(conv_out, scale_factor=2, mode='bilinear', align_corners=True)
#             conv_out = torch.relu(self.conv3(upsampled_out))
#             conv_out = torch.relu(self.conv4(conv_out))
#             conv_out = self.conv5(conv_out)
#             outputs.append(conv_out)

#         outputs = torch.stack(outputs, dim=1)
#         return outputs

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 nn.init.constant_(m.bias, 0)


class FlowLSTMDataset(Dataset):
    def __init__(self, file_paths, input_len=5, pred_len=5):
        self.input_len = input_len
        self.pred_len = pred_len
        self.total_len = input_len + pred_len

        self.index_list = []  # 每一项是 (file_path, ep_key, t)

        for path in file_paths:
            if os.path.isdir(path):
                file_names = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.h5')]
            else:
                file_names = [path]

            for file in file_names:
                with h5py.File(file, "r") as f:
                    for ep_key in f.keys():
                        if "flow" not in f[ep_key]:
                            continue
                        T = f[ep_key]["flow"].shape[1]  # (1, T, ...)
                        if T >= self.total_len:
                            for t in range(T - self.total_len + 1):
                                self.index_list.append((file, ep_key, t))

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        while True:
            file_path, ep_key, t = self.index_list[idx]
            with h5py.File(file_path, "r") as f:
                grp = f[ep_key]
                flow = grp["flow"][0]  # (T, 2, 32, 32)
                speed = grp["speed"][0]

                input_seq = flow[t: t + self.input_len]  # (input_len, 2, 32, 32)
                target_seq = flow[t + self.input_len: t + self.total_len]  # (pred_len, 2, 32, 32)
                other_seq = np.concatenate([
                    speed[t: t + self.input_len],
                ], axis=-1)  # (input_len, other_dim)

                # === 裁剪超过绝对值上界 2.5 的值 ===
                input_seq = np.clip(input_seq, -2.5, 2.5)
                target_seq = np.clip(target_seq, -2.5, 2.5)
                other_seq = np.clip(other_seq, -2.5, 2.5)

                # === 自动跳过包含 NaN / Inf 的样本 ===
                if (
                    np.isnan(input_seq).any() or np.isinf(input_seq).any() or
                    np.isnan(other_seq).any() or np.isinf(other_seq).any() or
                    np.isnan(target_seq).any() or np.isinf(target_seq).any()
                ):
                    idx = np.random.randint(0, len(self.index_list))
                    continue

                return (
                    torch.tensor(input_seq, dtype=torch.float32),
                    torch.tensor(other_seq, dtype=torch.float32),
                    torch.tensor(target_seq, dtype=torch.float32),
                )


# 训练函数
def train(
    model, train_loader, criterion, optimizer, scaler, device, epoch, num_epochs, log_dir,
    grad_clip_value=0.5, initial_loss_clip_threshold=1e3, dynamic_k=3, min_threshold=1000.0
):
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

    for batch_idx, (flow_seq, other_seq, target_seq) in enumerate(train_loader):
        flow_seq = flow_seq.to(device)
        other_seq = other_seq.to(device)
        target_seq = target_seq.to(device)

        optimizer.zero_grad()
        # with autocast():
        with autocast('cuda'):
            outputs = model(flow_seq, other_seq)
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
        for batch_idx, (flow_seq, other_seq, target_seq) in enumerate(test_loader):
            flow_seq = flow_seq.to(device)
            other_seq = other_seq.to(device)
            target_seq = target_seq.to(device)

            # with autocast():
            with autocast('cuda'):
                outputs = model(flow_seq, other_seq)  # ✅ 使用两个输入
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


def train_main(data_paths, flow_encoder_output_size,
               _lr=1e-4, num_epochs=10, batch_size=64, CUDA_ID=0, model_id='001'):
    device = torch.device(f'cuda:{CUDA_ID}' if torch.cuda.is_available() else 'cpu')

    print("Using device:", device)

    # Step 1: 构建 Dataset 和 DataLoader
    dataset = FlowLSTMDataset(data_paths, input_len=5, pred_len=5)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Step 2: 构建模型
    model = LSTMConvUpsamplePredictor(flow_encoder_output_size,
                                      hidden_size=256, num_layers=2, output_steps=5).to(device)

    # Step 3: 优化器、损失函数、GradScaler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=_lr)
    # scaler = torch.cuda.amp.GradScaler()
    scaler = torch.amp.GradScaler('cuda')

    # 日志目录
    model_dir = f'06_prediction_models/ID_{model_id}'
    log_dir = f'06_prediction_logs/ID_{model_id}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, scaler, device, epoch, num_epochs, log_dir)
        val_loss = test(model, val_loader, criterion, device, epoch, num_epochs, log_dir)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        torch.save(model.state_dict(), os.path.join(model_dir, f'model_epoch_{epoch+1}.pth'))

    torch.save(model.state_dict(), os.path.join(model_dir, 'model_final.pth'))


def load_model(model_path, input_size, hidden_size, num_layers, output_steps, device, other_size=3):
    # (self, flow_encoder_output_size, hidden_size, num_layers, output_steps)
    model = LSTMConvUpsamplePredictor(input_size, 3, hidden_size, num_layers, output_steps).to(device)
    # 先进行一次虚拟 forward 初始化 LSTM
    dummy_flow = torch.zeros(1, 5, 2, 32, 32).to(device)
    dummy_other = torch.zeros(1, 5, other_size).to(device)  # 请确保 other_size 与实际相符
    _ = model(dummy_flow, dummy_other)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def plot_comparison_from_batch(input_seq, other_seq, target_seq, model, device, vmin=-2.5, vmax=2.5):
    """
    可视化：
    - 输入流场 (input_seq)
    - 预测流场 (predicted_output)
    - 真实流场 (target_seq)

    input_seq, other_seq, target_seq 均为 (1, T, ...)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    input_len = input_seq.shape[1]
    pred_len = target_seq.shape[1]

    model.eval()
    with torch.no_grad():
        predicted_output = model(input_seq.to(device), other_seq.to(device)).cpu().numpy()  # (1, pred_len, 2, 32, 32)

    input_seq_np = input_seq.cpu().numpy()[0]      # (input_len, 2, 32, 32)
    target_seq_np = target_seq.cpu().numpy()[0]    # (pred_len, 2, 32, 32)
    predicted_output_np = predicted_output[0]      # (pred_len, 2, 32, 32)

    max_len = max(input_len, pred_len)

    # ========== u 分量 ==========
    fig, axes = plt.subplots(3, max_len, figsize=(3 * max_len, 9),
                             gridspec_kw={"right": 0.85})  # 右侧留空间放 colorbar
    fig.suptitle("Comparison of u component", fontsize=16)

    for t in range(input_len):
        im = axes[0, t].imshow(input_seq_np[t, 0], cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, t].set_title(f'Input v t={t}')
        axes[0, t].axis('off')

    for t in range(pred_len):
        im = axes[1, t].imshow(target_seq_np[t, 0], cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
        axes[1, t].set_title(f'True v t+{t}')
        axes[1, t].axis('off')

        im = axes[2, t].imshow(predicted_output_np[t, 0], cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
        axes[2, t].set_title(f'Pred v t+{t}')
        axes[2, t].axis('off')

    # for t in range(input_len):
    #     im = axes[0, t].imshow(np.rot90(input_seq_np[t, 0]), cmap='jet', vmin=vmin, vmax=vmax)
    #     axes[0, t].set_title(f'Input u t={t}')
    #     axes[0, t].axis('off')

    # for t in range(pred_len):
    #     im = axes[1, t].imshow(np.rot90(target_seq_np[t, 0]), cmap='jet', vmin=vmin, vmax=vmax)
    #     axes[1, t].set_title(f'True u t+{t}')
    #     axes[1, t].axis('off')

    #     im = axes[2, t].imshow(np.rot90(predicted_output_np[t, 0]), cmap='jet', vmin=vmin, vmax=vmax)
    #     axes[2, t].set_title(f'Pred u t+{t}')
    #     axes[2, t].axis('off')

    # 专门的 colorbar 轴
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)
    # colorbar 轴创建后，改为手动调整
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05, hspace=0.3, wspace=0.2)
    plt.show()

    # ========== v 分量 ==========
    fig, axes = plt.subplots(3, max_len, figsize=(3 * max_len, 9),
                             gridspec_kw={"right": 0.85})  # 右侧留空间放 colorbar
    fig.suptitle("Comparison of v component", fontsize=16)

    for t in range(input_len):
        im = axes[0, t].imshow(input_seq_np[t, 1], cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
        axes[0, t].set_title(f'Input v t={t}')
        axes[0, t].axis('off')

    for t in range(pred_len):
        im = axes[1, t].imshow(target_seq_np[t, 1], cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
        axes[1, t].set_title(f'True v t+{t}')
        axes[1, t].axis('off')

        im = axes[2, t].imshow(predicted_output_np[t, 1], cmap='jet', vmin=vmin, vmax=vmax, origin='lower')
        axes[2, t].set_title(f'Pred v t+{t}')
        axes[2, t].axis('off')

    # for t in range(input_len):
    #     im = axes[0, t].imshow(np.rot90(input_seq_np[t, 1]), cmap='jet', vmin=vmin, vmax=vmax)
    #     axes[0, t].set_title(f'Input v t={t}')
    #     axes[0, t].axis('off')

    # for t in range(pred_len):
    #     im = axes[1, t].imshow(np.rot90(target_seq_np[t, 1]), cmap='jet', vmin=vmin, vmax=vmax)
    #     axes[1, t].set_title(f'True v t+{t}')
    #     axes[1, t].axis('off')

    #     im = axes[2, t].imshow(np.rot90(predicted_output_np[t, 1]), cmap='jet', vmin=vmin, vmax=vmax)
    #     axes[2, t].set_title(f'Pred v t+{t}')
    #     axes[2, t].axis('off')

    # 专门的 colorbar 轴
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    # colorbar 轴创建后，改为手动调整
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05, hspace=0.3, wspace=0.2)
    plt.show()


def sample_test_batch_from_existing(
    source_path,
    input_len=5,
    pred_len=5,
    ep_idx=0,
    t_start=20,
    device='cpu'
):
    """
    从现有 h5 文件中采样一组数据（input_seq, other_seq, target_seq）
    可直接用于模型验证，无需另行保存。
    """
    total_len = input_len + pred_len

    with h5py.File(source_path, "r") as f:
        ep_keys = list(f.keys())
        ep_key = ep_keys[ep_idx]
        grp = f[ep_key]

        flow = grp["flow"][0]      # (T, 2, 32, 32)
        speed = grp["speed"][0]    # (T, 1)

        assert t_start + total_len <= flow.shape[0], \
            f"t_start + total_len exceeds data length: {t_start + total_len} > {flow.shape[0]}"

        input_seq = flow[t_start: t_start + input_len]             # (input_len, 2, 32, 32)
        target_seq = flow[t_start + input_len: t_start + total_len]  # (pred_len, 2, 32, 32)
        other_seq = speed[t_start: t_start + input_len]            # (input_len, 1)

        # === 裁剪到 [-2.5, 2.5] 保持稳定 ===
        input_seq = np.clip(input_seq, -2.5, 2.5)
        target_seq = np.clip(target_seq, -2.5, 2.5)
        other_seq = np.clip(other_seq, -2.5, 2.5)

        # 转 Tensor
        input_seq = torch.tensor(input_seq, dtype=torch.float32, device=device).unsqueeze(0)   # (1, input_len, 2, 32, 32)
        target_seq = torch.tensor(target_seq, dtype=torch.float32, device=device).unsqueeze(0) # (1, pred_len, 2, 32, 32)
        other_seq = torch.tensor(other_seq, dtype=torch.float32, device=device).unsqueeze(0)   # (1, input_len, 1)

        return input_seq, other_seq, target_seq

def test_main(_ep_id, _t_start):
    input_data, other_data, output_data = sample_test_batch_from_existing(source_path = test_path, ep_idx=_ep_id, t_start=_t_start)

    # print("============================================================================================")
    # set device to cpu or cuda
    device = torch.device('cpu')
    # if torch.cuda.is_available():
    #     device = torch.device('cuda:0')
    #     torch.cuda.empty_cache()
    #     print("Device set to : " + str(torch.cuda.get_device_name(device)))
    # else:
    #     print("Device set to : cpu")
    # print("============================================================================================")

    input_size = 4  # x, y, u, v
    hidden_size = 256
    num_layers = 2
    output_steps = 5



    model = load_model(model_path, input_size, hidden_size, num_layers, output_steps, device)
    plot_comparison_from_batch(input_seq = input_data, other_seq = other_data, target_seq = output_data, model = model, device = device, vmin=-2.5, vmax=2.5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test Flow Prediction Visualization")
    parser.add_argument("ep_id", type=int, nargs='?', default=0, help="Episode index to sample")
    parser.add_argument("t_start", type=int, nargs='?', default=10, help="Start time step for sampling")

    args = parser.parse_args()
    test_main(_ep_id=args.ep_id, _t_start=args.t_start)


