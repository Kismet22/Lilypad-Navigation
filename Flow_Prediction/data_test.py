import h5py
import numpy as np
import matplotlib.pyplot as plt
import json

# file_path = "./train_data/flow_data_seed_1.h5"

# with h5py.File(file_path, "r") as f:
#     print("==== Keys ====")
#     for ep_key in f.keys():
#         print(f"Episode key: {ep_key}")
#         grp = f[ep_key]
#         print("  Fields:", list(grp.keys()))
#         for k in grp.keys():
#             try:
#                 dset = grp[k]
#                 print(f"    {k}: shape = {dset.shape}, dtype = {dset.dtype}")
#             except Exception as e:
#                 print(f"    {k}: Error reading dataset: {e}")

# folders = [
#     "./train_data/flow_data_1.h5",
#     "./train_data/flow_data_2.h5",
#     "./train_data/flow_data_3.h5",
#     "./train_data/flow_data_4.h5",
#     "./train_data/flow_data_5.h5",
#     "./train_data/flow_data_6.h5",
#     "./train_data/flow_data_7.h5",
#     "./train_data/flow_data_8.h5"
# ]

def print_datasets(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"  Dataset: {name}, shape: {obj.shape}")

folders = [
    f"./train_data_3/flow_data_{i}.h5"
    for i in range(1, 14)
]

total = 0

for path in folders:
    print(f"\nInspecting {path} ...")
    with h5py.File(path, 'r') as f:
        f.visititems(print_datasets)
# def collect_data_distribution(folders, keys=["pressure", "speed", "action"]):
#     data_dict = {k: [] for k in keys}

#     for file_path in folders:
#         with h5py.File(file_path, "r") as f:
#             for ep_key in f.keys():
#                 grp = f[ep_key]
#                 for k in keys:
#                     if k in grp:
#                         data = grp[k][0]  # shape (T, D)
#                         if np.isnan(data).any() or np.isinf(data).any():
#                             continue  # 跳过含nan/inf数据
#                         data_dict[k].append(data)

#     for k in keys:
#         if data_dict[k]:
#             data_dict[k] = np.concatenate(data_dict[k], axis=0)  # (Total_T, D)
#             print(f"{k} shape: {data_dict[k].shape}")
            
#             # 打印每个维度的 min 和 max
#             D = data_dict[k].shape[1]
#             for d in range(D):
#                 d_min = np.min(data_dict[k][:, d])
#                 d_max = np.max(data_dict[k][:, d])
#                 print(f"{k}[{d}]: min = {d_min:.6f}, max = {d_max:.6f}")
#         else:
#             data_dict[k] = None
#             print(f"{k} data not found.")

#     return data_dict






# TODO:start
# def collect_data_distribution(folders, keys=["pressure", "speed", "action"], save_path="./train_data/min_max.h5"):
#     data_dict = {k: [] for k in keys}
#     quantiles_dict = {}

#     for file_path in folders:
#         with h5py.File(file_path, "r") as f:
#             for ep_key in f.keys():
#                 grp = f[ep_key]
#                 for k in keys:
#                     if k in grp:
#                         data = grp[k][0]
#                         if np.isnan(data).any() or np.isinf(data).any():
#                             continue
#                         data_dict[k].append(data)

#     with h5py.File(save_path, "w") as f_out:
#         for k in keys:
#             if data_dict[k]:
#                 data_dict[k] = np.concatenate(data_dict[k], axis=0)
#                 print(f"{k} shape: {data_dict[k].shape}")

#                 D = data_dict[k].shape[1]
#                 grp = f_out.create_group(k)
#                 q05 = np.quantile(data_dict[k], 0.05, axis=0)
#                 q95 = np.quantile(data_dict[k], 0.95, axis=0)
#                 d_min = np.min(data_dict[k], axis=0)
#                 d_max = np.max(data_dict[k], axis=0)

#                 # 保存四个关键值到组
#                 grp.create_dataset("q05", data=q05, compression="gzip")
#                 grp.create_dataset("q95", data=q95, compression="gzip")
#                 grp.create_dataset("min", data=d_min, compression="gzip")
#                 grp.create_dataset("max", data=d_max, compression="gzip")

#                 for d in range(D):
#                     print(
#                         f"{k}[{d}]: min={d_min[d]:.6f}, max={d_max[d]:.6f}, "
#                         f"5%={q05[d]:.6f}, 95%={q95[d]:.6f}"
#                     )
#             else:
#                 print(f"{k} data not found.")

#     print(f"Quantiles saved to {save_path}")
#     return data_dict



# def plot_data_distribution(data_dict, bins=100, quantile_range=(0.005, 0.995)):
#     for k, data in data_dict.items():
#         if data is None:
#             continue
#         D = data.shape[1]

#         # 使用 constrained_layout 自动避免重叠
#         fig, axs = plt.subplots(D, 1, figsize=(10, 2 * D), constrained_layout=True)

#         for i in range(D):
#             data_i = data[:, i]
#             q_min = np.quantile(data_i, quantile_range[0])
#             q_max = np.quantile(data_i, quantile_range[1])

#             # 添加边距
#             margin = 0.01 * (q_max - q_min + 1e-6)
#             xmin = q_min - margin
#             xmax = q_max + margin

#             axs[i].hist(data_i, bins=bins, alpha=0.7, color='skyblue', edgecolor='black', range=(xmin, xmax))
#             axs[i].set_ylabel(f"{k}[{i}]")
#             axs[i].set_xlim(xmin, xmax)
#             axs[i].grid(True)

#             true_min = np.min(data_i)
#             true_max = np.max(data_i)
#             axs[i].set_title(
#                 f"{k}[{i}]: true_min={true_min:.3f}, true_max={true_max:.3f}, showing {quantile_range[0]*100:.1f}-{quantile_range[1]*100:.1f}%",
#                 fontsize=10
#             )

#         axs[-1].set_xlabel("Value")
#         fig.suptitle(f"{k} Distribution", fontsize=18)

#         plt.show()





# # ======================== RUN ========================

# data_dict = collect_data_distribution(folders)
# plot_data_distribution(data_dict)