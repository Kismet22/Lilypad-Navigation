import pandas as pd
import math
import matplotlib.pyplot as plt

env_name = 'Flow_env_debug_zyp'
start_ID = 43
num = 1
repeat = 1


def run(ID, ax1):
    # 读取CSV文件
    file_path = f'./PPO_logs/{env_name}/PPO{env_name}_log_{ID}.csv'
    data = pd.read_csv(file_path)

    # 在同一坐标轴画avg_return的折线图
    ax1.plot(data['timestep'], data['avg_return_in_period'], 'b-', label='Average Return')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Average Return', color='b')
    ax1.tick_params('y', colors='b')
    ax1.grid(True)

    # 创建一个共享x轴的副坐标轴
    ax2 = ax1.twinx()
    ax2.plot(data['timestep'], data['action_std'], 'r-', label='Action Std')
    ax2.set_ylabel('Action Std', color='r')
    ax2.tick_params('y', colors='r')

    # 检测并标注显著变化
    for i in range(1, len(data['action_std'])):
        if data['action_std'][i] != data['action_std'][i - 1] or i == 1:
            ax2.annotate(f'{data["action_std"][i]:.3f}',
                         (data['timestep'][i], data['action_std'][i]),
                         textcoords="offset points",
                         xytext=(0, -10),
                         ha='center')

    # 设置子图标题为日志ID
    log_id = file_path.split('_')[-1].split('.')[0]
    ax1.set_title(f'Log ID: {log_id}')

    return ax1, ax2


if __name__ == '__main__':

    for n in range(repeat):
        rows = 1 if num <= 4 else 2
        IDs = list(range(start_ID, start_ID + num))

        # 创建一个大的图和设置子图布局
        cols = math.ceil(num / rows)
        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))  # 1行，num列
        handles, labels = [], []
        for i, ID in enumerate(IDs):

            if rows > 1:
                ax = axs[math.floor(i / cols)][i - math.floor(i / cols) * cols]
            else:
                ax = axs[i] if num > 1 else axs
            ax1, ax2 = run(ID, ax)
            if i == 0:  # Only take legend handles and labels from the first subplot
                handles1, labels1 = ax1.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                handles.extend(handles1 + handles2)
                labels.extend(labels1 + labels2)

        # 显示图例
        fig.legend(handles, labels, loc='upper right')
        plt.tight_layout()
        plt.show()
        start_ID += num
