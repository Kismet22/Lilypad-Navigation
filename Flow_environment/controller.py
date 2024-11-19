import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import argparse
from env import flow_field_env
import numpy as np


max_steps = 400
parser_1 = argparse.ArgumentParser()
args_1, unknown = parser_1.parse_known_args()
args_1.action_interval = 10
env = flow_field_env.foil_env(args_1, max_step=max_steps)

# 绘制环境函数
def plot_env(ax, start_point, target, circles, agent_pos, history):
    ax.clear()
    # 起点
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
    # 当前智能体位置
    ax.scatter(*agent_pos, color='orange', label='Agent', zorder=5)

    # 在右下角显示智能体的坐标
    ax.text(0.95, 0.05, f"Agent (x, y): ({agent_pos[0]:.2f}, {agent_pos[1]:.2f})", 
            transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    ax.set_xlim(0, env.x_range)
    ax.set_ylim(0, env.y_range)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    ax.set_title("Moving Trajectory")
    plt.draw()

# 重置函数：重置环境和历史轨迹
def reset_env():
    global history, agent_pos
    env.reset()
    agent_pos = env.agent_pos
    history = [agent_pos]
    plot_env(ax, start_point, target, circles, agent_pos, history)
    status_label.config(text="Status: Ready", foreground="green")
    canvas.draw()

# Tkinter 界面
def update_trajectory():
    try:
        # 获取用户输入的动作，处理为浮动数值
        action = [float(entry_x.get()), float(entry_y.get()), float(entry_z.get())]
        _, _, terminated, truncated, _ = env.step(action)
        
        # 获取新的智能体位置，并记录到历史轨迹中
        agent_pos = env.agent_pos
        history.append(agent_pos)
        
        # 绘制环境和轨迹
        plot_env(ax, start_point, target, circles, agent_pos, history)

        # 根据是否结束状态更新提示信息
        if terminated:
            status_label.config(text="Status: Terminated", foreground="red")
        else:
            status_label.config(text="Status: Running", foreground="green")

        # 更新画布显示
        canvas.draw()
        
    except ValueError:
        status_label.config(text="Invalid input. Please enter numeric values.", foreground="red")

# 环境初始化
env.reset()
start_point = env.agent_pos
target = env.target_position
circles = env.circles
history = [start_point]

# 创建 Tkinter 主窗口
root = tk.Tk()
root.title("Trajectory Controller")

# 创建 matplotlib 图
fig, ax = plt.subplots(figsize=(16, 9))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(row=0, column=0, columnspan=4)
plot_env(ax, start_point, target, circles, start_point, history)

# 控制面板
control_frame = tk.Frame(root)
control_frame.grid(row=1, column=0, columnspan=4, pady=10)

# 输入框和标签
tk.Label(control_frame, text="a_x:").grid(row=0, column=0, padx=5)
entry_x = ttk.Entry(control_frame, width=10)
entry_x.grid(row=0, column=1, padx=5)

tk.Label(control_frame, text="a_y:").grid(row=0, column=2, padx=5)
entry_y = ttk.Entry(control_frame, width=10)
entry_y.grid(row=0, column=3, padx=5)

tk.Label(control_frame, text="a_w:").grid(row=0, column=4, padx=5)
entry_z = ttk.Entry(control_frame, width=10)
entry_z.grid(row=0, column=5, padx=5)

# 按钮用于更新轨迹
update_button = ttk.Button(control_frame, text="Update", command=update_trajectory)
update_button.grid(row=0, column=6, padx=10)

# 重置按钮
reset_button = ttk.Button(control_frame, text="Reset", command=reset_env)
reset_button.grid(row=0, column=7, padx=10)

# 状态显示标签
status_label = tk.Label(root, text="Status: Ready", foreground="green")
status_label.grid(row=2, column=0, columnspan=4, pady=10)

# 启动 Tkinter 主循环
root.mainloop()
