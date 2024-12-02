import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import argparse
from env import flow_field_env
import numpy as np


max_steps = 400
parser_1 = argparse.ArgumentParser()
args_1, unknown = parser_1.parse_known_args()
args_1.action_interval = 10
env = flow_field_env.foil_env(args_1, max_step=max_steps, include_flow=True)

from matplotlib.patches import Ellipse
import numpy as np

""""
def plot_env(ax, start_point, target, circles, agent_pos, history, agent_angle):
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
    
    # 当前智能体椭圆形状
    ellipse_height = circle["radius"]  # 2a = 8
    ellipse_width = ellipse_height / 1.5  # a/b = 1.5/1
    ellipse = Ellipse(
        xy=agent_pos, width=ellipse_width, height=ellipse_height, 
        angle=np.degrees(agent_angle), color='orange', alpha=0.7, zorder=4
    )
    ax.add_patch(ellipse)
    
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
"""

def plot_env(ax, start_point, target, circles, agent_pos, history, agent_angle, flow_x = None, flow_y = None, speed=None, sample_rate=10):
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
    # TODO:改变网格试试呢
    x = np.linspace(0, env.x_range, flow_x.shape[1])[::sample_rate]
    y = np.linspace(0, env.y_range, flow_x.shape[0])[::sample_rate]
    X, Y = np.meshgrid(x, y)
    sampled_flow_x = flow_x[::sample_rate, ::sample_rate]
    sampled_flow_y = flow_y[::sample_rate, ::sample_rate]
    ax.quiver(
    X, Y, sampled_flow_x * 2, sampled_flow_y * 2,  # 放大流场
    scale=60, scale_units='width', color='black', alpha=0.5, zorder=2,
    width=0.002, pivot='middle',
    headwidth=3, headaxislength=3
            )


    # ax.quiver(X, Y, sampled_flow_x, sampled_flow_y, scale=50, color='black', alpha=0.5, zorder=2)
    # 流线图
    # ax.streamplot(X, Y, sampled_flow_x, sampled_flow_y, color='black', linewidth=1, density=0.8, arrowsize=1.2)

    # 当前智能体椭圆形状
    ellipse_height = circle["radius"]  # 2a = 8
    ellipse_width = ellipse_height / 1.5  # a/b = 1.5/1
    ellipse = Ellipse(
        xy=agent_pos, width=ellipse_width, height=ellipse_height, 
        angle=np.degrees(agent_angle), color='orange', alpha=0.7, zorder=4
    )
    ax.add_patch(ellipse)
    
    # 在右下角显示智能体的坐标、速度和流场速度
    ax.text(
        0.95, 0.05, 
        (f"Agent (x, y): ({agent_pos[0]:.2f}, {agent_pos[1]:.2f})\n"
            f"Angle: {agent_angle:.2f}\n"
            f"Speed: {speed:.2f}"
         ),
        transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
    )

    # 设置图形属性
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
    agent_angle = env.state[5]
    history = [agent_pos]
    flow_x = env.u_flow
    flow_y = env.v_flow
    a_speed = env.speed
    f_speed = env.flow_speed
    plot_env(ax, start_point, target, circles, agent_pos, history, agent_angle, flow_x, flow_y, a_speed)
    #plot_env(ax, start_point, target, circles, agent_pos, history, agent_angle)
    status_label.config(text="Status: Ready", foreground="green")
    canvas.draw()

# Tkinter 界面
def update_trajectory():
    try:
        action = [float(entry_x.get()), float(entry_y.get()), float(entry_z.get())]
        _, _, terminated, truncated, _ = env.step(action)
        
        # 获取新的智能体位置和角度
        agent_pos = env.agent_pos
        agent_angle = env.state[5]
        flow_x = env.u_flow
        flow_y = env.v_flow
        a_speed = env.speed
        f_speed = env.flow_speed
        history.append(agent_pos)
        
        # 绘制环境和轨迹
        plot_env(ax, start_point, target, circles, agent_pos, history, agent_angle, flow_x, flow_y, a_speed)
        #plot_env(ax, start_point, target, circles, agent_pos, history, agent_angle)

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
flow_x = env.u_flow
flow_y = env.v_flow
a_speed = env.speed
history = [start_point]

# 创建 Tkinter 主窗口
root = tk.Tk()
root.title("Trajectory Controller")

# 创建 matplotlib 图
fig, ax = plt.subplots(figsize=(16, 9))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(row=0, column=0, columnspan=4)
plot_env(ax, start_point, target, circles, start_point, history, 0, flow_x, flow_y, a_speed)
#plot_env(ax, start_point, target, circles, start_point, history, 0)

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
