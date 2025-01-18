from numpy import *
from scipy.linalg import solve_continuous_are


class heading_controller(object):
    def __init__(self, sample_time, factor=None, speed_adaption=.3, max_controll=10):
        self.sample_time = sample_time  # 采样时间
        # TODO:暂时不需要step_adaption
        self.speed_adaption = speed_adaption
        self.factor = 1  # 使用一个简化的系数
        self.max_controll = max_controll  # 最大舵角

        # controller params:
        self.KP = 5
        self.KI = 1
        self.KD = 30
        # self.KP = 5
        # self.KI = 1
        # self.KD = 30


        # initial values:
        self.summed_error = 0  # 误差的积分

    # 航向参数计算
    def calculate_controller_params(self, yaw_time_constant=None, store=True, Q=None, r=None):
        # TODO:旋转响应先设置为1试试
        if yaw_time_constant is None:
            try:
                yaw_time_constant = YAW_TIMECONSTANT  # 时间常数
            except:
                raise Exception("Yaw time constant is required.")

        # 系统线性化建模
        A = array([[0, 1, 0], [0, -1. / yaw_time_constant, 0], [-1, 0, 0]])  # 状态矩阵
        B = array([0, 1, 1])  # 输入，舵角

        if Q is None:
            Q = diag([1E-1, 1, 0.3])  # 权重矩阵
            r = ones((1, 1)) * 30  # 期望代价

        # 通过Riccati方程计算反馈增益
        P = solve_continuous_are(A, B[:, None], Q, r)
        K = sum(B[None, :] * P, axis=1) / r[0, 0]

        if store:
            # 设置PID参数
            self.KP = K[0]
            self.KD = K[1]
            self.KI = -K[2]

        return list(K)

    def controll(self, desired_heading, heading, yaw_rate, speed=1, drift_angle=0):
        # 比例误差：期望航向 - 当前航向
        heading_error = desired_heading - heading

        # 确保误差在 [-pi, pi] 范围内
        while heading_error > pi:
            heading_error -= 2 * pi
        while heading_error < -pi:
            heading_error += 2 * pi

        # 积分误差（含偏移量）
        self.summed_error += self.sample_time * (heading_error - drift_angle)

        # 物理建模系数
        # if speed < self.speed_adaption:
        #     speed = self.speed_adaption
        # factor2 = -1. / self.factor / speed ** 2
        factor2 = 1

        # 计算舵角加速度（控制量为舵角的二阶导数）
        # 通过PID控制公式计算舵角加速度
        rudder_angle_acceleration = factor2 * (self.KP * heading_error + self.KI * self.summed_error - self.KD * yaw_rate)

        # 控制量超出限制范围
        if abs(rudder_angle_acceleration) > self.max_controll:
            rudder_angle_acceleration = sign(rudder_angle_acceleration) * self.max_controll
            self.summed_error = (rudder_angle_acceleration / factor2 - (self.KP * heading_error - self.KD * yaw_rate)) / self.KI

        return rudder_angle_acceleration
