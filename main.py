# -*- coding:utf-8 -*-
# @Project    : InsectMicrolens
# @FileName   : main_pytorch.py
# @Author     : Spring
# @Time       : 2024/4/1 17:12
# @Description:
import numpy as np
import matplotlib.pyplot as plt


class InsectMetalens:
    """
    透镜优化
    """

    def __init__(self):
        """
        初始化参数
        """
        # 光速,单位为m/s
        self.speed_of_light = 0.3
        # 中心波长,单位为微米
        self.wavelength_center = 10.6
        # 焦距
        self.focal_length = 130 * self.wavelength_center
        # 外径
        self.outer_radius = 500.25
        # 采样间隔，环带间隔
        self.sample_interval = 7.25
        # 角频率
        self.angular_frequency = 2 * np.pi * self.wavelength_center / self.speed_of_light
        # 计算环带数量
        self.num_rings = int(np.floor(self.outer_radius / self.sample_interval) + 1)
        # 创建环带半径数组
        self.radius_array = np.arange(0, (self.num_rings - 1) * self.sample_interval + self.sample_interval,
                                      self.sample_interval).astype(np.float32)
        # 计算初始半径
        self.R_0 = (self.num_rings - 1) * self.sample_interval

        # 波长列表
        self.lambda_list = [8, 9.3, 10.6, 11.3, 12]
        # 初始相位变量
        self.phases = np.zeros((self.num_rings, len(self.lambda_list)), dtype=np.float32)
        # 初始群延迟
        self.group_delay = (np.sqrt(self.R_0 ** 2 + self.focal_length ** 2) - np.sqrt(
            self.radius_array ** 2 + self.focal_length ** 2)) / self.speed_of_light

    def init_phases(self):
        """
        初始化波长的相位
        :return:
        """
        for wavelength in self.lambda_list:
            # 计算数据，即每个环带的相位延迟
            phase_delay = 2 * np.pi * (np.sqrt(self.R_0 ** 2 + self.focal_length ** 2) - np.sqrt(
                self.radius_array ** 2 + self.focal_length ** 2)) / wavelength
            self.phases[:, self.lambda_list.index(wavelength)] = phase_delay

    def calculate_group_delay(self):
        delta_omega = 2 * np.pi * self.speed_of_light * (1 / 8.0 - 1 / 12.0)
        delta_phi = self.phases[:, 0] - self.phases[:, 4]
        # 计算群延迟，要求0 < gd < 60
        self.group_delay = delta_phi / delta_omega

        delta_phi_1 = np.abs(self.phases[:, 0] - self.phases[:, 2])
        delta_phi_2 = np.abs(self.phases[:, 1] - self.phases[:, 2])
        delta_phi_4 = np.abs(self.phases[:, 3] - self.phases[:, 2])
        delta_phi_5 = np.abs(self.phases[:, 4] - self.phases[:, 2])

        plt.plot(self.radius_array, delta_phi_1, label='delta_phi_1')
        plt.plot(self.radius_array, delta_phi_2, label='delta_phi_2')
        plt.plot(self.radius_array, delta_phi_4, label='delta_phi_4')
        plt.plot(self.radius_array, delta_phi_5, label='delta_phi_5')
        plt.xlabel('Radius (um)')
        plt.ylabel('Phase Phi (rad)')
        plt.title('Delta Phi vs Radius')
        plt.legend()
        plt.show()
        return self.group_delay

    def plot_phases(self):
        """
        绘制相位
        :return:
        """
        for i, wavelength in enumerate(self.lambda_list):
            plt.plot(self.radius_array, self.phases[:, i], label=f'lambda = {wavelength}')
        plt.xlabel('Radius (um)')
        plt.ylabel('Phase Delay (rad)')
        plt.title('Phase Delay vs Radius')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_group_delay(self):
        """
        绘制群延迟
        :return:
        """
        plt.plot(self.radius_array, self.group_delay)
        plt.xlabel('Radius (um)')
        plt.ylabel('Group Delay (fs)')
        plt.title('Group Delay vs Radius')
        plt.grid(True)
        plt.show()

    def sgd_phase_optimization(self, learning_rate=0.8, epochs=100, momentum=0.9, weight_decay=0.01,
                               target_group_delay=60):
        """
        梯度下降优化相位
        :param learning_rate:
        :param epochs:
        :param momentum:
        :param weight_decay:
        :param target_group_delay:
        :return:
        """
        print("Optimized Phase......")
        delta_phi = self.phases[:, 0] - self.phases[:, 4]
        self.phases[:, 0] = self.phases[:, 0] - learning_rate * delta_phi
        if 0 < self.calculate_group_delay().all() < target_group_delay:
            return self.phases

    def asm(self, phases):
        pass


if __name__ == '__main__':
    lens = InsectMetalens()
    lens.init_phases()
    # 优化前
    lens.plot_phases()
    lens.plot_group_delay()
    lens.sgd_phase_optimization()
    # 优化后
    lens.plot_phases()
    lens.plot_group_delay()
    print(lens.phases.shape)
