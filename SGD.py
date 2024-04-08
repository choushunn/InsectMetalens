# -*- coding:utf-8 -*-
# @Project    : InsectMetalens
# @FileName   : SGD.py
# @Author     : Spring
# @Time       : 2024/4/2 15:38
# @Description:
# from FresnelDiffraction import FresnelDiffraction

import numpy as np
from tqdm import tqdm


class SGD:
    def __init__(self, learning_rate=0.05, max_iter=1000, tol=1e-4, verbose=False, metalens=None):
        self.learning_rate = learning_rate  # 学习率
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛阈值
        self.verbose = verbose  # 是否打印迭代信息
        # 获取透镜参数
        self.metalens = metalens
        self.phases = self.metalens.phases  # 初始化相位数组
        self.group_delay = self.metalens.group_delay  # 初始化群延迟数组
        self.target_group_delay = 60  # 初始化目标群延迟

    def fit(self):
        p_bar = tqdm(total=self.max_iter)
        for i in range(self.max_iter):
            self.optimize_step()  # 执行一步优化
            if self.verbose:
                p_bar.update(1)
                p_bar.set_description(f"第 {i + 1} 次迭代, loss: {self.calculate_loss()}")

    def optimize_step(self):
        gradient = self.compute_gradient()  # 计算梯度
        # print(gradient[:, 0])
        self.update_phases(gradient)  # 更新相位

    def compute_gradient(self):
        # 使用有限差分法计算梯度

        grad = np.zeros_like(self.phases)
        # print(len(self.phases), len(self.phases[0]))
        for i in range(len(self.phases)):
            for j in range(len(self.phases[i])):
                original_phase = self.phases[i, j]

                # Compute the gradient using finite differences
                epsilon = 1e-8
                self.phases[i, j] = original_phase + epsilon
                loss_plus = self.calculate_loss()

                self.phases[i, j] = original_phase - epsilon
                loss_minus = self.calculate_loss()

                grad[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

                # Restore the original phase
                self.phases[i, j] = original_phase

        return grad

    def calculate_loss(self):
        # 使用平方误差作为损失函数
        self.calculate_group_delay()
        return np.sum((self.group_delay - self.target_group_delay) ** 2)

    def calculate_group_delay(self):
        # 重新计算 target_group_delay
        delta_omega = 2 * np.pi * 0.3 * (1 / 8.0 - 1 / 12.0)
        delta_phi_0 = np.abs(self.phases[:, 0] - self.phases[:, 4])
        # 计算群延迟，要求0 < gd < 60
        self.group_delay = delta_phi_0 / delta_omega

    def update_phases(self, gradient):
        self.phases -= self.learning_rate * gradient  # 根据梯度更新相位


if __name__ == '__main__':
    phases = np.array([[0.1, 0.2, 0.3],
                       [0.4, 0.5, 0.6]])

    group_delay = np.array([10, 15])
    target_group_delay = np.array([5, 10])

    # 创建SGD对象
    optimizer = SGD(learning_rate=0.01, max_iter=100000, tol=1e-4, verbose=True)

    # 使用SGD进行优化
    optimizer.fit(phases, group_delay, target_group_delay)

    # 获取优化后的相位
    optimized_phases = optimizer.phases

#
# # 将每个波长的相位形成一个面
# phase_plane = self.create_2d_plane(self.phases[:, 2])
# # 对形成的面进行衍射
# fresnel_diffraction = FresnelDiffraction(self.lambda_list[2], self.focal_length, phase_plane, self.Dx)
# fresnel_diffraction.compute_diffraction_field()
# fresnel_diffraction.compute_intensity_distribution()
# # 显示衍射后的图像
# fresnel_diffraction.show_diffraction_field()
# fresnel_diffraction.show_intensity_distribution()
# # 计算评价结果
# fresnel_diffraction.compute_all()
# # 计算群延迟
# self.calculate_group_delay(show=True)
# # 如果群延迟不在指定范围内，则执行优化操作
# if not (0 < self.group_delay.all() < target_group_delay):
#     # 衍射计算
#     print("评价指标：", self.asm())
#     # 判断结果，执行优化操作
#     # function()
#     # 优化操作，调整相位值
#     self.phases[:, 0] = self.phases[:, 0] - 0.7 * self.delta_phi_values[0]
# phase_optimizer = PhaseOptimizer(self.phases, self.group_delay, self.target_group_delay)
