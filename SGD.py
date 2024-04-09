# -*- coding:utf-8 -*-
# @Project    : InsectMetalens
# @FileName   : SGD.py
# @Author     : Spring
# @Time       : 2024/4/2 15:38
# @Description:


import numpy as np
from tqdm import tqdm

from FresnelDiffraction import FresnelDiffraction


class SGD:
    def __init__(self, learning_rate=0.05, max_iter=1000, tol=1e-4, verbose=False, metalens=None):
        self.learning_rate = learning_rate  # 学习率
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛阈值
        self.verbose = verbose  # 是否打印迭代信息
        # 获取透镜参数
        self.metalens = metalens
        self.phases = self.metalens.phases
        self.asm = FresnelDiffraction(self.metalens)
        self.target_params = None

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
        # 修改每个波长的相位，计算梯度
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
        # 是当前的评价参数
        current_params = self.asm.compute_all(self.phases)
        print(current_params)
        # exit()
        # todo:设计当前参数和目标参数的loss函数
        # return np.sum((current_params - self.target_params) ** 2)
        return 0.1

    def update_phases(self, gradient):
        self.phases -= self.learning_rate * gradient  # 根据梯度更新相位
