# -*- coding:utf-8 -*-
# @Project    : InsectMetalens
# @FileName   : Adam.py
# @Author     : Spring
# @Time       : 2024/4/2 15:39
# @Description:
import numpy as np
from tqdm import tqdm


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate  # 学习率
        self.beta1 = beta1  # 第一个指数衰减的参数
        self.beta2 = beta2  # 第二个指数衰减的参数
        self.epsilon = epsilon  # 避免除零的小量
        self.m = None  # 存储一阶矩量
        self.v = None  # 存储二阶矩量
        self.t = 0  # 时间步数，用于纠正偏差

    def initialize_params(self, params):
        # 初始化一阶和二阶矩量
        self.m = np.zeros_like(params)
        self.v = np.zeros_like(params)

    def optimize(self, params, target_group_delay, max_iter=1000000, tol=1e-4, verbose=False):
        target_group_delay = target_group_delay[:, np.newaxis]  # 添加一个新的维度到目标群延迟数组
        p_bar = tqdm(total=max_iter)
        for i in range(max_iter):
            group_delay = self.calculate_group_delay(params)
            gradients = self.compute_gradient(group_delay, target_group_delay)

            self.t += 1
            self.m = self.beta1 * self.m + (1 - self.beta1) * gradients  # 更新一阶矩量
            self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)  # 更新二阶矩量

            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)

            params -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            loss = self.calculate_loss(group_delay, target_group_delay)

            p_bar.update(1)

            if verbose:
                p_bar.set_description(f"Iteration {i + 1}, Loss: {loss}")

            if np.linalg.norm(gradients) < tol:
                if verbose:
                    print("收敛.")
                break

        return params

    def calculate_group_delay(self, params):
        # 计算相位数组的群延迟
        return np.sum(params, axis=1)

    def compute_gradient(self, group_delay, target_group_delay):
        # 计算梯度
        return 2 * (group_delay - target_group_delay)

    def calculate_loss(self, group_delay, target_group_delay):
        # 计算损失
        return np.sum((group_delay - target_group_delay) ** 2)


if __name__ == '__main__':
    adam_optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
    phases = np.random.rand(10, 10)
    adam_optimizer.initialize_params(phases)
    target_group_delay = np.random.rand(10)
    optimized_params = adam_optimizer.optimize(phases, target_group_delay, verbose=True)
