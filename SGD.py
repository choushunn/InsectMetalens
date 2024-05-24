# -*- coding:utf-8 -*-
# @Project    : InsectMetalens
# @FileName   : SGD.py
# @Author     : Spring
# @Time       : 2024/4/2 15:38
# @Description:
import os
from datetime import datetime

import pandas as pd
import torch
# import numpy as np
from tqdm import tqdm
import cupy as np
from FresnelDiffraction import FresnelDiffraction

# 随机数种子
np.random.seed(2024)


class SGD:
    def __init__(self, learning_rate=0.05, max_iter=1000, tol=1e-6, verbose=False, metalens=None):
        self.learning_rate = learning_rate  # 学习率
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛阈值
        self.verbose = verbose  # 是否打印迭代信息
        # 获取透镜参数
        self.metalens = metalens
        self.phases = self.metalens.phases
        self.asm = FresnelDiffraction(self.metalens)
        self.target_params = {
            "TargetFWHM": 1.5,
            "TargetSideLobeRatio": 0.05,
            "TargetPeakIntensity": 2000,
            "TargetFocalOffset": 0.00001
        }
        self.total_loss = 0
        self.file_name = datetime.now().strftime("result/data_%Y%m%d_%H%M%S.csv")
        self.file_exists = os.path.exists(self.file_name)

    def fit(self):
        iteration = 0
        gradient_norm = np.inf
        params = self.phases
        prev_loss = np.inf  # 用于判断收敛性的前一次损失值
        tol = self.tol
        p_bar = tqdm(total=self.max_iter)
        while iteration < self.max_iter:
            # 随机选择样本
            indices = np.random.choice(len(params), size=len(params), replace=False)

            # 计算梯度
            gradient = self.compute_gradient(params, indices)

            # 更新参数
            params -= self.learning_rate * gradient

            # 计算损失函数
            cur_params, loss = self.loss_function(params)
            self.save_results(cur_params)
            # 判断收敛性,loss越小越好
            if abs(loss - prev_loss) < tol:
                break

            prev_loss = loss

            p_bar.set_description(f"第 {iteration + 1} 次迭代, loss: {self.total_loss}")
            p_bar.update(1)
            iteration += 1

    def compute_gradient(self, params, indices):
        """
        计算梯度
        :param params:
        :param indices:
        :return:
        """
        # 生成[0,pi]的随机数
        epsilon = 1e-3 * np.random.uniform(0, np.pi)  # 微小扰动的大小
        gradient = np.zeros_like(params)  # 初始化梯度为零向量
        # 计算每个相位的梯度
        for i in indices:
            # 对第i个参数进行微小扰动
            params_plus = params.copy()
            params_plus[i] += epsilon

            # 计算扰动后的损失函数值
            _, loss_plus = self.loss_function(params_plus)

            # 对第i个参数进行微小扰动
            params_minus = params.copy()
            params_minus[i] -= epsilon

            # 计算扰动后的损失函数值
            _, loss_minus = self.loss_function(params_minus)

            # 计算第i个参数的梯度
            gradient[i] = (loss_plus - loss_minus) / (2 * epsilon)

        return gradient

    def loss_function(self, phases):
        """
        计算损失函数
        :param phases:
        :return:
        """
        # 当前的评价参数
        current_params = self.asm.compute_all(phases)

        # 计算差异
        diff_fwhm = np.max(current_params["FWHM"]) - self.target_params["TargetFWHM"]
        diff_sidelobe_ratio = np.max(current_params["side_lobe_ratio"]) - self.target_params["TargetSideLobeRatio"]
        diff_peak_intensity = np.min(current_params["intensity_peak"]) - self.target_params["TargetPeakIntensity"]
        diff_focal_offset = np.max(current_params["focal_offset"]) - self.target_params["TargetFocalOffset"]
        # 计算焦深偏差越小越好
        DOF_loss = np.std(current_params["DOF"])
        intensity_sum_loss = 10 ** 4 / np.min(current_params["intensity_sum"])  # max min()
        # 计算损失函数
        fwhm_loss = diff_fwhm ** 2
        sidelobe_ratio_loss = diff_sidelobe_ratio ** 2
        peak_intensity_loss = diff_peak_intensity ** 2
        focal_offset_loss = diff_focal_offset ** 2
        # 损失权重
        d = [10, 0.1, 10, 2, 0.01, 1]
        total_loss = d[0] * fwhm_loss + d[1] * sidelobe_ratio_loss + d[2] * peak_intensity_loss + d[3] * focal_offset_loss + d[
            4] * DOF_loss + d[
                         5] * intensity_sum_loss
        self.total_loss = total_loss

        return current_params, total_loss

    def save_results(self, cur_params, code=0):
        """
        保存结果
        :param cur_params:保存的参数
        :param code:
        :return:
        """
        # 一次优化后，保存当前的参数
        df = pd.DataFrame({key: np.asnumpy(value) for key, value in cur_params.items()})
        if not self.file_exists:
            df.to_csv(self.file_name, mode='w', header=True, index=False)
            self.file_exists = True
        else:
            df.to_csv(self.file_name, mode='a', header=False, index=False)
        # print(f"数据已追加保存到 {self.file_name}")

    def update_learning_rate(self, loss, learning_rate=0.01, decay_factor=0.1, min_learning_rate=1e-6, patience=0, epoch=0,
                             best_loss=float('inf')):
        """
        更新学习率
        :param loss:
        :param learning_rate:
        :param decay_factor:
        :param min_learning_rate:
        :param patience:
        :param epoch:
        :param best_loss:
        :return:
        """
        if loss < best_loss:
            best_loss = loss
            patience = 0
        else:
            patience += 1
            decay_patience = 10  # 衰减耐心的阈值
            if patience > decay_patience:
                learning_rate *= decay_factor
                learning_rate = max(learning_rate, min_learning_rate)
                patience = 0

        return learning_rate, patience, best_loss
