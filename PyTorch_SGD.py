"""
# -*- coding:utf-8 -*-
# @Project    : InsectMetalens
# @FileName   : Pytorch_SGD.py
# @Author     : SP
# @Time       : 2024/5/2 0:44
# @Description: 
"""
import torch
import numpy as np

# 假设你的相位数据是一个(5, 92)的张量，可以随机初始化或从其他来源加载
phase_data = torch.randn(5, 92, requires_grad=True)

# 定义优化算法的超参数
learning_rate = 0.01
num_iterations = 100


# 自定义优化算法（用于优化相位编码）
def optimize_phase(phase_data, learning_rate, num_iterations):
    for i in range(num_iterations):
        # 将梯度缓存设置为零
        phase_data.grad = None

        # 执行自定义的优化步骤
        # 这里使用了一个简单的示例：将相位数据乘以一个小的常数
        optimized_phase_data = phase_data * 0.9

        # 计算损失（可以根据你的任务定义适当的损失函数）
        loss = custom_loss_function(optimized_phase_data)

        # 执行反向传播
        loss.backward()

        # 更新相位数据
        phase_data.data -= learning_rate * phase_data.grad

        # 打印损失
        print(f"Iteration {i + 1}, Loss: {loss.item()}")

    return phase_data


def custom_loss_function(phases):
    current_params = self.asm.compute_all(phases)
    fwhm_loss = (np.max(current_params["FWHM"]) - self.target_params["TargetFWHM"]) ** 2
    sidelobe_ratio_loss = (np.max(current_params["side_lobe_ratio"]) - self.target_params["TargetSideLobeRatio"]) ** 2
    peak_intensity_loss = (np.max(current_params["intensity_peak"]) - self.target_params["TargetPeakIntensity"]) ** 2
    focal_offset_loss = (np.max(current_params["focal_offset"]) - self.target_params["TargetFocalOffset"]) ** 2
    total_loss = fwhm_loss + sidelobe_ratio_loss + peak_intensity_loss + focal_offset_loss
    return total_loss


# 调用优化函数进行相位编码优化
optimized_phase_data = optimize_phase(phase_data, learning_rate, num_iterations)

# 输出优化后的相位编码
print(optimized_phase_data)
# 输出优化后的相位编码
print(optimized_phase_data.shape)
