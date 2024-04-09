# -*- coding:utf-8 -*-
# @Project    : InsectMetalens
# @FileName   : utils.py
# @Author     : Spring
# @Time       : 2024/4/8 0:02
# @Description:
import matplotlib.pyplot as plt
import cupy as np

plt.rcParams['font.family'] = 'SimHei'

import torch


def check_gpu_memory_cpu_memory():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device)
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
        print(f"GPU: {gpu_name}, GPU Memory: {gpu_memory:.2f} GB")
    else:
        print("No GPU available.")

    import psutil

    cpu_count = psutil.cpu_count(logical=False)
    cpu_percent = psutil.cpu_percent()
    mem_info = psutil.virtual_memory()
    mem_total = mem_info.total / 1024 ** 3
    mem_used = mem_info.used / 1024 ** 3

    print(f"CPU Cores: {cpu_count}, CPU Usage: {cpu_percent}%")
    print(f"Memory: Total: {mem_total:.2f} GB, Used: {mem_used:.2f} GB")


def show_contour(data, title):
    """
    绘制等值线图
    :param data:
    :param title:
    :return:
    """
    data = data.get()
    plt.contourf(data)
    plt.colorbar()
    plt.title(title)
    plt.show()


def show_image(data, title):
    """
    绘制图像
    :param data:
    :param title:
    :return:
    """
    plt.imshow(data)
    plt.title(title)
    plt.show()


def show_group_delay(radius_array, group_delay):
    """
    绘制群延时随半径变化的图像
    :param radius_array:
    :param group_delay:
    :return:
    """
    plt.plot(radius_array, group_delay)
    plt.xlabel('Radius (um)')
    plt.ylabel('Group Delay (fs)')
    plt.title('Group Delay vs Radius')
    plt.grid(True)
    plt.show()


def show_phases(radius_array, phases, lambda_list):
    """
    绘制相位随半径变化的图像
    :param radius_array:
    :param phases:
    :param lambda_list:
    :return:
    """
    for i, wavelength in enumerate(lambda_list):
        plt.plot(radius_array, phases[i, :], label=f'lambda = {wavelength}')
    plt.xlabel('Radius (um)')
    plt.ylabel('Phase (rad)')
    plt.title('Phase vs Radius')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_delta_phi_vs_radius(radius_array, delta_phi_values):
    """
    绘制相位差随半径变化的图像
    :param delta_phi_values: 相位差值数组
    """
    for i in range(len(delta_phi_values)):
        plt.plot(radius_array, delta_phi_values[i], label=f'delta_phi_{i + 1}')
    plt.xlabel('Radius (um)')
    plt.ylabel('Phase Phi (rad)')
    plt.title('Delta Phi vs Radius')
    plt.legend()
    plt.show()


def random_small_phase(shape):
    """
    产生一个范围在[-0.1pi,0.1pi]的随机数
    :param shape:
    :return:
    """
    return np.random.uniform(-0.1 * np.pi, 0.1 * np.pi, shape)


import time


def calculate_runtime(func):
    """
    计算函数运行时间
    :param func:
    :return:
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"函数 {func.__name__} 的运行时长为 {runtime} 秒")
        return result

    return wrapper


if __name__ == '__main__':
    check_gpu_memory_cpu_memory()
