# -*- coding:utf-8 -*-
# @Project    : InsectMetalens
# @FileName   : utils.py
# @Author     : Spring
# @Time       : 2024/4/8 0:02
# @Description:
import matplotlib.pyplot as plt


def show_contour(data, title):
    plt.contourf(data)
    plt.colorbar()
    plt.title(title)
    plt.show()


def show_image(data, title):
    plt.imshow(data)
    plt.title(title)
    plt.show()


def show_group_delay(radius_array, group_delay):
    plt.plot(radius_array, group_delay)
    plt.xlabel('Radius (um)')
    plt.ylabel('Group Delay (fs)')
    plt.title('Group Delay vs Radius')
    plt.grid(True)
    plt.show()


def show_phases(radius_array, phases, lambda_list):
    for i, wavelength in enumerate(lambda_list):
        plt.plot(radius_array, phases[:, i], label=f'lambda = {wavelength}')
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
