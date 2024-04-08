# -*- coding:utf-8 -*-
# @Project    : InsectMetalens
# @FileName   : MetalensOptimization.py
# @Author     : Spring
# @Time       : 2024/4/8 1:24
# @Description:
import datetime
import os

import numpy as np

import pandas as pd
import yaml

from Adam import Adam
from FresnelDiffraction import FresnelDiffraction
from SGD import SGD
from utils import show_contour, show_phases, show_group_delay, plot_delta_phi_vs_radius


class MetalensOptimization:
    def __init__(self, opt: any = None, *args, **kwargs):
        """
        step1.初始化器件参数
        """
        self.data_file = opt.data_file
        self.method = opt.method
        self.device = opt.device
        self.show = opt.show

        # 光速,单位为m/s
        self.speed_of_light = 0.3
        # 中心波长,单位为微米
        self.wavelength_center = 10.6
        # 波长列表
        self.lambda_list = [8, 9.3, 10.6, 11.3, 12]
        # 外径
        self.outer_radius = 500.25
        # 采样间隔，环带间隔
        self.sample_interval = 7.25
        # 采样点个数
        self.N_sampling = 2048
        # 加载固定参数
        self.load_yaml()
        # =================下面是计算参数=================
        # 焦距
        self.focal_length = 130 * self.wavelength_center
        # 采样点
        self.sample_points = np.arange(self.N_sampling)
        # 角频率
        self.angular_frequency = 2 * np.pi * self.wavelength_center / self.speed_of_light
        # 计算环带数量
        self.Nr_outter = int(np.floor(self.outer_radius / self.sample_interval) + 1)
        # 创建环带半径数组
        self.radius_array = np.arange(0, (self.Nr_outter - 1) * self.sample_interval + self.sample_interval,
                                      self.sample_interval).astype(np.float32)
        # 计算初始半径
        self.R_0 = (self.Nr_outter - 1) * self.sample_interval
        # 计算每个环带的采样间隔
        self.Dx = 2 * self.outer_radius / self.N_sampling
        # =================初始化相位和群延迟=================
        # 初始相位变量
        self.phases = np.zeros((self.Nr_outter, len(self.lambda_list)), dtype=np.float32)
        # 初始群延迟
        self.group_delay = (np.sqrt(self.R_0 ** 2 + self.focal_length ** 2) - np.sqrt(
            self.radius_array ** 2 + self.focal_length ** 2)) / self.speed_of_light
        self.delta_phi_values = None
        # 执行初始化
        self.init_phases()

    def init_phases(self):
        """
        step2.初始化不同波长的相位
        :return:
        """
        print("002.初始化不同波长的相位..")
        for wavelength in self.lambda_list:
            # 计算每个环带的相位延迟
            phase = 2 * np.pi * (np.sqrt(self.R_0 ** 2 + self.focal_length ** 2) - np.sqrt(
                self.radius_array ** 2 + self.focal_length ** 2)) / wavelength
            self.phases[:, self.lambda_list.index(wavelength)] = phase
        if self.show:
            # 可视化
            show_phases(self.radius_array, self.phases, self.lambda_list)
            show_group_delay(self.radius_array, self.group_delay)

    def calculate_group_delay(self):
        """
        计算群延迟
        """
        delta_omega = 2 * np.pi * self.speed_of_light * (1 / 8.0 - 1 / 12.0)
        delta_phi_0 = np.abs(self.phases[:, 0] - self.phases[:, 4])
        # 计算群延迟，要求0 < gd < 60
        self.group_delay = delta_phi_0 / delta_omega

        delta_phi_1 = np.abs(self.phases[:, 0] - self.phases[:, 2])
        delta_phi_2 = np.abs(self.phases[:, 1] - self.phases[:, 2])
        delta_phi_4 = np.abs(self.phases[:, 3] - self.phases[:, 2])
        delta_phi_5 = np.abs(self.phases[:, 4] - self.phases[:, 2])
        self.delta_phi_values = [delta_phi_0, delta_phi_1, delta_phi_2, delta_phi_4, delta_phi_5]
        if self.show:
            # 可视化
            plot_delta_phi_vs_radius(self.radius_array, self.delta_phi_values)

    def phase_optimization(self):
        """
        step3.优化相位
        """
        print("003.优化相位...")
        # 调用不同的优化方法(SGD,Adam)，优化相位，优化目标为target_group_delay，需要优化的参数为phases
        if self.method == "SGD":
            optimizer = SGD(learning_rate=0.01, max_iter=100000, tol=1e-4, verbose=True, metalens=self)
            optimizer.fit()
            # return sgd.fit()
        elif self.method == "Adam":
            adam = Adam()
            # return adam.optimize()
        else:
            raise ValueError("Invalid optimization method. Please choose 'SGD' or 'Adam'.")

    def load_yaml(self):
        """
        从 YAML 文件中加载参数
        :return:
        """
        print("001.初始化器件参数..")
        with open(self.data_file) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        # 从 YAML 数据中提取参数并初始化类属性
        parameters = data.get('parameters', {})
        for key, value in parameters.items():
            setattr(self, key, value)

        target_parameters = data.get('target_parameters', {})

        for key, value in target_parameters.items():
            setattr(self, key, value)

    def save_results(self):
        """
        step4.保存结果
        """
        try:
            print("004.保存优化结果...")
            # 保存数据
            df = pd.DataFrame(self.phases)
            # 创建保存结果的文件夹路径
            result_folder = 'result/' + datetime.datetime.now().strftime('%Y%m%d')

            # 检查结果文件夹是否存在，不存在则创建
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)

            # 日期加上时间戳作为文件名
            filename = result_folder + '/' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')

            # 保存为CSV文件
            df.to_csv(filename + ".csv", mode='a', header=False, index=False)
        except Exception as e:
            print(f"Error occurred while saving results: {str(e)}")

    def print_parameters(self):
        """
        打印参数
        :return:
        """
        print(vars(self))
