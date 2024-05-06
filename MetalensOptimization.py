"""
# -*- coding:utf-8 -*-
# @Project    : InsectMetalens
# @FileName   : MetalensOptimization.py
# @Author     : Spring
# @Time       : 2024/4/8 1:24
# @Description:
"""
import datetime
import os

import cupy as np
# import numpy as np
import pandas as pd

import yaml

# from Adam import Adam
from SGD import SGD
from utils import show_phases, show_group_delay, random_small_phase


class MetalensOptimization:
    def __init__(self, opt: any = None):
        """
        step1.初始化器件参数
        """
        self.data_file = opt.data_file
        self.method = opt.method
        self.device = opt.device
        self.show = opt.show

        # 光速,单位为 3e8 m/s, 0.3e9m/s
        self.speed_of_light = 0.3
        # 中心波长,单位为um,1e-6m, 10.6e-6um
        self.wavelength_center = 10.6
        # 波长列表
        self.lambda_list = [8, 9.3, 10.6, 11.3, 12]
        # 外径,单位为um
        self.outer_radius = 500.25
        # 内径,单位为um
        self.inner_radius = 0 * self.wavelength_center
        # 采样间隔，环带间隔, P_metasurface的间隔
        self.sample_interval = 7.25
        # 采样点个数
        self.n_sampling = 2048
        # 计算不同传播面
        self.nz = 60
        # 内径
        self.r_inner = 0 * self.wavelength_center
        # Z轴范围
        self.z_range = 30 * self.wavelength_center
        # 显示区域范围
        self.n_n = 100  # 设置 200 面阵大小
        # 波数
        self.k = 2 * np.pi / self.wavelength_center
        # 加载固定参数
        self.load_yaml()

        # =================下面是计算参数=================
        # 焦距
        self.focal_length = 130 * self.wavelength_center
        # 采样点
        self.sample_points = np.arange(self.n_sampling)
        # 角频率
        self.angular_frequency = 2 * np.pi * self.wavelength_center / self.speed_of_light
        # 计算环带数量
        self.num_r_outer = int(np.floor(self.outer_radius / self.sample_interval) + 1)
        # 创建环带半径数组
        self.radius_array = np.arange(0, (self.num_r_outer - 1) * self.sample_interval + self.sample_interval,
                                      self.sample_interval).astype(np.float32)
        # 计算初始半径
        self.R_0 = (self.num_r_outer - 1) * self.sample_interval
        # 计算每个环带的采样间隔
        self.Dx = 2 * self.outer_radius / self.n_sampling
        # =================初始化相位和群延迟=================
        # 初始相位
        self.phases = None
        # 初始群延迟
        self.group_delay = None
        # 执行初始化
        self.init_phases()

    def init_phases(self):
        """
        step2.初始化不同波长的相位,依据中心波长产生一个基准相位，其他4个波长的相位通过附加相位来计算
        :return:
        """
        print("002.初始化不同波长的相位..")
        # 使用中心波长生成基准相位
        base_phase = 2 * np.pi * (np.sqrt(self.R_0 ** 2 + self.focal_length ** 2) - np.sqrt(
            self.radius_array ** 2 + self.focal_length ** 2)) / self.wavelength_center
        # 给基准相位附加一个微小的扰动相位,范围为[-pi,pi]
        base_phase += random_small_phase(base_phase.shape)
        # 计算不同相位
        phase_shifts = [2, 1, 0, -1, -2]  # 5个rad，结构能够提供的相位范围
        self.phases = np.array(
            [base_phase + shift + random_small_phase(base_phase.shape) for shift in phase_shifts])

        # delta_phi 的范围为 [-2pi, 2pi]
        # 使用gd来约束 delta_phi

        if self.show:
            # 可视化
            show_phases(self.radius_array.get(), self.phases.get(), self.lambda_list)
            self.calculate_group_delay()
            show_group_delay(self.radius_array.get(), self.group_delay.get())

    def calculate_group_delay(self):
        """
        计算群延迟
        """
        # 计算第1个和第5个的相位差
        delta_phi = np.abs(self.phases[0, :] - self.phases[4, :])
        delta_omega = 2 * np.pi * self.speed_of_light * (1 / 8.0 - 1 / 12.0)
        # 群延迟 gd 的范围为 [0, 60]fs，1s=1e15fs
        self.group_delay = delta_phi / delta_omega

    def phase_optimization(self):
        """
        step3.优化相位
        """
        print("003.优化相位...")
        # 调用不同的优化方法(SGD,Adam)，优化相位，优化目标为target_group_delay，需要优化的参数为phases
        if self.method == "SGD":
            optimizer = SGD(learning_rate=0.01, max_iter=100000, tol=1e-4, verbose=True, metalens=self)
            optimizer.fit()
        elif self.method == "Adam":
            # adam = Adam()
            pass
            # return adam.optimize()
        elif self.method == "PSO":
            pass
        else:
            raise ValueError("Invalid optimization method. Please choose 'SGD' or 'Adam'.")

    def load_yaml(self):
        """
        从 YAML 文件中加载参数
        :return:
        """
        print("001.初始化器件参数..")
        with open(self.data_file, 'r', encoding='utf-8') as f:
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
            df = pd.DataFrame(self.phases.get())
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
