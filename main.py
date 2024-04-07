# -*- coding:utf-8 -*-
# @Project    : InsectMicrolens
# @FileName   : main_pytorch.py
# @Author     : Spring
# @Time       : 2024/4/1 17:12
# @Description:
import datetime
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from FresnelDiffraction import FresnelDiffraction
from utils import show_contour, show_image, show_phases, show_group_delay, plot_delta_phi_vs_radius


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
        # 采样点个数
        self.N_sampling = 2048
        # 采样点
        self.sample_points = np.arange(self.N_sampling)
        # 角频率
        self.angular_frequency = 2 * np.pi * self.wavelength_center / self.speed_of_light
        # 计算环带数量Nr_outter
        self.num_rings = int(np.floor(self.outer_radius / self.sample_interval) + 1)
        # 创建环带半径数组
        self.radius_array = np.arange(0, (self.num_rings - 1) * self.sample_interval + self.sample_interval,
                                      self.sample_interval).astype(np.float32)
        # 计算初始半径
        self.R_0 = (self.num_rings - 1) * self.sample_interval
        # 计算每个环带的采样间隔
        self.Dx = 2 * self.outer_radius / self.N_sampling
        # 波长列表
        self.lambda_list = [8, 9.3, 10.6, 11.3, 12]
        # 初始相位变量
        self.phases = np.zeros((self.num_rings, len(self.lambda_list)), dtype=np.float32)
        # 初始群延迟
        self.group_delay = (np.sqrt(self.R_0 ** 2 + self.focal_length ** 2) - np.sqrt(
            self.radius_array ** 2 + self.focal_length ** 2)) / self.speed_of_light
        self.delta_phi_values = None

    def init_phases(self):
        """
        初始化波长的相位
        :return:
        """
        print("001.初始化相位...")
        for wavelength in self.lambda_list:
            # 计算数据，即每个环带的相位延迟
            phase = 2 * np.pi * (np.sqrt(self.R_0 ** 2 + self.focal_length ** 2) - np.sqrt(
                self.radius_array ** 2 + self.focal_length ** 2)) / wavelength
            self.phases[:, self.lambda_list.index(wavelength)] = phase
        # 可视化
        show_phases(self.radius_array, self.phases, self.lambda_list)
        show_group_delay(self.radius_array, self.group_delay)

    def calculate_group_delay(self, show=False):
        """
        计算群延迟
        :param show:
        :return:
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
        if show:
            # 可视化
            plot_delta_phi_vs_radius(self.radius_array, self.delta_phi_values)

    def phase_optimization(self, learning_rate=0.8, epochs=100, momentum=0.9, weight_decay=0.01,
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
        print("002.Optimized Phase......")
        self.calculate_group_delay(show=True)
        if 0 < self.group_delay.any() < target_group_delay:
            # 衍射计算
            print("评价指标：", self.asm())
            # 判断结果，执行优化操作
            # function()
            self.phases[:, 0] = self.phases[:, 0] - learning_rate * self.delta_phi_values[0]

    def asm(self):
        """
        角谱衍射/菲涅尔近场衍射
        :return:
        """
        print("003.角谱衍射...")
        # 将每个波长的相位形成一个面
        phase_plane = self.create_2d_plane(self.phases[:, 2])
        # 对形成的面进行衍射
        fresnel_diffraction = FresnelDiffraction(self.lambda_list[2], self.focal_length, phase_plane, self.Dx)
        fresnel_diffraction.compute_diffraction_field()
        fresnel_diffraction.compute_intensity_distribution()
        # 显示衍射后的图像
        fresnel_diffraction.show_diffraction_field()
        fresnel_diffraction.show_intensity_distribution()
        # 计算评价结果
        return fresnel_diffraction.compute_all()

    def create_2d_plane(self, phase_values: np.ndarray):
        """
        创建相位数组在二维平面上的分布
        :param phase_values: 相位值数组
        :return: 形成的二维平面
        """
        # 使用采样点数量创建 X 和 Y 矩阵，X 和 Y 分别表示二维平面上的横纵坐标
        X = self.Dx * np.ones((self.N_sampling, 1)) * (self.sample_points - self.N_sampling / 2 + 0.5)
        Y = X.transpose()

        # 计算每个采样点处的等效距离，即所属超表面结构单元中心位置到透镜中心的距离
        equivalent_distance = np.sqrt(
            np.ceil((np.abs(X) - 0.5 * self.sample_interval) / self.sample_interval) ** 2 + np.ceil(
                (np.abs(Y) - 0.5 * self.sample_interval) / self.sample_interval) ** 2) * self.sample_interval

        # 显示等效距离的等高线图
        show_contour(equivalent_distance, 'Equivalent Distance')

        # 计算基因索引数组，用于标识每个采样点所属的超表面结构单元
        gene_index_array = np.floor(equivalent_distance / self.sample_interval) + 1

        # 将超出指定环数的基因索引值限制为 num_rings
        gene_index_array[gene_index_array > self.num_rings] = self.num_rings

        # 显示基因索引数组的等高线图
        show_contour(gene_index_array, 'Gene Index Array')

        # 根据基因索引数组从相位值数组中获取相应的相位值
        phase_array = phase_values[gene_index_array.astype(int) - 1]

        # 显示相位数组的等高线图
        show_contour(phase_array, 'Phase Array')

        return phase_array

    def save_results(self):
        """
        保存结果
        """

        try:
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

            print("Results saved successfully.")
        except Exception as e:
            print(f"Error occurred while saving results: {str(e)}")

    def load_yaml(self):
        with open('data.yaml') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        # 从 YAML 数据中提取参数并初始化类属性
        parameters = data.get('parameters', {})
        sampling = data.get('sampling', {})
        calculations = data.get('calculations', {})

        self.speed_of_light = parameters.get('speed_of_light', 0.3)
        self.wavelength_center = parameters.get('wavelength_center', 10.6)
        self.focal_length = parameters.get('focal_length', 1366.8)
        self.outer_radius = parameters.get('outer_radius', 500.25)

        self.sample_interval = sampling.get('sample_interval', 7.25)
        self.N_sampling = sampling.get('N_sampling', 2048)
        self.lambda_list = sampling.get('lambda_list', [8, 9.3, 10.6, 11.3, 12])

        self.num_rings = calculations.get('num_rings', 70)
        self.R_0 = calculations.get('R_0', 500.25)
        self.Dx = calculations.get('Dx', 0.48828125)


if __name__ == '__main__':
    lens = InsectMetalens()

    # lens.init_phases()
    # lens.phase_optimization()
    # lens.save_results()
