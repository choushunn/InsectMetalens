# -*- coding:utf-8 -*-
# @Project    : InsectMetalens
# @FileName   : FresnelDiffraction.py
# @Author     : Spring
# @Time       : 2024/4/7 17:06
# @Description:
import numpy as np
from matplotlib import pyplot as plt

from utils import show_contour


class FresnelDiffraction:
    def __init__(self, wavelength: float, distance: float, phase_plane: np.ndarray, grid_size: float,
                 N_sampling: int = 2048,
                 **kwargs) -> None:
        """
        初始化Fresnel衍射的参数
        :param wavelength: 波长
        :param distance: 衍射距离
        :param phase_plane: 相位面
        :param grid_size: 网格大小
        :param num_samples: 采样点数
        :param kwargs: 其他参数
        """
        self.wavelength = wavelength
        self.distance = distance
        # 波数
        self.k = 2 * np.pi / wavelength
        self.phase_plane = phase_plane
        self.grid_size = grid_size
        self.N_sampling = N_sampling
        # 采样点
        self.sample_points = np.arange(self.N_sampling)
        # 相移
        self.phase_shift = None
        # 衍射场
        self.diffraction_field = None
        # 强度分布
        self.intensity_distribution = None
        self.nn = 200

    def fourier_transform_2d(self, input_data: np.ndarray, flag: int = 1) -> np.ndarray:
        """
        二维傅里叶变换
        :param input_data:输入数据
        :param flag:1为正变换，-1为逆变换
        :return:变换后的数据
        """

        a = np.exp(1j * 2 * np.pi / self.N_sampling * (self.N_sampling / 2 - 0.5) * self.sample_points)
        # 计算二维外积
        A = np.outer(a, a)
        # 计算调制因子
        C = np.exp(-1j * 2 * np.pi / self.N_sampling * (self.N_sampling / 2 - 0.5) ** 2 * 2) * A
        trans_data = None
        if flag == 1:
            # 执行正变换
            trans_data = self.grid_size ** 2 * C * np.fft.fft2(A * input_data)
        if flag == -1:
            # 执行逆变换
            trans_data = (1. / (self.N_sampling * self.grid_size)) ** 2 * self.N_sampling ** 2 * np.conj(
                C) * np.fft.ifft2(
                np.conj(A) * input_data)
        return trans_data

    def diffraction_2d_trans_polar(self, initial_field, refractive_index: float = 1) -> np.ndarray:
        """
        二维极化衍射传输
        :param initial_field: 初始场
        :param refractive_index: 折射率，默认为1
        :return: 衍射传输后的场
        """
        # 计算频率
        freq = 1. / (self.N_sampling * self.grid_size) * (self.sample_points - self.N_sampling / 2 + 0.5)
        freq_x = np.outer(freq, np.ones(self.N_sampling))
        freq_y = freq_x.T

        # 计算频率因子
        fza = ((refractive_index / self.wavelength) ** 2 - freq_x ** 2 - freq_y ** 2).astype(np.complex128)
        # 计算极化因子
        fz = np.sqrt(fza)

        # 执行正向傅里叶变换
        SpectrumX = self.fourier_transform_2d(initial_field, 1)

        # 对频谱进行相位调制
        SpectrumX = SpectrumX * np.exp(1j * 2 * np.pi * fz * self.distance)

        # 执行逆向傅里叶变换
        Ex = self.fourier_transform_2d(SpectrumX, -1)

        # 选择感兴趣的区域
        Ex = Ex[int(self.N_sampling / 2 - self.nn):int(self.N_sampling / 2 + 2 + self.nn),
             int(self.N_sampling / 2 - self.nn):int(self.N_sampling / 2 + 2 + self.nn)]

        return Ex

    def compute_diffraction_field(self):
        """
        计算衍射场
        :return: 衍射场
        """
        # 计算相移
        self.phase_shift = np.exp(1j * self.k * self.distance)

        # 初始化初始场
        initial_amplitude = np.ones(self.phase_plane.shape) * 0.66
        initial_field = initial_amplitude * np.exp(1j * self.phase_plane)

        # 计算衍射场
        self.diffraction_field = self.diffraction_2d_trans_polar(initial_field, 1)

    def compute_intensity_distribution(self):
        """
        计算菲涅尔衍射的强度分布
        :return: 强度分布
        """
        if self.diffraction_field is None:
            raise ValueError("请先计算衍射场")
        self.intensity_distribution = np.abs(self.diffraction_field) ** 2

    def compute_FWHM(self) -> float:
        """
        计算半高全宽（Full Width at Half Maximum）
        :return: 半高全宽
        """
        max_intensity = np.max(self.intensity_distribution)
        half_max_intensity = max_intensity / 2.0

        # 找到超过半高全宽的点
        indices_above_half_max = np.where(self.intensity_distribution >= half_max_intensity)[0]

        # 获取第一个和最后一个超过半高全宽的点
        first_index = indices_above_half_max[0]
        last_index = indices_above_half_max[-1]

        # 计算半高全宽
        FWHM = last_index - first_index

        return FWHM

    def compute_SideLobeRatio(self) -> float:
        """
        计算旁瓣比
        :return: 旁瓣比
        """
        # 找到峰值强度
        peak_intensity = np.max(self.intensity_distribution)
        intensity_distribution = self.intensity_distribution.copy()
        # 将峰值强度之外的部分置零
        intensity_distribution[intensity_distribution < peak_intensity] = 0
        # 找到第二大的值（即第一个旁瓣）
        second_max_intensity = np.sort(intensity_distribution.flatten())[-2]
        # 计算旁瓣比
        SideLobeRatio = peak_intensity / second_max_intensity
        return SideLobeRatio

    def compute_PeakIntensity(self) -> float:
        """
        计算峰值强度
        :return: 峰值强度
        """
        return np.max(self.intensity_distribution)

    def compute_FocalOffset(self) -> int:
        """
        计算焦移
        :return: 焦移
        """
        if self.intensity_distribution is None:
            raise ValueError("请先计算强度分布")
        # 找到峰值强度所在的索引
        peak_index = np.argmax(self.intensity_distribution)
        # 计算焦移
        return peak_index - self.intensity_distribution.shape[0] // 2

    def compute_DOF(self) -> float:
        """
        计算焦深
        :return: 焦深
        """
        # 找到半高全宽
        FWHM = self.compute_FWHM()
        # 计算焦深
        return 0.443 * self.wavelength / (FWHM ** 2)

    def compute_IntensitySum(self) -> float:
        """
        计算能量求和
        :return: 能量求和
        """
        if self.intensity_distribution is None:
            raise ValueError("请先计算强度分布")
        return np.sum(self.intensity_distribution)

    def compute_all(self) -> dict:
        """
        计算所有参数
        :return:所有评价参数
        """
        return {
            "FWHM": self.compute_FWHM(),
            "SideLobeRatio": self.compute_SideLobeRatio(),
            "PeakIntensity": self.compute_PeakIntensity(),
            "FocalOffset": self.compute_FocalOffset(),
            "DOF": self.compute_DOF(),
            "IntensitySum": self.compute_IntensitySum()
        }

    def show_intensity_distribution(self):
        """
        绘制强度分布
        """
        if self.intensity_distribution is None:
            raise ValueError("请先计算强度分布")

        # 可视化强度分布
        plt.imshow(self.intensity_distribution)
        plt.title('Intensity Distribution')
        plt.show()

    def show_diffraction_field(self):
        """
        绘制衍射场
        """
        if self.diffraction_field is None:
            raise ValueError("请先计算衍射场")
        # 可视化衍射场
        plt.contourf(np.abs(self.diffraction_field) ** 2)
        plt.colorbar()  # 添加颜色条
        plt.title('Diffraction Field')
        plt.show()

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


if __name__ == '__main__':
    # 示例用法
    wavelength = 0.6328e-6  # 波长（单位：米）
    z = 1.0  # 衍射距离（单位：米）
    grid_size = 512  # 网格大小
    phase_plane = np.random.rand(grid_size, grid_size)  # 生成随机相位数组作为输入

    # 创建 FresnelDiffraction 实例
    fresnel_diffraction = FresnelDiffraction(wavelength=wavelength, z=z, phase_plane=phase_plane)

    # 假设有一个相位面 phase_plane 和衍射距离 z
    # 计算衍射场
    fresnel_diffraction.compute_diffraction_field()

    # 计算强度分布
    fresnel_diffraction.compute_intensity_distribution()

    print(fresnel_diffraction.compute_all())
