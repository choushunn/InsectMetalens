# -*- coding:utf-8 -*-
# @Project    : InsectMetalens
# @FileName   : FresnelDiffraction.py
# @Author     : Spring
# @Time       : 2024/4/7 17:06
# @Description:
from time import sleep
from typing import Set

import cupy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
# import  numpy as np
from utils import show_contour, calculate_runtime


class FresnelDiffraction:
    def __init__(self, metalens) -> None:
        """
        初始化Fresnel衍射的参数
        """
        # 透镜参数
        self.metalens = metalens

    def fourier_transform_2d(self, input_data: np.ndarray, flag: int = 1) -> np.ndarray:
        """
        二维傅里叶变换
        :param input_data:输入数据
        :param flag:1为正变换，-1为逆变换
        :return:变换后的数据
        """
        n_sampling = self.metalens.n_sampling
        a = np.exp(1j * 2 * np.pi / n_sampling * (
                n_sampling / 2 - 0.5) * self.metalens.sample_points)
        # 计算二维外积
        A = np.outer(a, a)
        # 计算调制因子
        C = np.exp(-1j * 2 * np.pi / n_sampling * (n_sampling / 2 - 0.5) ** 2 * 2) * A
        trans_data = None
        if flag == 1:
            # 执行正变换
            trans_data = self.metalens.Dx ** 2 * C * np.fft.fft2(A * input_data)
        if flag == -1:
            # 执行逆变换
            trans_data = (1. / (
                    n_sampling * self.metalens.Dx)) ** 2 * n_sampling ** 2 * np.conj(
                C) * np.fft.ifft2(
                np.conj(A) * input_data)
        return trans_data

    def compute_all(self, phases) -> dict:
        """
        计算所有参数
        :return:所有评价参数
        """
        Nz = self.metalens.nz  # Nz为传播面数
        dZ = self.metalens.z_range / (2 * Nz) if Nz > 0 else 0  # dZ是每个传播面之间的间距

        Zd = self.metalens.focal_length + dZ * np.arange(-Nz, Nz + 1)  # Zd 数组表示了从焦点向前和向后的传播面深度值。
        # Ex存储所有波长的所有传播面的衍射结果,形状为(n_phases, 2*Nz+1, 2*n_n+2, 2*n_n+2)
        Ex = np.zeros((len(phases), 2 * Nz + 1, 2 * self.metalens.n_n + 2, 2 * self.metalens.n_n + 2), dtype=complex)
        # p_bar = tqdm(total=len(phases))
        # 逐个波长计算
        for i, phase in enumerate(phases):
            for nnz in range(0, 2 * Nz + 1):
                # nnz 是第几个传播面
                # 更新进度条描述
                # p_bar.set_description(f"第{i + 1}/{len(phases)}个波长, 第{nnz + 1}/{2 * Nz + 1}个传播面")
                # step1:先执行衍射计算
                Ex[i, nnz, :, :] = self.Diffra2DAngularSpectrum_BerryPhase(self.metalens.lambda_list[i], phase, Zd[nnz])
            # 更新进度条进度
            # p_bar.update(1)
        # ================5个波长衍射结束后计算的参数===================
        # print("5个波长121个传播面的Ex:", Ex.shape)
        # 所有的光强
        intensities = np.abs(Ex) ** 2  # intensities 形状 (5,121,402,402),和Ex形状一致
        # n_n_intensity 替换 XX_Itotal_Ir_Iphi_IzDisplay
        # todo:n_n 应该是Nz 60,焦平面的光强
        n_n_intensity = intensities[:, :, :, self.metalens.n_n]  # n_n平面的光强,n_n_intensity形状 (5,121,402,1)
        # 显示每个波长的n_n平面的光强
        # for i in range(len(phases)):
        #     plt.imshow(n_n_intensity[i, :, :].get())
        #     plt.colorbar()
        #     plt.title(f"lambda={self.metalens.lambda_list[i]}")
        #     plt.xlabel("Nz=121")
        #     plt.ylabel("n_n=402")
        #     plt.show()

        DOF = np.zeros(len(phases))
        intensity_sum = np.zeros(len(phases))
        focal_offset = np.zeros(len(phases))
        FWHM = np.zeros(len(phases))
        side_lobe_ratio = np.zeros(len(phases))
        intensity_peak = np.zeros(len(phases))

        x_coordinates = (np.arange(self.metalens.n_sampling / 2 - self.metalens.n_n,
                                   self.metalens.n_sampling / 2 + 1 + self.metalens.n_n) - self.metalens.n_sampling / 2 + 0.5) * self.metalens.Dx
        # x_coordinates替换XX,沿着 x 轴取样相关的一系列 x 坐标
        # IPeak 的形状应该为 (5, 2 * Nz + 1)
        # todo:先取每个截面的中心线
        IPeak = np.max(n_n_intensity, axis=2)  # 5个波长121个传播面的IPeak

        for i in range(len(phases)):
            # TODO:计算 DOF 焦深, DOF 形状为 (len(phases))
            DOF[i] = self.fun_calculate_DOF(IPeak[i, :], Zd)
            # TODO:计算 intensity_sum 形状为 (len(phases))
            intensity_sum[i] = np.sum(
                IPeak[i, (2 * Nz + 1) // 3:2 * (2 * Nz + 1) // 3]) * 2 - np.sum(IPeak[i, :])
            if intensity_sum[i] < 0:
                intensity_sum[i] = np.sum(
                    IPeak[i, (2 * Nz + 1) // 3:2 * (2 * Nz + 1) // 3])
            # TODO:计算 FWHM, side_lobe_ratio, intensity_peak 形状为 (len(phases))
            Nz_intensity = n_n_intensity[i, Nz, :]  # 设定焦平面,n_n=200, Nz=60, Nz一共为121个平面
            # print("Intensity_z:", Intensity_z.shape)
            FWHM[i], side_lobe_ratio[i], intensity_peak[i] = self.EfieldParameters(Nz_intensity, x_coordinates, self.metalens.n_n)
            # TODO:计算 focal_offset 形状为 (len(phases))
            In = np.argmax(IPeak[i])  # argmax()函数来获取最大值的索引位置# 找到最大强度对应的位置平面,取最后一个 In 中的元素
            if In not in range(0, 2 * Nz + 1):  # 判断最大强度对应的位置平面是否在范围内
                In = Nz
            focal_offset[i] = np.abs(Zd[In] - self.metalens.focal_length) / self.metalens.focal_length  # 焦距的偏移量
        # 返回评价结果,焦移(Focal Offset)\强度和(Intensity Sum)\峰值半宽(FWHM)\旁瓣比(Side Lobe Ratio)\峰值强度(Intensity Peak)
        return {
            "focal_offset": focal_offset,
            "intensity_sum": intensity_sum,
            "FWHM": FWHM,
            "DOF": DOF,
            "side_lobe_ratio": side_lobe_ratio,
            "intensity_peak": intensity_peak
        }

    def calculate_ex_ey(self, phase, DT=0):
        """
        计算Ex、Ey
        :param phase:
        :param DT:
        :return:
        """
        P_metasurface = self.metalens.sample_interval
        n_sampling = self.metalens.n_sampling
        # 使用采样点数量创建 X 和 Y 矩阵，X 和 Y 分别表示二维平面上的横纵坐标
        X = self.metalens.Dx * np.ones((n_sampling, 1)) * (
                self.metalens.sample_points - n_sampling / 2 + 0.5)
        Y = X.transpose()

        # 计算每个采样点处的等效距离，即所属超表面结构单元中心位置到透镜中心的距离
        Rij = np.sqrt(
            np.ceil((np.abs(X) - 0.5 * P_metasurface) / P_metasurface) ** 2 + np.ceil(
                (np.abs(
                    Y) - 0.5 * P_metasurface) / P_metasurface) ** 2) * P_metasurface
        GeneN_ij = np.floor(Rij / P_metasurface) + 1
        GeneN_ij[GeneN_ij > self.metalens.num_r_outer] = self.metalens.num_r_outer

        # 给出每个采样点所属超表面单元的相位
        Phase_ijUnit1 = phase[GeneN_ij.astype(int) - 1]  # Gene(1:Nring)为相位；
        AmpProfile_ij1 = np.ones((n_sampling, n_sampling)) * 0.66
        AmpProfile_ij1[Rij >= self.metalens.outer_radius] = 0
        Phase_ijUnit1[Rij >= self.metalens.outer_radius] = 0
        Phase_ijUnit1[Rij < self.metalens.inner_radius] = 0

        # ================扩充器件边缘的非结构区域====================
        Phase_ijUnit = np.zeros((n_sampling + DT, n_sampling + DT))
        AmpProfile_ij = np.zeros((n_sampling + DT, n_sampling + DT))
        Phase_ijUnit[DT // 2:n_sampling + DT // 2,
        DT // 2:n_sampling + DT // 2] = Phase_ijUnit1[0:n_sampling,
                                        0:n_sampling]
        AmpProfile_ij[DT // 2:n_sampling + DT // 2,
        DT // 2:n_sampling + DT // 2] = AmpProfile_ij1[0:n_sampling,
                                        0:n_sampling]
        # print(Phase_ijUnit.shape)
        # --计算器件出射场-------------------------------
        Ex0 = AmpProfile_ij * np.exp(1j * Phase_ijUnit)

        Ey0 = np.zeros((n_sampling + DT, n_sampling + DT))
        return Ex0, Ey0

    def Diffra2DAngularSpectrum_BerryPhase(self, wavelength, phase, Z):
        """

        :param wavelength:当前需要计算的波长
        :param phase: 当前需要计算的相位
        :param Z: 当前需要计算的传播面
        :return:
        """
        Ex0, Ey0 = self.calculate_ex_ey(phase)

        Ex = self.Diffraction2DTransPolar(Ex0, Ey0, Z, wavelength)
        return Ex

    def c_fft2d(self, Ex0, Dx, N):
        """

        :param Ex0:
        :param Dx:
        :param N:
        :return:
        """
        # 生成复数向量 a，长度为 N
        a = np.exp(1j * 2 * np.pi / N * (N / 2 - 0.5) * np.arange(N)).astype(np.complex64)

        # 向量的外积
        A = np.outer(a, a)
        # 修正系数 C，与 A 逐元素相乘
        C = A * np.exp(-1j * 2 * np.pi / N * (N / 2 - 0.5) ** 2 * 2)

        C.astype(np.complex64)  # 将 C 转换为复数类型
        # 计算结果，先进行傅里叶变换，再与 C 逐元素相乘，再乘以 Dx 的平方
        result = Dx ** 2 * C * np.fft.fft2(A * Ex0)

        return result

    def Diffraction2DTransPolar(self, Ex0, Ey0, Z, wavelength):
        """

        :param Ex0:
        :param Ey0:
        :param Z:
        :param wavelength:
        :return:
        """
        n_sampling = self.metalens.n_sampling
        # 计算频率
        refractive_index = 1
        freq = 1. / (n_sampling * self.metalens.Dx) * (
                self.metalens.sample_points - n_sampling / 2 + 0.5)
        freq_x = np.outer(freq, np.ones(n_sampling))
        freq_y = freq_x.T

        # 计算频率因子
        fza = ((refractive_index / wavelength) ** 2 - freq_x ** 2 - freq_y ** 2).astype(np.complex128)
        # 计算极化因子
        fz = np.sqrt(fza)

        # 执行正向傅里叶变换
        SpectrumX = self.fourier_transform_2d(Ex0, 1)
        # SpectrumX = self.c_fft2d(Ex0, self.metalens.Dx, self.metalens.n_sampling)
        # 对频谱进行相位调制
        SpectrumX = SpectrumX * np.exp(1j * 2 * np.pi * fz * Z)

        # 执行逆向傅里叶变换
        Ex = self.fourier_transform_2d(SpectrumX, -1)

        # 选择感兴趣的显示区域
        Ex = Ex[int(n_sampling / 2 - self.metalens.n_n):int(
            n_sampling / 2 + 2 + self.metalens.n_n),
             int(n_sampling / 2 - self.metalens.n_n):int(
                 n_sampling / 2 + 2 + self.metalens.n_n)]
        Ey = 0.1 * Ex
        return Ex

    def fun_calculate_DOF(self, IPeak, Zd):
        X_Center = self.metalens.nz
        x = Zd / self.metalens.wavelength_center
        nX = IPeak.size
        IntensX = IPeak
        Imax = IntensX[X_Center]
        Xfwhm1 = 0
        Xfwhm2 = 0
        flag = 0
        for i in range(X_Center, 0, -1):
            if IntensX[i] < 0.5 * Imax:
                if flag == 0:
                    x1 = x[i + 1]
                    I1 = IntensX[i + 1]
                    x2 = x[i]
                    I2 = IntensX[i]
                    b = (I2 - I1) / (x2 - x1)
                    c = I2 - b * x2
                    Xfwhm1 = (0.5 * Imax - c) / b
                    flag = 1

        flag = 0
        for i in range(X_Center, nX - 1, 1):
            if IntensX[i] < 0.5 * Imax:
                if flag == 0:
                    x1 = x[i - 1]
                    I1 = IntensX[i - 1]
                    x2 = x[i]
                    I2 = IntensX[i]
                    b = (I2 - I1) / (x2 - x1)
                    c = I2 - b * x2
                    Xfwhm2 = (0.5 * Imax - c) / b
                    flag = 1

        FWHM = Xfwhm2 - Xfwhm1
        return FWHM

    def EfieldParameters(self, Intensity_z, XX, nn):
        X = XX
        IntensX = Intensity_z
        X_Center = nn
        nX = IntensX.size
        Imax = IntensX[X_Center]
        Xfwhm1 = 0
        Xfwhm2 = 0
        flag = 0

        for i in range(X_Center, 0, -1):
            if IntensX[i] < 0.5 * Imax:
                if flag == 0:
                    x1 = X[i + 1]
                    I1 = IntensX[i + 1]
                    x2 = X[i]
                    I2 = IntensX[i]
                    b = (I2 - I1) / (x2 - x1)
                    c = I2 - b * x2
                    Xfwhm1 = (0.5 * Imax - c) / b
                    flag = 1

        flag = 0
        for i in range(X_Center, nX - 1, 1):
            if IntensX[i] < 0.5 * Imax:
                if flag == 0:
                    x1 = X[i - 1]
                    I1 = IntensX[i - 1]
                    x2 = X[i]
                    I2 = IntensX[i]
                    b = (I2 - I1) / (x2 - x1)
                    c = I2 - b * x2
                    Xfwhm2 = (0.5 * Imax - c) / b
                    flag = 1

        FWHM = Xfwhm2 - Xfwhm1
        flag = 0
        SideLobe = 0

        for j in range(X_Center, nX - 1):
            if (IntensX[j] <= IntensX[j - 1] and IntensX[j] <= IntensX[j + 1] and j < nX - 1):
                if (flag == 0):
                    sideIn = np.max(IntensX[j:nX])
                    SideLobe = sideIn / Imax
                    flag = 1

        return FWHM, SideLobe, Imax

    def calculate_FWHM(self, IntensX, X, X_Center):
        """
        计算峰值半宽(FWHM)

        参数:
            IntensX: 电场强度的一维数组
            X: 与电场对应的位置的一维数组
            X_Center: 电场峰值所在位置的索引

        返回值:
            FWHM: 峰值半宽

        注意:
            如果找不到满足条件的边界位置，则返回None
        """
        nX = IntensX.size
        Imax = IntensX[X_Center]
        x1_fwhm = None
        x2_fwhm = None

        # 从峰值位置向左查找边界位置x1_fwhm
        for i in range(X_Center, 0, -1):
            if IntensX[i] < 0.5 * Imax:
                x1_fwhm = X[i]
                break

        # 从峰值位置向右查找边界位置x2_fwhm
        for i in range(X_Center, nX):
            if IntensX[i] < 0.5 * Imax:
                x2_fwhm = X[i]
                break

        # 如果两个边界位置都找到了，则计算峰值半宽(FWHM)
        if x1_fwhm is not None and x2_fwhm is not None:
            FWHM = x2_fwhm - x1_fwhm
            return FWHM
        else:
            return None


if __name__ == '__main__':
    # 菲涅尔衍射模拟
    pass
