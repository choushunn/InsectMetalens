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
        Nz = self.metalens.nz
        if Nz > 0:
            dZ = self.metalens.z_range / (2 * Nz)
        else:
            dZ = 0
        Zd = self.metalens.focal_length + dZ * np.arange(-Nz, Nz + 1)

        # 402 * 121, 121个传播面上
        XX_Itotal_Ir_Iphi_IzDisplay = np.zeros((2 * self.metalens.n_n + 2, 2 * Nz + 1))
        IPeak = np.zeros(2 * Nz + 1)
        DOF = np.zeros(len(phases))
        Intensity_sum = np.zeros(len(phases))
        # 5个波长一起计算
        for i, phase in enumerate(phases):
            p_bar = tqdm(total=2 * Nz + 1)
            for nnz in range(0, 2 * Nz + 1):
                p_bar.update(1)
                p_bar.set_description(f"第{i + 1}/{len(phases)}个波长, 第{nnz + 1}/{2 * Nz + 1}个传播面")
                # 执行衍射计算
                Ex = self.Diffra2DAngularSpectrum_BerryPhase(self.metalens.lambda_list[i], phase, Zd[nnz])
                Intensity = np.abs(Ex) ** 2
                XX_Itotal_Ir_Iphi_IzDisplay[:, nnz] = Intensity[:, self.metalens.n_n]
                IPeak[nnz] = np.max(np.max(XX_Itotal_Ir_Iphi_IzDisplay[:, nnz]))
            print(XX_Itotal_Ir_Iphi_IzDisplay.shape)
            print(IPeak.shape)

            # 1个波长的传播面计算结束
            # ====================下面计算一个波长的评价参数====================
            DOF[i] = self.fun_calculate_DOF(IPeak, Zd)
            print(DOF)
            Intensity_sum[i] = np.sum(IPeak[(2 * Nz + 1) // 3:2 * (2 * Nz + 1) // 3]) * 2 - np.sum(
                IPeak)  # 取中间 DOF 区域的强度和
            if Intensity_sum[i] < 0:
                Intensity_sum[i] = np.sum(
                    IPeak[(2 * Nz + 1) // 3:2 * (2 * Nz + 1) // 3])
            IPeakmax = np.max(IPeak)
            In = np.where(IPeak == IPeakmax)[0]  # 找到最大强度对应的位置平面,取最后一个 In 中的元素
            Intensity_z = XX_Itotal_Ir_Iphi_IzDisplay[:, Nz]  # 设定焦平面
            # print(Intensity_z)
            n_sampling = self.metalens.n_sampling
            XX = ((np.arange(n_sampling / 2 - self.metalens.n_n,
                             n_sampling / 2 + 1 + self.metalens.n_n) - n_sampling / 2 + 0.5) * self.metalens.Dx)
            FWHM_x, SideLobeRatio_x, IntensPeak_x = self.EfieldParameters(Intensity_z, XX, self.metalens.n_n)
            IntensPeak_x = np.average(XX_Itotal_Ir_Iphi_IzDisplay[self.metalens.n_n, Nz - 2: Nz + 2])

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
        Phase_ijUnit1[Rij < self.metalens.outer_radius] = 0

        ####扩充器件边缘的非结构区域#######################
        Phase_ijUnit = np.zeros((n_sampling + DT, n_sampling + DT))
        AmpProfile_ij = np.zeros((n_sampling + DT, n_sampling + DT))
        Phase_ijUnit[DT // 2:n_sampling + DT // 2,
        DT // 2:n_sampling + DT // 2] = Phase_ijUnit1[0:n_sampling,
                                        0:n_sampling]
        AmpProfile_ij[DT // 2:n_sampling + DT // 2,
        DT // 2:n_sampling + DT // 2] = AmpProfile_ij1[0:n_sampling,
                                        0:n_sampling]
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
