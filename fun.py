import cmath

# import cupyx
from matplotlib import pyplot as plt

# import matplotlib.pyplot as plt

try:
    import cupy as np
    # import numpy as np

except ModuleNotFoundError as e:
    import numpy as np

import time


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print("Function {} took {:.6f} seconds to run.".format(func.__name__, end_time - start_time))
        return result

    return wrapper


# @timer
def Fun_GeneratingPhase(GDR, Gene_phase, Nr_gene, Nr_outter, lam, r, FocalLength, c):
    """
    生成相位
    :param GDR:
    :param Gene_PersonalPresent:
    :param Nr_gene:
    :param Nr_outter:
    :param lam:波长
    :param r:
    :param FocalLength:焦距
    :param c:光速
    :return:phase,三个波长的相位
    """
    lamc = 10.6
    # phase = np.zeros((3, 159))
    GDRmax = np.max(GDR)
    p = np.where(GDR == GDRmax)[-1]
    # print(p, type(p))
    phasec = np.zeros((Nr_outter))
    rr = np.zeros(((Nr_gene)))
    L = np.zeros(((Nr_gene)))
    # Gene_phase = np.zeros((Nr_gene))
    for j in range(Nr_gene):
        # print(Nr_gene)
        if j < Nr_gene - 1:
            # print(p[j])
            rr[j] = r[p[j]]
            L[j] = np.sqrt(rr[j] ** 2 + FocalLength ** 2)
            # print(p[j],p[j+1],type(p[j+1]),r[p[j]:p[j + 1]])

            phasec[int(p[j]):int(p[j + 1])] = 2 * np.pi / lamc * (
                        np.sqrt(r[int(p[j + 1] - 1)] ** 2 + FocalLength ** 2) - np.sqrt(
                    r[int(p[j]):int(p[j + 1])] ** 2 + FocalLength ** 2))
            if j == 0:
                rr[j] = 0
                L[j] = np.sqrt(rr[j] ** 2 + FocalLength ** 2)
            phasec[p[j]:p[j + 1]] = phasec[p[j]:p[j + 1]] - 2 * np.pi / lamc * (L[j] - L[0]) + Gene_phase[j]
            # phasec[p[j]:p[j + 1]] = phasec[p[j]:p[j + 1]] + Gene_phase[j]

        else:
            rr[j] = r[p[j]]
            L[j] = np.sqrt(rr[j] ** 2 + FocalLength ** 2)
            phasec[p[j]:Nr_outter] = 2 * np.pi / lamc * (np.sqrt(r[Nr_outter - 1] ** 2 + FocalLength ** 2) - np.sqrt(
                r[p[j]:Nr_outter] ** 2 + FocalLength ** 2))
            phasec[p[j]:Nr_outter] = phasec[p[j]:Nr_outter] - 2 * np.pi / lamc * (L[j] - L[0]) + Gene_phase[j]
            # phasec[p[j]:Nr_outter] = phasec[p[j]:Nr_outter] + Gene_phase[j]
    # print( phasec)
    # plt.plot(phasec.get())
    # plt.show()
    # exit()
    phasec = np.transpose(phasec)
    phase0 = phasec + GDR * (2 * np.pi * c * (1 / lam[0] - 1 / lamc))
    phase1 = phasec + GDR * (2 * np.pi * c * (1 / lam[1] - 1 / lamc))
    phase2 = phasec + GDR * (2 * np.pi * c * (1 / lam[2] - 1 / lamc))
    phase3 = phasec + GDR * (2 * np.pi * c * (1 / lam[3] - 1 / lamc))
    phase4 = phasec + GDR * (2 * np.pi * c * (1 / lam[4] - 1 / lamc))
    phase5 = phasec + GDR * (2 * np.pi * c * (1 / lam[5] - 1 / lamc))
    # phase6 = phasec + GDR * (2 * np.pi * c * (1 / lam[6] - 1 / lamc))
    # phase7 = phasec + GDR * (2 * np.pi * c * (1 / lam[7] - 1 / lamc))
    # phase8 = phasec + GDR * (2 * np.pi * c * (1 / lam[8] - 1 / lamc))
    # phase[0, :] = phase0
    # phase[1, :] = phase1
    # phase[2, :] = phase2
    return [phase0, phase1, phase2, phase3, phase4, phase5]


# @timer
def FourrierTrans2D(g, Dx, N, flag):
    """

    :param g:
    :param Dx:
    :param N:
    :param flag:
    :return:
    """
    # import cupy as cp
    # g = cp.asarray(g)
    # Dx = cp.asarray(Dx)
    # N = cp.asarray(N)
    # flag = cp.asarray(flag)

    num = np.arange(N)
    a = np.exp(1j * 2 * np.pi / N * (N / 2 - 0.5) * num).astype(np.complex64)

    A = a.reshape(-1, 1) * a
    C = np.exp(-1j * 2 * np.pi / N * (N / 2 - 0.5) ** 2 * 2) * A
    C.astype(np.complex64)
    G = None
    if flag == 1:
        G = Dx ** 2 * C * np.fft.fft2(A * g)
        # G = np.fft.fft2(A * g)  # 二维FFT变换后会发生错位
        # G = np.roll(G, -1, axis=0)  # 先进行错位纠正
        # G = Dx**2 * C * G  # 再进行二维FFT计算结果的系数修正
    if flag == -1:
        G = (1. / (N * Dx)) ** 2 * N ** 2 * np.conj(C) * np.fft.ifft2(np.conj(A) * g)
    return G.astype(np.complex64)
    # 二维FFT正变换进行错位修正后，反变换时不需再进行错位修正
    # G = np.roll(G, -1, axis=0)  # 修补，没解决根本问题(为什么傅里叶变换后会错位)07-01-2018尚待找到原因


# @timer
# def FourrierBessel(g, Dx, N, flag):
#
#     # import numpy as cp
#     # g = cp.asarray(g)
#     # Dx = cp.asarray(Dx)
#     # N = cp.asarray(N)
#     # flag = cp.asarray(flag)
#
#     num = np.arange(N)
#     a = np.exp(1j * 2 * np.pi / N * (N / 2 - 0.5) * num)
#     # A = a.reshape(-1, 1) * a
#     C = np.exp(-1j * 2 * np.pi / N * (N / 2 - 0.5) ** 2) * a
#     G = None
#     if flag == 1:
#         G = Dx * C * np.fft.fft(a * g)
#         # G = np.fft.fft2(A * g)  # 二维FFT变换后会发生错位
#         # G = np.roll(G, -1, axis=0)  # 先进行错位纠正
#         # G = Dx**2 * C * G  # 再进行二维FFT计算结果的系数修正
#     if flag == -1:
#         G = (1. / (N * Dx)) * N * np.conj(C) * np.fft.ifft(np.conj(a) * g)
#     return G
#     # 二维FFT正变换进行错位修正后，反变换时不需再进行错位修正
#     # G = np.roll(G, -1, axis=0)  # 修补，没解决根本问题(为什么傅里叶变换后会错位)07-01-2018尚待找到原因


# @timer
def Diffraction2DTransPolar(Ex0, Ey0, Z, Wavelen0, n_refr, Dx, N, nn):

    num = np.arange(N)
    freq = 1. / (N * Dx) * (num - N / 2 + 0.5)
    freq_x = np.outer(freq, np.ones(N))
    freq_y = freq_x.T
    fza = ((n_refr / Wavelen0) ** 2 - freq_x ** 2 - freq_y ** 2).astype(np.complex128)
    fz = np.sqrt(fza).astype(np.complex64)

    SpectrumX = FourrierTrans2D(Ex0, Dx, N, 1)

    SpectrumX = SpectrumX * np.exp(1j * 2 * np.pi * fz * Z)
    SpectrumX.astype(np.complex64)

    Ex = FourrierTrans2D(SpectrumX, Dx, N, -1)

    del SpectrumX
    nn = int(nn)
    Ex = Ex[int(N / 2 - nn):int(N / 2 + 2 + nn), int(N / 2 - nn):int(N / 2 + 2 + nn)]


    return Ex


# @timer
def Fun_Diffra2DAngularSpectrum_BerryPhase(Wavelen0, Zd, R_outter, R_inner, Dx, N, n_refra, P_metasurface, Gene_Lens0,
                                           Nring, nn):

    num = np.arange(N)
    N_sampling = N
    DT = 0  # 偶数
    # 产生X，Y矩阵
    Y = Dx * np.ones((N, 1)) * (num - N / 2 + 0.5)
    X = Y.transpose()

    # 计算每个采样点处的等效距离，即所属超表面结构单元中心位置到透镜中心的距离
    Rij = np.sqrt(np.ceil((np.abs(X) - 0.5 * P_metasurface) / P_metasurface) ** 2 + np.ceil(
        (np.abs(Y) - 0.5 * P_metasurface) / P_metasurface) ** 2) * P_metasurface


    # 确定每个采样点（超表面单元）对应的基因，在Gene_Lens中对应的基因片段（即对应基因片段的位置（第几个“环带”））；
    GeneN_ij = np.floor(Rij / P_metasurface) + 1
    GeneN_ij[GeneN_ij > Nring] = Nring

    # 给出每个采样点所属超表面单元的相位
    Phase_ijUnit1 = Gene_Lens0[GeneN_ij.astype(int) - 1]  # Gene(1:Nring)为相位；
    AmpProfile_ij1 = np.ones((N, N)) * 0.66
    AmpProfile_ij1[Rij >= R_outter] = 0
    Phase_ijUnit1[Rij >= R_outter] = 0
    Phase_ijUnit1[Rij < R_inner] = 0
    del Rij
    ####扩充器件边缘的非结构区域#######################
    Phase_ijUnit = np.zeros((N_sampling + DT, N_sampling + DT))
    AmpProfile_ij = np.zeros((N_sampling + DT, N_sampling + DT))
    Phase_ijUnit[DT // 2:N_sampling + DT // 2, DT // 2:N_sampling + DT // 2] = Phase_ijUnit1[0:N_sampling, 0:N_sampling]
    AmpProfile_ij[DT // 2:N_sampling + DT // 2, DT // 2:N_sampling + DT // 2] = AmpProfile_ij1[0:N_sampling,
                                                                                0:N_sampling]
    del Phase_ijUnit1
    del AmpProfile_ij1

    # --计算器件出射场-------------------------------
    Ex0 = AmpProfile_ij * np.exp(1j * Phase_ijUnit)
    print("ex0", Ex0.shape)
    Ey0 = np.zeros((N_sampling + DT, N_sampling + DT))
    del Phase_ijUnit
    del AmpProfile_ij

    Ex = Diffraction2DTransPolar(Ex0, Ey0, Zd, Wavelen0, n_refra, Dx, N + DT, nn)
    del Ex0
    del Ey0
    return Ex, 0, 0


# @timer
def CPSWFs_FWHM_calculation(IntensX, x, X_Center):
    """

    :param IntensX:
    :param x:
    :param X_Center:
    :return:
    """
    # 数据类型为行向量
    nX = IntensX.size
    Imax = IntensX[X_Center]
    Xfwhm1 = 0
    Xfwhm2 = 0
    flag = 0
    # print(X_Center)
    # print(X_Center[::-1])
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
    for i in range(X_Center, nX-1, 1):
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

    # @timer
    # def Fun_EfieldParameters(Intensity, X, Center, SpotType):
    """

    :param Intensity:
    :param X:
    :param Center:
    :param SpotType:
    :return:
    """
    # print(Intensity.shape)
    # Intensity = Intensity.T[0]
    # print(Intensity.shape)
    # N = Intensity.size
    # FWHM = 0
    # SideLobeRatio = 0
    # IntensPeak = 0
    # if SpotType == 0:  # solid focal spot
    #     # calculating Peak Intensity
    #     IntensPeak = Intensity[Center]
    #
    #     # calculating FWHM
    #     k1 = Center
    #     while Intensity[k1] > IntensPeak / 2 and k1 > 2:
    #         k1 = k1 - 1
    #     k2 = Center + 1
    #     while Intensity[k2] > IntensPeak / 2 and k2 < N - 1:
    #         k2 = k2 + 1
    #     x1 = X[k1]
    #     y1 = Intensity[k1]
    #     x2 = X[k1 + 1]
    #     y2 = Intensity[k1 + 1]
    #     b = (y2 - y1) / (x2 - x1)
    #     c = y2 - b * x2
    #     xL = (IntensPeak / 2 - c) / b
    #
    #     x1 = X[k2]
    #     y1 = Intensity[k2]
    #     x2 = X[k2 - 1]
    #     y2 = Intensity[k2 - 1]
    #     b = (y2 - y1) / (x2 - x1)
    #     c = y2 - b * x2
    #     xR = (IntensPeak / 2 - c) / b
    #
    #     FWHM = np.abs(xR - xL)
    #
    #     # calculating Sidelobe
    #     k1 = Center
    #     while not ((Intensity[k1 - 1] >= Intensity[k1] and Intensity[k1] < Intensity[k1 + 1]) or k1 > 2):
    #         k1 = k1 - 1
    #     if k1 == Center:
    #         k1 = 1
    #     k2 = Center + 1
    #     while not (
    #             (Intensity[k2 - 1] > Intensity[k2] and Intensity[k2] <= Intensity[k2 + 1]) or k2 < N - 1):
    #         k2 = k2 + 1
    #     if k2 == Center + 1:
    #         k2 = N
    #     if k1 <= 0:
    #         k1 = 1
    #     if k2 > Intensity.size:
    #         k2 = Intensity.size - 1
    #     # max_index = np.argmax(Intensity)
    #     # if max_index < Nr_outter:
    #     #     max_index = Nr_outter
    #     # SideLobeRatio = np.max(Intensity[Nr_outter:]) / np.max(Intensity)
    #     # SideLobeRatio =
    #     # print(k1,k2,N)
    #     SideLobeRatio = max(np.max(Intensity[0:k1]), np.max(Intensity[k2 - 1:N])) / IntensPeak
    # elif SpotType == 1:  # hollow focal spot
    #     # calculation peak
    #     k1 = Center
    #     while not (Intensity[k1 - 1] <= Intensity[k1] and Intensity[k1] > Intensity[k1 + 1] and k1 > 2):
    #         k1 = k1 - 1
    #     k2 = Center + 1
    #     while not (Intensity[k2 - 1] < Intensity[k2] and Intensity[k2] >= Intensity[k2 + 1] and k2 < N - 1):
    #         k2 = k2 + 1
    #
    #     IntensPeak = max(Intensity[k1], Intensity[k2])
    #
    #     # calculation FWHM
    #     kc1 = k1
    #     kc2 = k2
    #     while Intensity[k1] > IntensPeak / 2 and k1 < N - 1:
    #         k1 = k1 + 1
    #     while Intensity[k2] > IntensPeak / 2 and k2 > 2:
    #         k2 = k2 - 1
    #
    #     x1 = X[k1]
    #     y1 = Intensity[k1]
    #     x2 = X[k1 - 1]
    #     y2 = Intensity[k1 - 1]
    #     b = (y2 - y1) / (x2 - x1)
    #     c = y2 - b * x2
    #     xL = (IntensPeak / 2 - c) / b
    #
    #     x1 = X[k2]
    #     y1 = Intensity[k2]
    #     x2 = X[k2 + 1]
    #     y2 = Intensity[k2 + 1]
    #     b = (y2 - y1) / (x2 - x1)
    #     c = y2 - b * x2
    #     xR = (IntensPeak / 2 - c) / b
    #
    #     FWHM = abs(xR - xL)
    #
    #     # calculating Sidelobe ratio
    #     k1 = kc1
    #     k2 = kc2
    #     while not (Intensity[k1 - 1] >= Intensity[k1] and Intensity[k1] < Intensity[k1 + 1] and k1 > 2):
    #         k1 = k1 - 1
    #     while not (Intensity[k2 - 1] > Intensity[k2] and Intensity[k2] <= Intensity[k2 + 1] and k2 < N - 1):
    #         k2 = k2 + 1
    #
    #     SideLobeRatio = max(max(Intensity[0:k1]), max(Intensity[k2 - 1:N])) / IntensPeak
    #
    # return FWHM, SideLobeRatio, IntensPeak


def Fun_EfieldParameters(IntensX, X, X_Center, SpotType):
    nX = IntensX.size
    Imax = IntensX[X_Center]
    Xfwhm1 = 0
    Xfwhm2 = 0
    flag = 0
    # print(X_Center)
    # print(X_Center[::-1])
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
    for i in range(X_Center, nX-1, 1):
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
    # for j=X_Center:+1: nX - 1
    for j in range(X_Center, nX - 1):
        # if (IntensX(j) <= IntensX(j - 1) & & IntensX(j) <= IntensX(j + 1) & & j < nX - 1) % 找到第一个波谷
        if (IntensX[j] <= IntensX[j - 1] and IntensX[j] <= IntensX[j + 1] and j < nX - 1):
            if (flag == 0):
                sideIn = np.max(IntensX[j:nX])
                SideLobe = sideIn / Imax
                flag = 1
            # if (flag == 0)
            #     sideIn = max(IntensX(j:nX));
            #     SideLobe = sideIn / Imax;
            #     flag = 1;

    return FWHM, SideLobe, Imax


# @timer
def Fun_FitnessLenserror(TargetFWHM, TargetSidlobe, TargetPeakIntensity, TargetFocal_offset, FWHM, SideLobe, IntensPeak,
                         Focal_offset, DOF, Intensity_sum):
    """

    :param TargetFWHM:
    :param TargetSidlobe:
    :param TargetPeakIntensity:
    :param TargetFocal_offset:
    :param FWHM:
    :param SideLobe:
    :param IntensPeak:
    :param Focal_offset:
    :param DOF:
    :param Intensity_sum:
    :return:
    """
    d = [10, 2, 10, 0.1, 0.01, 1]
    fitness1 = abs(TargetPeakIntensity - IntensPeak) / TargetPeakIntensity*1  # 强度偏差500
    fitness2 = abs(Focal_offset - TargetFocal_offset) *100  # 焦斑偏移量0.001
    fitness3 = abs(FWHM - TargetFWHM) / TargetFWHM  # FWHM 0.1
    fitness4 = abs(SideLobe - TargetSidlobe) / TargetSidlobe
    # fitness5 = abs(DOF[0] - DOF[1]) + abs(DOF[2] - DOF[1])  # 计算其他波长较深与中心波长较深的偏离
    # fitness5 = abs(DOF[0] - DOF[2]) + abs(DOF[1] - DOF[2]) + abs(DOF[3] - DOF[2]) + abs(DOF[4] - DOF[2])# 计算其他波长较深与中心波长较深的偏离
    # fitness6 = (abs(Intensity_sum[0] - Intensity_sum[1]) + abs(Intensity_sum[2] - Intensity_sum[1])) / 10
    fitness5 = np.std(DOF)*1
    # fitness6 = np.std(Intensity_sum) / np.max(Intensity_sum)
    fitness6 = 10**6/np.min(Intensity_sum) #max min()
    # fitness = fitness1 + fitness2 ** 2 + fitness3 + fitness4 + fitness5 ** 2 + fitness6
    fitness = (fitness1*d[0])**2+(fitness2*d[1])**2+(fitness3*d[2])**2+(fitness4*d[3])**2+(fitness5*d[4])**2+(fitness6*d[5])**2
    return fitness


# @timer
def Fun_GeneRandomGenerationSinglet(N_gene, Nr_gene):
    """

    :param N_gene:
    :param Nr_gene:
    :return:
    """
    Gene_PersonalPresent = N_gene * (np.random.rand(Nr_gene) - 0.5) * 2  # 产生-2pi~2pi的相位跳变；
    Fitness_PersonalPresent = np.inf
    Fitness_PersonalBest = np.inf
    Gene_PersonalBest = Gene_PersonalPresent
    Velocity_PersonalPresent = np.sign(np.random.rand(Nr_gene) - 0.5) * np.random.rand(Nr_gene)
    # print(type(Gene_PersonalPresent), type(Fitness_PersonalPresent), type(Fitness_PersonalBest),
    #       type(Gene_PersonalBest), type(Velocity_PersonalPresent))
    # print(Gene_PersonalPresent.shape, Fitness_PersonalPresent, Fitness_PersonalBest, Gene_PersonalBest.shape,
    #       Velocity_PersonalPresent.shape)
    return Gene_PersonalPresent, Fitness_PersonalPresent, Fitness_PersonalBest, Gene_PersonalBest, Velocity_PersonalPresent


# @timer
def Fun_GeneRandomGeneration(N_gene, Nr_gene, N_particle, Max_iteration):
    """

    :param N_gene:
    :param Nr_gene:
    :param N_particle:
    :param Max_iteration:
    :return:
    """
    # Gene_PersonalPresent = N_gene * (np.random.rand(N_particle, Nr_gene) - 0.5) * 2  # 产生-2pi~2pi的相位跳变；
    Gene_PersonalPresent = N_gene * (np.random.rand(N_particle, Nr_gene) - 0.5) * 1  # 产生-pi~pi的相位跳变；
    Fitness_PersonalPresent = np.zeros((N_particle), dtype="float32")
    # Fitness_PersonalBest = np.ones((N_particle), dtype="float32")
    Fitness_PersonalBest = np.full((N_particle,), np.inf, dtype="float32")
    # Fitness_PersonalBest = np.inf
    # print(Fitness_PersonalBest)
    Gene_PersonalBest = Gene_PersonalPresent
    # Gene_PersonalBest = Gene_PersonalPresent[N_particle - 5, :]
    Velocity_PersonalPresent = np.sign(np.random.rand(N_particle, Nr_gene) - 0.5) * np.random.rand(N_particle, Nr_gene)
    # print(Velocity_PersonalPresent.shape)
    return Gene_PersonalPresent, Fitness_PersonalPresent, Fitness_PersonalBest, Gene_PersonalBest.astype(
        "float32"), Velocity_PersonalPresent


def Fun_UpdatePersonalBest_GeneAndPara(Fitness_PersonalBest, Gene_PersonalBest, FWHM_PersonalBest,
                                       SideLobeRatio_PersonalBest,
                                       IntensPeak_PersonalBest, Focal_offset_PersonalBest, Fitness_PersonalPresent,
                                       Gene_PersonalPresent, FWHM_PersonalPresent, SideLobeRatio_PersonalPresent,
                                       IntensPeak_PersonalPresent, Focal_offset_PersonalPresent, N_particle):
    # print(N_particle,type(N_particle))
    for k in range(N_particle):
        # print(Fitness_PersonalPresent[k],Fitness_PersonalBest[k])
        if Fitness_PersonalPresent[k] < Fitness_PersonalBest[k]:
            Fitness_PersonalBest[k] = Fitness_PersonalPresent[k]
            Gene_PersonalBest[k, :] = Gene_PersonalPresent[k, :]
            FWHM_PersonalBest[k] = FWHM_PersonalPresent[k]
            SideLobeRatio_PersonalBest[k] = SideLobeRatio_PersonalPresent[k]
            IntensPeak_PersonalBest[k] = IntensPeak_PersonalPresent[k]
            Focal_offset_PersonalBest[k] = Focal_offset_PersonalPresent[k]
    # Fitness_LensPersonalBest0 = Fitness_LensPersonalBest
    # Gene_LensPersonalBest0 = Gene_LensPersonalBest
    # FWHM_PersonalBest0 = FWHM_PersonalBest
    # SideLobeRatio_PersonalBest0 = SideLobeRatio_PersonalBest
    # IntensPeak_PersonalBest0 = IntensPeak_PersonalBest
    # Focal_offset_PersonalBest0 = Focal_offset_PersonalBest
    return Fitness_PersonalBest, Gene_PersonalBest, FWHM_PersonalBest, SideLobeRatio_PersonalBest, IntensPeak_PersonalBest, Focal_offset_PersonalBest


# @timer
def Fun_UpdateParticleSingletBerryPhase(Gene_Lens, Gene_LensPersonalBest, Gene_LensGlobalBest, Gene_LensPersonalBestL,
                                        Velocity, N_gene, N_particle, Nr_gene1, n_iteration, Max_iteration):
    """

    :param Gene_Lens:
    :param Gene_LensPersonalBest:
    :param Gene_LensGlobalBest:
    :param Gene_LensPersonalBestL:
    :param Velocity:
    :param N_gene:
    :param N_particle:
    :param Nr_gene1:
    :return:
    """
    # Particle Swarm Parameters
    c1_ini=2.5
    c1_fin=0.5
    c2_ini=1
    c2_fin=2.25
    c1 = c1_ini+(c1_fin-c1_ini)*n_iteration/Max_iteration
    c2 = c2_ini+(c2_fin-c2_ini)*n_iteration/Max_iteration  # 为计算粒子速度的系数 2，2
    w_ini = 0.9
    w_end = 0.4
    w = w_ini-(w_ini-w_end)*n_iteration/Max_iteration
    # w = 0.5  # 计算粒子速度是的权重 0.5
    Vmax = 1
    Vmin = -1
    dt = 0.5
    q = 0.3  # 局部社会因子系数

    for k in range(N_particle):
        # Update velocity
        Velocity[k, :] = w * Velocity[k, :] + c1 * np.random.rand(Nr_gene1) * (
                Gene_LensPersonalBest[k, :] - Gene_Lens[k, :]) + c2 * (
                                 q * np.random.rand(Nr_gene1) * (Gene_LensGlobalBest - Gene_Lens[k, :]) + (
                                 1 - q) * np.random.rand(Nr_gene1) * (
                                         Gene_LensPersonalBestL - Gene_Lens[k, :]))

        # Update genes
        Gene_Lens[k, :] = Gene_Lens[k, :] + Velocity[k, :] * dt

        # If the gene values reach their boundary, then the values are set as the boundary value, and the sign of velocity is inverted.

        # Lower boundary
        Velocity[k, np.where(Gene_Lens[k, :] <= -N_gene)] = np.abs(
            Velocity[k, np.where(Gene_Lens[k, :] <= -N_gene)])
        Gene_Lens[k, np.where(Gene_Lens[k, :] <= -N_gene)] = -N_gene
        Velocity[k, np.where(Velocity[k, :] < Vmin)] = Vmin

        # Upper boundary velocity
        Velocity[k, np.where(Velocity[k, :] > Vmax)] = Vmax

        # Upper boundary gene of Lens #1 and Lens#2
        Gene_Lens[k, np.where(Gene_Lens[k, :] >= N_gene)] = N_gene
        Velocity[k, np.where(Gene_Lens[k, :] >= N_gene)] = -np.abs(
            Velocity[k, np.where(Gene_Lens[k, :] >= N_gene)])

    return Gene_Lens, Velocity


# @timer
def cal_fft():
    import cupy as cp
    N = int(32768 / 2)
    g = cp.random.randn(N, N) + cp.random.randn(N, N) * 1j
    num = cp.arange(N)
    a = cp.exp(1j * 2 * cp.pi / N * (N / 2 - 0.5) * num)
    A = a.reshape(-1, 1) * a
    # cupyx.scipy.fft.hfft2(A*g)
    cupyx.scipy.fft.hfft2(A * g)
    # cp.fft.fft2(A * g)
    # Dx = 2 *  1200 * 10.6 / N

# if __name__ == '__main__':
# cal_fft()
# import numpy as np
# Dx = 2 * 1200 * 10.6 / (32768 / 2)
# print(Dx)
# NA = np.sin(np.arctan(1200*10.6 / 1000 * 10.6))
# print(NA)
def score(phase, lam,n_lam, TargetFieldPolar,XX, FocalLength,
                                       Zd,R_outter,TargetFWHM, TargetSidlobe, TargetPeakIntensity,
                                                              TargetFocal_offset, R_inner,
                                       N_sampling, n_refra1, P_metasurface, Dx,Nr_outter,nn,Nz):
    """

    :param Gene_Lens:
    :param Gene_LensPersonalBest:
    :param Gene_LensGlobalBest:
    :param Gene_LensPersonalBestL:
    :param Velocity:
    :param N_gene:
    :param N_particle:
    :param Nr_gene1:
    :return:
    """
    # 重新计算fitness

    SpotType= 0
    lamc = 10.6
    DOF = np.zeros(n_lam)
    Intensity_sum = np.zeros(n_lam)
    Intensity = np.zeros((nn * 2 + 2, nn * 2 + 2))
    XX_Itotal_Ir_Iphi_IzDisplay = np.zeros((2 * nn + 2, 2 * Nz + 1))  # 存放不同传播面上总场数据
    # 放当前的参数：半高全宽、旁瓣比、强度(多波长)
    FWHM_PersonalPresent0 = np.zeros((n_lam), dtype="float32")

    SideLobeRatio_PersonalPresent0 = np.zeros((n_lam), dtype="float32")
    IntensPeak_PersonalPresent0 = np.zeros((n_lam), dtype="float32")
    Focal_offset_PersonalPresent0 = np.zeros((n_lam), dtype="float32")
    # 循环三个通道
    for i in range(n_lam):
        wavelength = lam[i]
        # 当前计算的波长对应相位
        phasei = phase[i]

        IPeak = np.zeros((2 * Nz + 1))
        # 计算不同传播面
        for nnz in range(2 * Nz + 1):
            Ex, _, _ = Fun_Diffra2DAngularSpectrum_BerryPhase(wavelength, Zd[nnz], R_outter,
                                                              R_inner, Dx,
                                                              N_sampling, n_refra1, P_metasurface,
                                                              phasei,
                                                              Nr_outter, nn)

            if TargetFieldPolar == 0:  # Transvers components
                Intensity = (np.abs(Ex) ** 2) * 1 #intensive polar
            elif TargetFieldPolar == 1:  # Longitudinal components
                # Intensity = np.abs(Ez) ** 2
                pass
            elif TargetFieldPolar == 2:  # All components
                Intensity = (np.abs(Ex) ** 2) * 2
            XX_Itotal_Ir_Iphi_IzDisplay[:, nnz] = Intensity[:, nn]  # change index nn,nn 07-17-2023
            IPeak[nnz] = np.max(np.max(XX_Itotal_Ir_Iphi_IzDisplay[:, nnz]))
            del Intensity
        # plt.plot(IPeak.get())
        # plt.show()
        # 传播面上光场计算结束
        DOF[i] = CPSWFs_FWHM_calculation(IPeak, Zd / lamc, Nz)  # 计算传播面上的焦深 行向量%选择中间的传播面 07-17-2023
        if abs(DOF[i]) > 20:
            DOF[i] = 5
        # DOF[i], _, _ = Fun_EfieldParameters(IPeak.T, Zd.T / self.lamc, self.Nz, SpotType)
        # print(DOF1)
        # print(DOF)
        Intensity_sum[i] = np.sum(
            IPeak[(2 * Nz + 1) // 3:2 * (2 * Nz + 1) // 3]) * 2 - np.sum(IPeak)  # 取中间 DOF 区域的强度和
        if Intensity_sum[i] < 0:
            Intensity_sum[i] = np.sum(
                IPeak[(2 * Nz + 1) // 3:2 * (2 * Nz + 1) // 3])
        IPeakmax = np.max(IPeak)
        In = np.where(IPeak == IPeakmax)[0]  # 找到最大强度对应的位置平面,取最后一个 In 中的元素
        Intensity_z = XX_Itotal_Ir_Iphi_IzDisplay[:, Nz]  # 设定焦平面
        X_Center = nn
        FWHM_x, SideLobeRatio_x, IntensPeak_x = Fun_EfieldParameters(Intensity_z, XX, X_Center, SpotType)
        IntensPeak_x = np.average(XX_Itotal_Ir_Iphi_IzDisplay[X_Center, Nz - 2: Nz + 2])
        # print(FWHM_x)
        FWHM_PersonalPresent0[i] = FWHM_x / 10.6
        IntensPeak_PersonalPresent0[i] = IntensPeak_x
        SideLobeRatio_PersonalPresent0[i] = SideLobeRatio_x
        if len(In) > 1:
            In = In[0]
        if In not in range(0, 2 * Nz + 1):
            # if ((In < 0) or (In > 2 * self.Nz)).all():
            In = Nz
        Focal_offset_PersonalPresent0[i] = np.abs(Zd[In] - FocalLength) / FocalLength  # 焦距的偏移量
        del Intensity_z
        # 计算波长结束
    ma, mbb = np.max(IntensPeak_PersonalPresent0), np.argmax(IntensPeak_PersonalPresent0)
    FWHM_PersonalPresent = FWHM_PersonalPresent0[mbb]  # 选择同一个波长下的数据2024-01-19
    IntensPeak_PersonalPresent= IntensPeak_PersonalPresent0[mbb]
    SideLobeRatio_PersonalPresent= SideLobeRatio_PersonalPresent0[mbb]
    Focal_offset_PersonalPresent= Focal_offset_PersonalPresent0[mbb]
    # FWHM_PersonalPresent = np.max(FWHM_PersonalPresent0)
    # IntensPeak_PersonalPresent = np.min(IntensPeak_PersonalPresent0)
    # SideLobeRatio_PersonalPresent = np.max(SideLobeRatio_PersonalPresent0)
    # Focal_offset_PersonalPresent = np.max(Focal_offset_PersonalPresent0)
    Fitness_PersonalPresent= Fun_FitnessLenserror(TargetFWHM, TargetSidlobe, TargetPeakIntensity,
                                                      TargetFocal_offset, FWHM_PersonalPresent,
                                                      SideLobeRatio_PersonalPresent,
                                                      IntensPeak_PersonalPresent,
                                                      Focal_offset_PersonalPresent, DOF, Intensity_sum)

    return Fitness_PersonalPresent


