# -*- coding:utf-8 -*-
# @Project    : InsectMetalens
# @FileName   : SGD.py
# @Author     : Spring
# @Time       : 2024/4/2 15:38
# @Description:

from main import MetalensOptimization


class SGD:
    def __init__(self):
        self.im = MetalensOptimization()
        self.phases = self.im.init_phases()


if __name__ == '__main__':
    sgd = SGD()
    print(sgd.im.phases)
    # sgd.im.plot_phases()
