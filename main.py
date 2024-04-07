# -*- coding:utf-8 -*-
# @Project    : InsectMicrolens
# @FileName   : main_pytorch.py
# @Author     : Spring
# @Time       : 2024/4/1 17:12
# @Description:
from MetalensOptimization import MetalensOptimization


def main():
    lens = MetalensOptimization(data_file='data/data1um.yaml')
    lens.init_phases()
    lens.phase_optimization()
    lens.save_results()


if __name__ == '__main__':
    main()
