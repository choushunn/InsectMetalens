# -*- coding:utf-8 -*-
# @Project    : InsectMicrolens
# @FileName   : main_pytorch.py
# @Author     : Spring
# @Time       : 2024/4/1 17:12
# @Description:
from MetalensOptimization import MetalensOptimization


def parse_args():
    """
    解析命令行参数
    :return:
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='data/data1um.yaml')
    parser.add_argument('--method', type=str, default='SGD', help='优化方法，SGD/Adam/AdamW')
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument('--show', action='store_true', help='是否显示图像')
    args = parser.parse_args()
    return args


def main(opt):
    """
    主函数
    :param opt:
    :return:
    """
    lens = MetalensOptimization(opt)
    lens.phase_optimization()
    lens.save_results()


if __name__ == '__main__':
    opt = parse_args()
    main(opt)
