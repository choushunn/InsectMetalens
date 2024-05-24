"""
# -*- coding:utf-8 -*-
# @Project    : InsectMetalens
# @FileName   : DataSaverCSV.py
# @Author     : SP
# @Time       : 2024/5/24 19:58
# @Description: 
"""
import pandas as pd
import os
from datetime import datetime


class DataSaverCSV:
    def __init__(self):
        # 使用当前日期和时间戳生成文件名
        self.file_name = datetime.now().strftime("data_%Y%m%d_%H%M%S.csv")
        self.file_exists = os.path.exists(self.file_name)

    def save_data(self, data):
        """
        data 是字典
        :param data:
        :return:
        """
        df = pd.DataFrame(data)
        if not self.file_exists:
            df.to_csv(self.file_name, mode='w', header=True, index=False)
            self.file_exists = True
        else:
            df.to_csv(self.file_name, mode='a', header=False, index=False)
        print(f"数据已追加保存到 {self.file_name}")
