# coding=utf-8
import numpy as np
from pylab import *
from matplotlib import pyplot as plt
import json

#
# # 读取数据
# with open('./CNN_dataprocess2/pathflow_labels.json', 'r') as f:   # 读取当前目录的json文件并解码成python数据
#     data = json.load(f)
#     data = np.array(data)
#     print("data_shape:", np.shape(data))
#     data = data.reshape(139, -1)
#     print("new:")
#     print(data)
#     print(np.shape(data))
#     # np.save("D:/ML_Proj/OD_CNN/labeldata.npy", data)
#
#
#
# x = [1, 2, 3, 4]
# y = [3, 5, 10, 25]
#
# # 创建Figure
# fig = plt.figure()
#
# # 创建一个或多个子图(subplot绘图区才能绘图)
# ax1 = fig.add_subplot(231)
# plt.plot(x, y, marker='D')  # 绘图及选择子图
# plt.sca(ax1)
#
# ax2 = fig.add_subplot(232)
# plt.scatter(x, y, marker='s', color='r')
# plt.sca(ax2)
# plt.grid(True)
#
# ax3 = fig.add_subplot(233)
# plt.bar(x, y, 0.5, color='c')  # 柱状图 width=0.5间距
# plt.sca(ax3)
#
# ax4 = fig.add_subplot(234)
# # 高斯分布
# mean = 0  # 均值为0
# sigma = 1  # 标准差为1 (反应数据集中还是分散的值)
# data = mean + sigma * np.random.randn(10000)
# plt.hist(data, 40, normed=1, histtype='bar', facecolor='yellowgreen', alpha=0.75)
# plt.sca(ax4)
#
# m = np.arange(-5.0, 5.0, 0.02)
# n = np.sin(m)
# ax5 = fig.add_subplot(235)
# plt.plot(m, n)
# plt.sca(ax5)
#
# ax6 = fig.add_subplot(236)
# xlim(-2.5, 2.5)  # 设置x轴范围
# ylim(-1, 1)  # 设置y轴范围
# plt.plot(m, n)
# plt.sca(ax6)
# plt.grid(True)
#
# plt.show()


# 寻找路网中的关键路径
def find_key_roads():
    data = np.load("../generate_data/labeldata.npy")
    data = np.reshape(data, (-1, 6, 311))
    print("data_shape:", np.shape(data))
    label = data[:, 0, :]
    label_data = np.append(label, data[-1][1:, ], axis=0)
    print("label_data_shape:", np.shape(label_data))
    road_sum = label_data.sum(axis=0)
    print("road_sum_shape:", np.shape(road_sum))
    index_ = np.argsort(road_sum)

    # ===================================================
    # # road_sum[index_[::-1]][:30]
    # # print(index_[::-1][:30])
    #
    # key_road_mask = np.zeros((311, ))
    #
    # for i in index_[::-1][:30]:
    #     key_road_mask[i] = 1
    # # print(key_road_mask)
    #
    # Key_road_num = 30
    # key_road_weight = np.ones((311,))
    # cnt = Key_road_num
    # for i in index_[::-1][:Key_road_num]:
    #     key_road_weight[i] = cnt
    #     cnt -= 1
    # # print(key_road_weight)
    # key_road_weight = np.concatenate((key_road_weight, key_road_weight, key_road_weight, key_road_weight,
    #                                   key_road_weight, key_road_weight), axis=0)
    # key_road_weight_ = np.reshape(key_road_weight, (1, -1))
    # # for i in range(10):
    # #     key_road_we
    # print("my:", np.shape(key_road_weight_))
    # # key_road_weight_i = np.concatenate((key_road_weight, key_road_weight),axis=1)
    # tmp_ = np.arange(50)
    # key_road_weight, _ = np.meshgrid(key_road_weight_, tmp_)
    # print("my:", np.shape(key_road_weight))
    # # for i in key_road_weight_:


    # print(road_sum)
    # print("\n hhh:\n",  index_[::-1])
    # print(road_sum[index_[::-1]])

    # print("\n hhh:\n", index_)
    # print(road_sum[index_])
    # 降序排列， index_[][0]表示最大数的索引
    return index_[::-1]


if __name__ == "__main__":
    find_key_roads()
