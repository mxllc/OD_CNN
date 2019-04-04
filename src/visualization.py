import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import pyplot
import json
from src.utils import find_key_roads

# 1.标签数据的 hotshot 生成
# f_label = open('../CNN_dataprocess2/pathflow_labels.json', 'r')
f_label = open('../CNN_data10/pathflow_labels/pathflow_labels_3.json', 'r')
raw_label = json.load(f_label)
raw_label = np.array(raw_label)
print("label_shape:", np.shape(raw_label))

label = raw_label[:, 0, :]
label_data = np.append(label, raw_label[-1][1:, ], axis=0)
label_data = label_data[5:, :]
print("24 * 6  shape:", np.shape(label_data))

# 画图
plt.figure('Hot', facecolor='lightgray')
plt.title('real_hotshot', fontsize=20)  # ---------------------------实际的hotshot
plt.xlabel('road', fontsize=12)  # ---------------------------修改名称
plt.ylabel('time', fontsize=12)  # ---------------------------修改名称
plt.tick_params(labelsize=10)
plt.grid(linestyle=':')  # 背景网格

plt.imshow(label_data, cmap='jet', origin='low')
plt.colorbar().set_label('z', fontsize=16)

plt.savefig("../myplot/real_hotshot.png")  # 图像保存
plt.close()
# ======================================================================================


#
# 2.预测数据的hotshot
raw_prediction = np.load("./generate_data/prediction_result.npy")
# prediction = raw_prediction[:, 0, :]
# prediction_data = np.append(prediction, raw_prediction[-1][1:, ], axis=0)
# prediction_data = prediction_data[5:, :]
prediction_data = raw_prediction

# 画图
plt.figure('Hot', facecolor='lightgray')
plt.title('prediction_hotshot', fontsize=20)  # ---------------------------实际的hotshot
plt.xlabel('road', fontsize=12)  # ---------------------------修改名称
plt.ylabel('time', fontsize=12)  # ---------------------------修改名称
plt.tick_params(labelsize=10)
plt.grid(linestyle=':')  # 背景网格

plt.imshow(prediction_data, cmap='jet', origin='low')
plt.colorbar().set_label('z', fontsize=16)

plt.savefig("../myplot/prediction_hotshot.png")  # 图像保存
plt.close()
# ======================================================================================


#
# 3.实际数据与预测数据的差值 Difference 的 hotshot
differ = label_data - prediction_data
Key_road_num = 30
# 关键路径
key_road_total = find_key_roads()
key_road = key_road_total[:Key_road_num]

# mask: only 0 or 1
key_road_mask = np.zeros((311, ))
for i in key_road[::-1][:Key_road_num]:
    key_road_mask[i] = 1
tmp_ = np.arange(139)

key_road_mask, _ = np.meshgrid(key_road_mask, tmp_)
differ = differ * key_road_mask


# 画图
plt.figure('Hot', facecolor='lightgray')
plt.title('difference_hotshot', fontsize=20)  # ------------------
plt.xlabel('road', fontsize=12)  # ---------------------------修改名称
plt.ylabel('time', fontsize=12)  # ---------------------------修改名称
plt.tick_params(labelsize=10)
plt.grid(linestyle=':')  # 背景网格

plt.imshow(differ, cmap='jet', origin='low')
plt.colorbar().set_label('z', fontsize=16)

plt.savefig("../myplot/difference_hotshot.png")  # 图像保存
plt.close()
# ==============================================================================


#
# 4.训练过程中agv的变化过程
avg_ = np.load("./generate_data/avg_record.npy")
k_avg_ = np.load("./generate_data/key_road_avg_record.npy")
epoch = np.arange(0, 10001, 100)  # 生成101个
# print(np.shape(avg_))
# print(np.shape(epoch))
plt.ylabel("dist")
plt.xlabel("epoch")
plt.title("avg_dist")
# plt.plot(x,y,label="$sin(x)$",color="red",linewidth=2)
plt.plot(epoch, avg_[:-1], color='red', label='total')
plt.plot(epoch, k_avg_[:-1], color='blue', label='key_road')
plt.legend(labels=['total', 'key_road'], loc='best')
# plt.show()
plt.savefig("../myplot/avg_dist.png")  # 图像保存
plt.close()

#
# # plt.rcParams['figure.figsize'] = (20.0, 8.0) # 设置figure_size尺寸
# # plt.rcParams['image.interpolation'] = 'nearest' # 设置 interpolation style
# # plt.rcParams['savefig.dpi'] = 3000 #图片像素
# # plt.rcParams['figure.dpi'] = 3000 #分辨率


# plt.show()