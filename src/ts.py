import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import pyplot
import json
from src.utils import find_key_roads

# 1.标签数据的 hotshot 生成
label_data = np.load("../generate_data/labeldata_test.npy")

# 画图
plt.figure('Hot', facecolor='lightgray')
plt.title('Real Hotshot', fontsize=20)  # ---------------------------实际的hotshot
plt.xlabel('Road', fontsize=12)  # ---------------------------修改名称
plt.ylabel('Time', fontsize=12)  # ---------------------------修改名称
plt.tick_params(labelsize=10)
plt.grid(linestyle=':')  # 背景网格

plt.imshow(label_data, cmap='jet', origin='low')
plt.colorbar().set_label('z', fontsize=16)

plt.savefig("../myplot/real_hotshot.png")  # 图像保存
plt.close()
print("real_hotshot has been generated.")
# ======================================================================================

#
# 2.预测数据的hotshot
raw_prediction = np.load("./generate_data/prediction_result.npy")
prediction_data = raw_prediction

# 画图
plt.figure('Hot', facecolor='lightgray')
plt.title('Prediction Hotshot', fontsize=20)  # ---------------------------实际的hotshot
plt.xlabel('Road', fontsize=12)  # ---------------------------修改名称
plt.ylabel('Time', fontsize=12)  # ---------------------------修改名称
plt.tick_params(labelsize=10)
plt.grid(linestyle=':')  # 背景网格

plt.imshow(prediction_data, cmap='jet', origin='low')
plt.colorbar().set_label('z', fontsize=16)

plt.savefig("../myplot/prediction_hotshot.png")  # 图像保存
plt.close()
print("prediction_hotshot has been generated.")
# ======================================================================================

#
#
# 3.实际数据与预测数据的差值 Difference 的 hotshot
differ = label_data - prediction_data

## 显示关键路径的differ
# Key_road_num = 30
# # 关键路径
# key_road_total = find_key_roads()
# key_road = key_road_total[:Key_road_num]
#
# # mask: only 0 or 1
# key_road_mask = np.zeros((311, ))
# for i in key_road[::-1][:Key_road_num]:
#     key_road_mask[i] = 1
# tmp_ = np.arange(139)
#
# key_road_mask, _ = np.meshgrid(key_road_mask, tmp_)
# differ = differ * key_road_mask

# 画图
plt.figure('Hot', facecolor='lightgray')
plt.title('Difference Hotshot', fontsize=20)  # ------------------
plt.xlabel('Road', fontsize=12)  # ---------------------------修改名称
plt.ylabel('Time', fontsize=12)  # ---------------------------修改名称
plt.tick_params(labelsize=10)
plt.grid(linestyle=':')  # 背景网格

plt.imshow(differ, cmap='jet', origin='low')
plt.colorbar().set_label('z', fontsize=16)

plt.savefig("../myplot/difference_hotshot.png")  # 图像保存
plt.close()
print("difference_hotshot has been generated.")
# ==============================================================================

#
# 4.训练过程中agv的变化过程
t_RMSE = np.load("./generate_data/test_RMSE.npy")
tk_RMSE = np.load("./generate_data/test_key_road_RMSE.npy")
v_RMSE = np.load("./generate_data/valid_RMSE.npy")
vk_RMSE = np.load("./generate_data/valid_key_road_RMSE.npy")

epoch = np.arange(0, 10001, 100)

# plt.plot(t_RMSE, "r-+", linewidth=2, label="total_train")
# plt.plot(v_RMSE, "g-", linewidth=3, label="total_val")
# plt.plot(tk_RMSE, "y-+", linewidth=2, label="key_train")
# plt.plot(vk_RMSE, "b-", linewidth=3, label="key_val")

plt.title("Learning Curve")
# plt.legend(loc="upper right", fontsize=14)  # not shown in the book
plt.xlabel("Training epoch", fontsize=14)  # not shown
plt.ylabel("RMSE", fontsize=14)  # not shown

plt.plot(epoch, t_RMSE, 'cx--', color='red', label='total_train')
plt.plot(epoch, v_RMSE, 'm*:', color='green', label='total_val')
plt.plot(epoch, tk_RMSE, 'k<-.', color='cyan', label='key_train')
plt.plot(epoch, vk_RMSE, color='blue', label='key_val')

plt.legend(labels=['total_train', 'total_val', 'key_train', 'key_val'], loc='upper right')
# plt.show()
plt.savefig("../myplot/Learning_Curve.png")  # 图像保存
plt.close()
print("learning_curve has been generated.")
