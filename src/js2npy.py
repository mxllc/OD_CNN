import json
import numpy as np
import os

# 读取数据
with open('../CNN_data10/input/input_11.json', 'r') as f:   # 读取当前目录的json文件并解码成python数据
# with open('../CNN_dataprocess2/input_tensor.json', 'r') as f:  # 读取当前目录的json文件并解码成python数据
    data = json.load(f)
    data = np.array(data)
    print(np.shape(data))
    # np.save("D:/ML_Proj/OD_CNN/indata.npy", data)
    np.save("../generate_data/indata_11.npy", data)

print("end--------------------------------")

# 读取数据
with open('../CNN_data10/pathflow_labels/pathflow_labels_11.json', 'r') as f:   # 读取当前目录的json文件并解码成python数据
    data = json.load(f)
    data = np.array(data)
    print(np.shape(data))
    data = data.reshape(139, -1)
    print("new:")
    # print(data)
    print(np.shape(data))
    # np.save("D:/ML_Proj/OD_CNN/labeldata.npy", data)
    np.save("../generate_data/labeldata_11.npy", data)




# filePath = '../CNN_data10/'
# f1_lst = os.listdir(filePath)
# for item in f1_lst:
#     path_1 = filePath + item
#     # print(path_1)
#     path_1 = path_1 + '/'
#     f2_lst = os.listdir(path_1)
#     t_data = []
#     for item2 in f2_lst:
#         path_2 = path_1 + item2
#         print(path_2)
#         f = open(path_2, 'r')
#         data = json.load(f)
#         data = np.array(data)
#         if len(t_data) == 0:
#             t_data = data
#         else:
#             t_data = np.r_[t_data, data]
#         print(np.shape(data))
#         # np.save(open("../generate_data/indata2.npy", 'ab'), data)
#     print("t_data:", np.shape(t_data))
#     if item == "input":
#         np.save("../generate_data/indata.npy", t_data)
#     else:
#         np.save("../generate_data/labeldata.npy", t_data)


