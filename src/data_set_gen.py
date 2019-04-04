import json
import numpy as np
import os

filePath = '../CNN_data/'
f_lst = os.listdir(filePath)
# 在 CNN_data中查找， 有input和pathflow_labels两个文件夹
# 在运行的时候，暂时先把第28个数据从inPut和pathflow中去除

#
# 处理输入
input_path = filePath + f_lst[0]
print("open input file: ", input_path)
input_path = input_path + '/'
f1_lst = os.listdir(input_path)
i_data = []

for item in f1_lst:
    input_js_path = input_path + item
    print("dealing with: ", input_js_path)
    f = open(input_js_path, 'r')
    data = json.load(f)
    data = np.array(data)
    data = data[0:-1, :, :, :]
    if len(i_data) == 0:
        i_data = data
    else:
        i_data = np.r_[i_data, data]
    print("processed data shape: ", np.shape(data))
np.save("../generate_data/indata.npy", i_data)
print("indata.npy has been generated!")

#
# 处理标签
pathflow_label_path = filePath + f_lst[1]
print("open pathflow label file: ", pathflow_label_path)
label_path = pathflow_label_path + '/'
f2_lst = os.listdir(label_path)
l_data = []
for item in f2_lst:
    label_js_path = label_path + item
    print("dealing with: ", label_js_path)
    f = open(label_js_path, 'r')
    data = json.load(f)
    data = np.array(data)
    data = data[1:, -1:, :]
    data = np.reshape(data, (-1, 311))
    if len(l_data) == 0:
        l_data = data
    else:
        l_data = np.r_[l_data, data]
    print("processed data shape: ", np.shape(data))
np.save("../generate_data/labeldata.npy", l_data)
print("labeldata.npy has been generated!")
