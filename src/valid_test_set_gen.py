import json
import numpy as np

valid_test = ["27", "28"]
ip_path = "../CNN_data/input/input_"
pl_path = "../CNN_data/pathflow_labels/pathflow_labels_"
ip_sav_path = "../generate_data/indata_"
pl_sav_path = "../generate_data/labeldata_"
open_suf_path = ".json"
save_suf_path = ".npy"

for m_str in valid_test:
    # input
    o_path = ip_path + m_str + open_suf_path
    print("processing file: ", o_path)
    with open(o_path, 'r') as f:  # 读取当前目录的json文件并解码成python数据
        data = json.load(f)
        data = np.array(data)
        data = data[0:-1, :, :, :]
        print("input_data shape: ", np.shape(data))
        sav_path = ip_sav_path + m_str + save_suf_path
        np.save(sav_path, data)
        print("end--------------------------------")

    # label
    o_path = pl_path + m_str + open_suf_path
    print("processing file: ", o_path)
    with open(o_path, 'r') as f:  # 读取当前目录的json文件并解码成python数据
        data = json.load(f)
        data = np.array(data)
        print("original label data shape: ", np.shape(data))
        data = data[1:, -1:, :]
        data = np.reshape(data, (-1, 311))
        print("new: ", np.shape(data))
        sav_path = pl_sav_path + m_str + save_suf_path
        np.save(sav_path, data)

    print(m_str, "finished")
