#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
from math import sqrt
from src.utils import find_key_roads


Init_Channel = 18
Conv1_Ch = 8
Conv2_Ch = 16
Conv3_Ch = 32

Conv1_size = 5
Conv2_size = 4
Conv3_size = 3

Conv1_stride = 3
Conv2_stride = 2
Conv3_stride = 1


OUTNUM = 400
BATCH = 50

NODE_NUM = 116
Road_num = 311

Key_road_num = 30

def get_key_road_mask():
    a = 1


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def createNetwork():
    # 定义深度神经网络的参数和偏置
    W_conv1 = weight_variable([Conv1_size, Conv1_size, Init_Channel, Conv1_Ch])
    b_conv1 = bias_variable([Conv1_Ch])

    W_conv2 = weight_variable([Conv2_size, Conv2_size, Conv1_Ch, Conv2_Ch])
    b_conv2 = bias_variable([Conv2_Ch])

    W_conv3 = weight_variable([Conv3_size, Conv3_size, Conv2_Ch, Conv3_Ch])
    b_conv3 = bias_variable([Conv3_Ch])

    # 输入层
    s = tf.placeholder("float", [None, NODE_NUM, NODE_NUM, Init_Channel])

    # 隐藏层
    h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(s, W_conv1, Conv1_stride), b_conv1))
    debug_p = np.shape(h_conv1).as_list()
    print("h_conv1: ", debug_p)

    h_pool1 = max_pool_2x2(h_conv1)
    debug_p = np.shape(h_pool1).as_list()
    print("h_pool1: ", debug_p)

    h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_pool1, W_conv2, Conv2_stride), b_conv2))
    debug_p = np.shape(h_conv2).as_list()
    print("h_conv2: ", debug_p)

    h_pool2 = max_pool_2x2(h_conv2)
    debug_p = np.shape(h_pool2).as_list()
    print("h_pool2: ", debug_p)

    h_conv3 = tf.nn.relu(tf.nn.bias_add(conv2d(h_pool2, W_conv3, Conv3_stride), b_conv3))
    debug_p = np.shape(h_conv3).as_list()
    print("h_conv3: ", debug_p)

    h_conv3_flat = tf.reshape(h_conv3, [-1, 800])  # -------------------------------修改这里

    W_fc1 = weight_variable([800, Road_num])
    b_fc1 = bias_variable([Road_num])

    # 输出层
    readout = tf.matmul(h_conv3_flat, W_fc1) + b_fc1

    return s, readout


def trainNetwork(s, readout, sess, max_epoch=10000, lr=1e-6):
    # 定义损失函数
    y_ = tf.placeholder("float", [None, Road_num])
    my_input = s

    # 关键路径
    key_road_total = find_key_roads()
    key_road = key_road_total[:Key_road_num]

    # mask: only 0 or 1
    key_road_mask = np.zeros((311, ))
    for i in key_road[::-1][:Key_road_num]:
        key_road_mask[i] = 1

    tmp_ = np.arange(138)

    key_road_mask, _ = np.meshgrid(key_road_mask, tmp_)

    # weight for loss
    key_road_weight = np.ones((311,))
    cnt = Key_road_num
    for i in key_road[::-1][:Key_road_num]:
        key_road_weight[i] = cnt
        cnt -= 1


    # key_road_weight = np.concatenate((key_road_weight, key_road_weight, key_road_weight, key_road_weight,
    #                                   key_road_weight, key_road_weight), axis=0)

    tmp_ = np.arange(BATCH)
    key_road_weight, _ = np.meshgrid(key_road_weight, tmp_)

    #
    #
    # 定义损失函数
    result_delta = tf.subtract(readout, y_)
    # abs_result = tf.abs(result_delta)
    result_square = tf.multiply(result_delta, result_delta)
    weight_result_square = tf.multiply(result_square, key_road_weight)
    key_road_square = tf.multiply(result_square, key_road_mask)

    weight_loss = tf.reduce_mean(weight_result_square)
    MSE_loss = tf.reduce_mean(result_square)
    key_road_MSE_loss = tf.reduce_mean(key_road_square)
    train_step = tf.train.AdamOptimizer(lr).minimize(weight_loss)

    # 用于加载或保存网络参数
    saver = tf.train.Saver(max_to_keep=3)
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    training_epoch = 0
    # 开始训练
    total_data = np.load("../generate_data/indata.npy")
    total_label = np.load("../generate_data/labeldata.npy")

    test_data = np.load("../generate_data/indata_test.npy")
    test_label = np.load("../generate_data/labeldata_test.npy")

    valid_data = np.load("../generate_data/indata_valid.npy")
    valid_label = np.load("../generate_data/labeldata_valid.npy")

    #
    # 记录训练中的偏差，用于生成学习曲线
    test_avg_list = []
    test_key_road_avg_list = []
    valid_avg_list = []
    valid_key_road_avg_list = []

    while training_epoch <= max_epoch:
        # 数据读取
        total_length = len(total_data)
        # 从所有数据中选择batch个
        random_choice = random.sample([i for i in range(total_length)], BATCH)
        valid_indices = np.zeros(total_length, np.bool_)

        for i in random_choice:
            valid_indices[i] = True

        # 使用'bool'来进行选择 下标为True的数据项被选出
        input_data = total_data[valid_indices]
        label_data = total_label[valid_indices]

        # 使用字典传递数据，开始训练
        w_loss, _ = sess.run([weight_loss, train_step], feed_dict={my_input: input_data, y_: label_data})

        if training_epoch % 100 == 0:
            test_MSE, test_k_MSE, = sess.run([MSE_loss, key_road_MSE_loss],
                                             feed_dict={my_input: test_data, y_: test_label})
            val_MSE, val_k_MSE, = sess.run([MSE_loss, key_road_MSE_loss],
                                           feed_dict={my_input: valid_data, y_: valid_label})

            test_avg_dist = sqrt(test_MSE)
            test_k_avg_dist = sqrt(test_k_MSE)
            val_avg_dist = sqrt(val_MSE)
            val_k_avg_dist = sqrt(val_k_MSE)

            print("epoch=%d, weight_loss=%.5f, v_k_RMSE=%.5f, v_RMSE=%.5f, t_k_RMSE=%.5f, t_RMSE=%.5f"
                  % (training_epoch, w_loss, val_k_avg_dist, val_avg_dist, test_k_avg_dist, test_avg_dist))

            test_avg_list.append(test_avg_dist)
            test_key_road_avg_list.append(test_k_avg_dist)
            valid_avg_list.append(val_avg_dist)
            valid_key_road_avg_list.append(val_k_avg_dist)

            saver.save(sess, 'saved_networks/od', global_step=training_epoch)
        # 每进行10000次迭代，保留一下网络参数
        # if (training_epoch + 1) % 1000 == 0:
        #     saver.save(sess, 'saved_networks/od', global_step=training_epoch)

        training_epoch += 1

    print("This time training has finished! Total training epoch are %d" % (training_epoch - 1, ))
    np_tavg = np.array(test_avg_list)
    np.save("./generate_data/test_RMSE.npy", np_tavg)
    np_tk_avg = np.array(test_key_road_avg_list)
    np.save("./generate_data/test_key_road_RMSE.npy", np_tk_avg)
    np_vavg = np.array(valid_avg_list)
    np.save("./generate_data/valid_RMSE.npy", np_vavg)
    np_vk_avg = np.array(valid_key_road_avg_list)
    np.save("./generate_data/valid_key_road_RMSE.npy", np_vk_avg)
    return


def predict_network(s, readout, sess):
    # 用于加载或保存网络参数
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
        return
    my_input = s
    total_data = np.load("../generate_data/indata_test.npy")
    # total_label = np.load("../generate_data/labeldata.npy")
    # predict_data = np.load("../generate_data/predict_data.npy")
    result = sess.run([readout], feed_dict={my_input: total_data})
    result = np.array(result)
    print(np.shape(result))
    result = result.reshape(-1, 311)
    # result = result.reshape(40, 40)

    print("output_shape:", np.shape(total_data[0]))
    np.save("./generate_data/prediction_result.npy", result)
    print("prediction results have been printed into the file.")
    # np.savetxt('new.csv', result, delimiter=',')
    # np.savetxt('total.csv', total_data[0], delimiter=',')

    # total = 0
    # for i in range(138):
    #     for j in range(i+1, 139):
    #         ts = np.array(total_data[i] != total_data[j])
    #         ts = ts.astype(int)
    #         total += ts.sum()
    #         # print(ts.sum())
    # # print(total_data[23] != total_data[92])
    # # ts = np.array(total_data[23] != total_data[82])
    # # ts = ts.astype(int)
    # # total = ts.sum()
    # print(total)

    # print(result)
    return result


def train_cnn():
    sess = tf.InteractiveSession()
    s, readout = createNetwork()
    trainNetwork(s, readout, sess)


def predict_cnn():
    sess = tf.InteractiveSession()
    s, readout = createNetwork()
    predict_network(s, readout, sess)


if __name__ == "__main__":
    chochr = input("a.training or b.prediction\n")
    if chochr == 'a':
        train_cnn()
    elif chochr == 'b':
        predict_cnn()
    else:
        print("error choose")
