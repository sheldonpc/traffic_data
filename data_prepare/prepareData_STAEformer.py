import numpy as np
import os
import pandas as pd
import argparse
import configparser
import warnings
import datetime
import re
import os
warnings.filterwarnings('ignore')

def time_add(data, week_start, interval=5, weekday_only=False, holiday_list=None, day_start=0, hour_of_day=24):
    # day and week
    if weekday_only:
        week_max = 5
    else:
        week_max = 7
    time_slot = hour_of_day * 60 // interval
    day_data = np.zeros_like(data)
    week_data = np.zeros_like(data)
    holiday_data = np.zeros_like(data)
    day_init = day_start
    week_init = week_start
    holiday_init = 1
    for index in range(data.shape[0]):
        if (index) % time_slot == 0:
            day_init = day_start
        day_init = day_init + 1 * (interval // 5)
        if (index) % time_slot == 0 and index !=0:
            week_init = week_init + 1
        if week_init > week_max:
            week_init = 1
        if day_init < 6:
            holiday_init = 1
        else:
            holiday_init = 2

        day_data[index:index + 1, :] = day_init
        week_data[index:index + 1, :] = week_init
        holiday_data[index:index + 1, :] = holiday_init

    if holiday_list is None:
        k = 1
    else:
        for j in holiday_list :
            holiday_data[j-1 * time_slot:j * time_slot, :] = 2
    return day_data, week_data, holiday_data

def data_type_init(DATASET, args):
    if DATASET == 'METR_LA' or DATASET == 'SZ_TAXI' or DATASET == 'PEMS07M':
        data_type = 'speed'
    elif DATASET == 'PEMS08' or DATASET == 'PEMS04' or DATASET == 'PEMS03' or DATASET == 'PEMS07':
        data_type = 'flow'
    elif DATASET == 'NYC_BIKE' or DATASET == 'NYC_TAXI' or DATASET == 'CHI_TAXI' or DATASET == 'CHI_BIKE':
        data_type = 'demand'
    elif DATASET == 'Electricity':
        data_type = 'MTS'
    elif DATASET == 'NYC_CRIME' or DATASET == 'CHI_CRIME':
        data_type = 'crime'
    elif DATASET == 'BEIJING_SUBWAY':
        data_type = 'people flow'
    elif DATASET == 'chengdu_didi' or 'shenzhen_didi':
        data_type = 'index'
    else:
        raise ValueError

    args.data_type = data_type

# load dataset
def load_st_dataset(dataset, args):
    #output B, N, D
    # 1 / 1 / 2018 - 2 / 28 / 2018 Monday
    if dataset == 'PEMS04':
        data_path = os.path.join('../data/PEMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0]  # onley the first dimension, traffic flow data
        print(data.shape, data[data==0].shape)
        week_start = 1
        interval = 5
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        holiday_list = [1, 15, 50]
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)

    # 7 / 1 / 2016 - 8 / 31 / 2016 Friday
    elif dataset == 'PEMS08':
        data_path = os.path.join('../data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0]  # only the first dimension, traffic flow data
        print(data.shape, data[data==0].shape)
        week_start = 5
        holiday_list = [4]
        interval = 5
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)

    #   9/1/2018 - 11/30/2018 Saturday
    elif dataset == 'PEMS03':
        data_path = os.path.join('../data/PEMS03/PEMS03.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
        week_start = 6
        interval = 5
        week_day = 7
        holiday_list = None
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)

    # 5 / 1 / 2017 - 8 / 31 / 2017 Monday
    elif dataset == 'PEMS07':
        data_path = os.path.join('../data/PEMS07/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, 0]  # only the first dimension, traffic flow data
        week_start = 1
        interval = 5
        week_day = 7
        holiday_list = None
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)

    # 1 / 1 / 2017 - 2 / 28 / 2017 Sunday
    elif dataset == 'CA_District5':
        data_path = os.path.join('../data/CA_District5/CA_District5.npz')
        data = np.load(data_path)['data'][:, :]  # only the first dimension, traffic flow data
        week_start = 7
        interval = 5
        week_day = 7
        holiday_list = None
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)

    # 1 / 1 / 2018 - 4 / 30 / 2018 Monday
    elif dataset == 'chengdu_didi':
        data_path = os.path.join('../data/chengdu_didi/chengdu_didi.npz')
        data = np.load(data_path)['data'][:, :, 0]  # only the first dimension, traffic index
        print(data.shape, data[data==0].shape)

        week_start = 1
        holiday_list = [4]
        interval = 10
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)

    # 5 / 1 / 2012 - 6 / 30 / 2012 Tuesday
    elif dataset == 'PEMS07M':
        data_path = os.path.join('../data/PEMS07M/PEMS07M.npz')
        data = np.load(data_path)['data']  # only traffic speed data
        week_start = 2
        weekday_only = True
        interval = 5
        week_day = 5
        args.interval = interval
        args.week_day = week_day
        holiday_list = []
        day_data, week_data, holiday_data = time_add(data, week_start, interval, weekday_only, holiday_list=holiday_list)

    elif dataset == 'NYC_BIKE':
        data_path = os.path.join('../data/NYC_BIKE/NYC_BIKE.npz')
        data = np.load(data_path)['data'][..., 0].astype(np.float64)
        print(data.dtype,)
        week_start = 5
        weekday_only = False
        interval = 30
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        holiday_list = []
        day_data, week_data, holiday_data = time_add(data, week_start, interval, weekday_only,
                                                     holiday_list=holiday_list)
    else:
        raise ValueError

    args.num_nodes = data.shape[1]

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
        day_data = np.expand_dims(day_data, axis=-1).astype(int)
        week_data = np.expand_dims(week_data, axis=-1).astype(int)
        # holiday_data = np.expand_dims(holiday_data, axis=-1).astype(int)
        data = np.concatenate([data, day_data, week_data], axis=-1)
    elif len(data.shape) > 2:
        day_data = np.expand_dims(day_data, axis=-1).astype(int)
        week_data = np.expand_dims(week_data, axis=-1).astype(int)
        data = np.concatenate([data, day_data, week_data], axis=-1)
    else:
        raise ValueError

    print('Load %s Dataset shaped: ' % dataset, data.shape, data[..., 0:1].max(), data[..., 0:1].min(),
          data[..., 0:1].mean(), np.median(data[..., 0:1]), data.dtype)
    return data

# te时间嵌入的分割
def seq2instance(data, num_his, num_pred):
    # 这里的步长 和 dims维度确认一下
    num_step, dims = data.shape
    # 为什么采样数据长度要减去历史 减去预测 与数据分割保持一致
    num_sample = num_step - num_his - num_pred + 1
    x = np.zeros(shape = (num_sample, num_his, dims))
    y = np.zeros(shape = (num_sample, num_pred, dims))
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
    return x, y

# 整体数据集的分割
def seq2instance_plus(data, num_his, num_pred):
    num_step = data.shape[0]
    # 确保最后一个x有对应的y值
    # 西雅图数据只取了第一列的数据 交通流量
    num_sample = num_step - num_his - num_pred + 1 # 5161
    x = []
    y = []
    # 从0开始 不重叠的取数据
    for i in range(num_sample):
        # x (24,170,1) 步长
        x.append(data[i: i + num_his])
        # pems的数据也可以这么取
        # y.append(data[i + num_his: i + num_his + num_pred])
        y.append(data[i + num_his: i + num_his + num_pred, :, :1])
    x = np.array(x)
    y = np.array(y)
    return x, y


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # 读取配置文件
    parser.add_argument("--config", default='./data_conf/PEMS08_3dim_construct_12.conf', type=str,
                        help="configuration file path")
    # 数据集 dataset loader
    parser.add_argument('--interval', type=int, help='Interval of time steps', default=5)
    parser.add_argument('--week_day', type=int, help='Number of days in a week', default=7)

    args = parser.parse_args()
    config = configparser.ConfigParser()
    print('Read configuration file: %s' % (args.config))
    config.read(args.config)
    # 区分两类数据 数据+训练
    data_config = config['Data']
    training_config = config['Training']

    time_slice_size = int(data_config['time_slice_size']) # 数据切片大小 （一个切片的长度是多少）
    train_ratio = float(data_config['train_ratio']) # 训练集所占比例
    val_ratio = float(data_config['val_ratio']) # 验证集所占比例
    test_ratio = float(data_config['test_ratio']) # 测试集所占比例
    num_his = int(training_config['num_his']) # 历史步长
    num_pred = int(training_config['num_pred']) # 预测步长
    num_of_vertices = int(data_config['num_of_vertices']) # 节点数量 （这里看一下怎么捕获空间依赖性）

    data_file = data_config['data_file'] # 数据文件地址
    filename = os.path.basename(data_file)

    if filename == "CA_District5.npz":
        dataset_name = "CA_District5"
        data = load_st_dataset(dataset_name, args)

        # 提取最后一个维度上的第二列数据
        second_col = data[:, :, 1]
        # 计算最小值和最大值
        min_val = np.min(second_col)
        max_val = np.max(second_col)

        # 执行归一化操作 (Min-Max 归一化)
        second_col_normalized = (second_col - min_val) / (max_val - min_val)

        # 将归一化后的数据放回原数组
        data[:, :, 1] = second_col_normalized

    elif filename == "PEMS03.npz":
        dataset_name = "PEMS03"
        data = load_st_dataset(dataset_name, args)

        # 提取最后一个维度上的第二列数据
        second_col = data[:, :, 1]
        # 计算最小值和最大值
        min_val = np.min(second_col)
        max_val = np.max(second_col)

        # 执行归一化操作 (Min-Max 归一化)
        second_col_normalized = (second_col - min_val) / (max_val - min_val)

        # 将归一化后的数据放回原数组
        data[:, :, 1] = second_col_normalized

    elif filename == "chengdu_didi.npz":
        dataset_name = "chengdu_didi"
        data = load_st_dataset(dataset_name, args)

        # 提取最后一个维度上的第二列数据
        second_col = data[:, :, 1]
        # 计算最小值和最大值
        min_val = np.min(second_col)
        max_val = np.max(second_col)

        # 执行归一化操作 (Min-Max 归一化)
        second_col_normalized = (second_col - min_val) / (max_val - min_val)

        # 将归一化后的数据放回原数组
        data[:, :, 1] = second_col_normalized

    elif filename == "PEMS07M.npz":
        dataset_name = "PEMS07M"
        data = load_st_dataset(dataset_name, args)

        # 提取最后一个维度上的第二列数据
        second_col = data[:, :, 1]
        # 计算最小值和最大值
        min_val = np.min(second_col)
        max_val = np.max(second_col)

        # 执行归一化操作 (Min-Max 归一化)
        second_col_normalized = (second_col - min_val) / (max_val - min_val)

        # 将归一化后的数据放回原数组
        data[:, :, 1] = second_col_normalized

    elif filename == "NYC_BIKE.npz":
        dataset_name = "NYC_BIKE"
        data = load_st_dataset(dataset_name, args)

        # 提取最后一个维度上的第二列数据
        second_col = data[:, :, 1]
        # 计算最小值和最大值
        min_val = np.min(second_col)
        max_val = np.max(second_col)

        # 执行归一化操作 (Min-Max 归一化)
        second_col_normalized = (second_col - min_val) / (max_val - min_val)

        # 将归一化后的数据放回原数组
        data[:, :, 1] = second_col_normalized

    else:
        files = np.load(data_file, allow_pickle=True) # 从 .npy 或 .npz 文件加载数组或者数组集合
        data=files['data']
    print(data.shape) # 3D Pems08 (17856, 170, 3)
    print("Dataset: ", data.shape, data[5, 0, :])

    slices = data.shape[0]
    train_slices = int(slices * 0.6) # 训练集数量
    val_slices = int(slices * 0.2) # 验证集数量
    test_slices = slices - train_slices - val_slices # 测试集数量 （为了能够整除 总量-训练-验证）
    train_set = data[ : train_slices] # 训练集
    print(train_set.shape) # 验证集
    val_set = data[train_slices : val_slices + train_slices] # 测试集
    print(val_set.shape)
    test_set = data[-test_slices : ]
    print(test_set.shape)

    sets = {'train': train_set, 'val': val_set, 'test': test_set}
    xy = {}
    te = {}
    for set_name in sets.keys():
        data_set = sets[set_name]
        X, Y = seq2instance_plus(data_set[..., :].astype("float64"), num_his, num_pred)
        xy[set_name] = [X, Y]

    # 训练集 验证集 测试集最终结果
    x_trains, y_trains = xy['train'][0], xy['train'][1]
    x_vals, y_vals = xy['val'][0], xy['val'][1]
    x_tests, y_tests = xy['test'][0], xy['test'][1]

    print("train: ", x_trains.shape, y_trains.shape)
    print("val: ", x_vals.shape, y_vals.shape)
    print("test: ", x_tests.shape, y_tests.shape)
    output_dir = data_config['output_dir']
    output_path = os.path.join(output_dir, "samples_" + str(num_his) + "_" + str(num_pred) + "_" + str(time_slice_size) + ".npz")
    print(f"save file to {output_path}")
    np.savez_compressed(
            output_path,
            train_x=x_trains, train_target=y_trains,
            val_x=x_vals, val_target=y_vals,
            test_x=x_tests, test_target=y_tests)
