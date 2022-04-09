import logging
from datetime import datetime
import sys
from logging.handlers import RotatingFileHandler
import os
import time

class Config:

    # 基金简称
    stock_name = None

    # 显示日期
    stock_date = None

    # 测试数据
    seq = None
    test_data = None

    # 归一化数据
    std = None
    mean = None

    # 随机数种子
    random_seed = 42
    

    # 数据参数
    # 要作为feature的列，按原数据从0开始计算
    feature_columns = list((1,3,6))
    # 要预测的列
    label_columns = [1,]

    # 预测未来的几天
    predict_day = 1


    # 网络参数
    # 这个参数很重要，是设置用前多少天的数据来预测，也是LSTM的time step数，请保证训练数据量大于它
    time_step = 30
    # 输入大小
    input_size = len(feature_columns)
    # 输出大小
    out_size = len(label_columns)

    # LSTM的隐藏层大小，也是输出大小
    hidden_size = 64
    # LSTM的堆叠层数
    lstm_layers = 1
    # dorpout概率
    dropout_rate = 0

    # 训练参数
    do_train = True
    # 预测
    do_predict = True
    # 训练数据占全部数据的占比
    train_data_rate = 0.7
    # 验证数据占训练数据的占比
    valid_data_rate = 0.15 
    # 每次训练把上一次的final_state作为下一次的init_state，仅用于RNN类型模型(不要动，留着拓展用的)
    do_continue_train = True
    # 是否对训练数据做shuffle
    shuffle_train_data = False
    # 是否使用gpu加速
    use_cuda = True
    # 是否载入已有模型参数进行训练
    add_train = False  
    # 学习率
    learning_rate = 0.01
    # 整个训练集被训练多少遍，不考虑早停的情况下
    epoch = 20
    # 训练多少个epoch开始验证， = -1 不使用早停
    patience = 5
    # 一次训练所选取的样本数
    batch_size = 64

    # log设置
    do_log_print_to_screen = True
    do_log_save_to_file = True

    do_train_visualized = False

    # 是否保存图片
    do_figure_save = True

    # 框架参数
    # 使用的学习框架
    frame = 'pytorch'
    model_postfix = {'pytorch': '.pth',}
    model_name = f"_{datetime.now().strftime('%Y%m%d')}{model_postfix[frame]}"

    # 路径参数
    url_path = './data/url.csv' 
    data_path = './data/'
    model_save_path = './checkpoint/'
    log_save_path = './log/'
    figure_save_path = './figure/'
    train_loss_path = './train_loss'
    valid_loss_path = './valid_loss'
    valid_ac_path = './valid_ac'
    read_data_path = None
    clean_data_path = None
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)    # makedirs 递归创建目录
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)
    if do_train and (do_log_save_to_file or do_train_visualized):
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        log_save_path = log_save_path + cur_time + '_' + frame + "/"
        os.makedirs(log_save_path)




    def load_logger(self):
        logger = logging.getLogger()
        logger.setLevel(level=logging.DEBUG)

        # StreamHandler
        if self.do_log_print_to_screen:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(level=logging.INFO)
            formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',
                                        fmt='[ %(asctime)s ] %(message)s')
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
    
        if self.do_log_save_to_file:
            file_handler = RotatingFileHandler(self.log_save_path + "out.log", maxBytes=1024 * 1000 * 10, backupCount=5)
            file_handler.setLevel(level=logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            # 把config信息也记录到log 文件中
            config_dict = {}
            for key in dir(self):
                if not key.startswith("_"):
                    config_dict[key] = getattr(self, key)
            config_str = str(config_dict)
            config_list = config_str[1:-1].split(", '")
            config_save_str = "\nConfig:\n" + "\n'".join(config_list)
            logger.info(config_save_str)

        return logger