
import os
import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from datetime import date, datetime, timedelta
from StockCrawler import *
frame = 'pytorch'
if frame == 'pytorch':
    from model.m_pytorch import train, predict
else:
    raise Exception("Wrong frame seletion")

class Data:

    def __init__(self,config):
        self.cfig = config
        self.cfig.frame = frame
        # self.data shape(len(init_data), len(init_data.columns.tolist()))
        self.data, self.data_column_name = self.read_data()
        # 颠倒数据，现在的数据是从现在的时间往前排的，预测是从前往后预测
        self.data = self.data[::-1]

        # 数据数量 行数
        self.data_num = self.data.shape[0]
        # 训练数据的数量 
        self.train_num = int(self.data_num * self.cfig.train_data_rate)

        # 求均值
        self.mean = np.mean(self.data, axis=0)
        self.cfig.mean = self.mean
        # 求方差
        self.std = np.std(self.data, axis=0)
        self.cfig.std = self.std
        # 归一化，去量纲
        self.norm_data = (self.data - self.mean)/self.std
        self.cfig.test_data = self.norm_data[self.train_num:]
        self.label_y = []
    

    def read_data(self):
        '''
        
        打开爬虫获取到的数据，进行数据清洗
        提取'单位净值' '上证涨跌' 两列作为特征数据 '涨跌' 作为标签数据

        '''
    
        init_data = pd.read_csv(self.cfig.read_data_path,encoding='gbk')
        # 删除‘日增长率’和‘上证涨跌’中那个的百分号
        init_data['日增长率'] = init_data['日增长率'].str[:-1]
        init_data['上证涨跌'] = init_data['上证涨跌'].str[:-1]
        # 将‘日增长率’中的符号改为0
        init_data.loc[init_data['日增长率']=='-','日增长率'] = 0.00
        # 添加根据‘日增长率’ 创建‘涨跌’列 
        # 日增涨率 为0时涨跌为0，日增长率 > 0 时涨跌为1 else 涨跌为-1
        init_data['涨跌'] = init_data['日增长率'].apply(lambda x: 0 if x == 0 else ( 1 if float(x) > 0 else -1))
        # 删除封闭期的数据
        # init_data = init_data.loc[init_data['申购状态']=='开放申购']
        # 将‘上证涨跌’中可能处出现的空字符转换成空值
        init_data.loc[init_data['上证涨跌']=='','上证涨跌'] = np.nan
        # 删除存在空值的行
        init_data = init_data.dropna()
        # 将‘上证涨跌’数据类型改为浮点型
        init_data['上证涨跌'] = init_data['上证涨跌'].astype('float')
        init_data['日增长率'] = init_data['日增长率'].astype('float')

        # 保存清洗好的数据
        self.cfig.clean_data_path = self.cfig.data_path + 'clean_' + self.cfig.stock_name + f'_{datetime.now().strftime("%Y%m%d")}.csv'
        init_data.to_csv(self.cfig.clean_data_path ,encoding='gbk',index=False)
        stock_date = init_data['净值日期'][:self.cfig.time_step]
        stock_date_ = stock_date.str[5:].tolist()
        stock_date_ = stock_date_[::-1]
        now_date = stock_date[0]
        for i in range(self.cfig.predict_day):
            date = (datetime.strptime(now_date, "%Y-%m-%d") + timedelta(days=i+1)).strftime("%m-%d")
            stock_date_.append(date)
        self.cfig.stock_date = stock_date_
        # 提取'单位净值','上证涨跌','涨跌'
        init_data = init_data[['单位净值','日增长率','上证涨跌']]

        # 返回init_data.values shape(len(init_data), len(init_data.columns.tolist()))
        # init_data.columns.tolist()为列名
        return init_data.values, init_data.columns.tolist()




    def get_train_and_valid_data(self):
        # 获取归一化后数据
        data = self.norm_data
        label = data[:,0]
        label = label.tolist()
        data = data.tolist()
        seq = []
        for i in range(len(data) - self.cfig.time_step):
             train_x = []
             train_y = []
             for j in range(i, i+self.cfig.time_step): 
                 x = []               
                 for c in range(len(self.cfig.feature_columns)):
                     x.append(data[j][c])
                 train_x.append(x)
             train_y.append(label[i + self.cfig.time_step])
             train_x = torch.FloatTensor(train_x)
             train_y = torch.FloatTensor(train_y).view(-1)
             seq.append((train_x, train_y))
        
        train_data_ = seq[0: int(len(seq)*self.cfig.train_data_rate)]
        self.cfig.seq = seq
        # self.cfig.test_data = data[int(len(seq)*self.cfig.train_data_rate):]

        # if test_len == 0:
            # test_data = seq[-self.cfig.batch_size:]
        train_data,valid_data = train_test_split(train_data_, test_size=self.cfig.valid_data_rate,
                                                                random_state=self.cfig.random_seed,
                                                                shuffle=self.cfig.shuffle_train_data)
        train_len = int(len(train_data) / self.cfig.batch_size) * self.cfig.batch_size
        test_len =  int(len(valid_data) / self.cfig.batch_size) * self.cfig.batch_size
        if test_len !=0:
            train_data, valid_data = train_data[:train_len],valid_data[:test_len]
        else:
            train_data, valid_data = train_data[:train_len],train_data_[-self.cfig.batch_size:]
        return train_data, valid_data
            


    def get_test_data(self):
        # 获取测试集数据
        feature_data = self.cfig.test_data.tolist()
        feature_data = torch.tensor(feature_data)
        sample_interval = min(feature_data.shape[0], self.cfig.time_step)     # 防止time_step大于测试集数量  feature_data.shape (28, 1)  sample_interval=28
        if sample_interval != self.cfig.time_step:
            self.cfig.time_step = sample_interval
        self.start_num_in_test = feature_data.shape[0] % sample_interval  # 这些天的数据不够一个sample_interval 不够一个sample_interval部分的天数
        # time_step_size = feature_data.shape[0] // sample_interval
        # self.cfig.batch_size = time_step_size
        feature_ = feature_data[self.start_num_in_test:]

        test_x = [feature_[i : i+sample_interval]for i in range(len(feature_)-sample_interval+1)]
        # self.label_y = [feature_data[:,0][self.start_num_in_test+(i+1)*sample_interval]for i in range(time_step_size)]

        self.cfig.batch_size = 1
        return test_x  

        
    def main(self):
        
        try:
            # 设置随机种子， 保证可复观
            np.random.seed(self.cfig.random_seed)
            

            if self.cfig.do_train:
                
                if os.path.exists(self.cfig.model_save_path + self.cfig.frame+'_'+self.cfig.stock_name+'_' +self.cfig.model_name):
                    res = input(f'{self.cfig.model_name} 训练文件已存在，是否重新训练(Y/N): \n')
                    if res.upper() == 'N':
                        logger.info('正在进行预测请稍等...')
                        train_loss = np.load(f'{self.cfig.train_loss_path}.npy').tolist()
                        valid_loss = np.load(f'{self.cfig.valid_loss_path}.npy').tolist()
                        valid_ac = np.load(f'{self.cfig.valid_ac_path}.npy').tolist()
                    elif res.upper() == 'Y':
                        train_data, test_data = self.get_train_and_valid_data()
                        train_loss, valid_loss, valid_ac=train(self.cfig, logger, [train_data,test_data])
                    else:
                        raise RuntimeError('输入有误，停止测试！')
                else:
                    train_data, test_data = self.get_train_and_valid_data()
                    train_loss, valid_loss, valid_ac=train(self.cfig, logger, [train_data, test_data])
            if self.cfig.do_predict:
                test_X = self.get_test_data()
                pred_result, label_Y = predict(self.cfig, logger, test_X)
                label_Y = label_Y[-self.cfig.time_step:]
                
                # draw()
            plt.figure(figsize=(8,8), dpi=80)
            plt.figure(1)
            ax1 = plt.subplot(211)
            plt.plot(range(len(train_loss)), train_loss, 'b', label='Training Loss')
            plt.plot(range(len(valid_loss)), valid_loss, 'r', label='Validation Loss')
            plt.plot(range(len(valid_ac)), valid_ac, 'g', label='Validation ACC')
            plt.title('Single Step Training and validation loss')
            plt.legend(loc='upper right')

            
            ax2 = plt.subplot(212)
            
            time_step = len(label_Y) + self.cfig.predict_day
            plt.xticks(range(time_step),self.cfig.stock_date,rotation=70)
            plt.plot(label_Y, 'b', label='Historical net worth')
            plt.plot(range(len(label_Y),time_step),pred_result, 'rx', markersize=10, label='Model Prediction')
            plt.title('Net Worth Forecast')
            plt.legend(loc='upper right')
            if self.cfig.do_figure_save:
                plt.savefig(self.cfig.figure_save_path+f'predict_{self.cfig.stock_name}_with_{self.cfig.frame}.png')
            plt.show()

        except Exception:
            logger.error("Run Error", exc_info=True)
        
        
        



if __name__ == '__main__':
    stock = Stock()
    cifg = stock.cfig
    logger = cifg.load_logger()
    stock.get_url()
    stock.get_data(input('请输入基金简称：'))
    datas = Data(cifg)
    datas.main()

