
from cProfile import label
from django import conf
from matplotlib import pyplot as plt
import torch
from torch.nn import Module, LSTM, Linear
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from path_config import *



class Net(Module):



    def __init__(self, config):
        self.config = config
        
        # 继承父类的属性    
        super(Net, self).__init__()
        # 设置lstm的输入参数
        self.lstm = LSTM(input_size=config.input_size, hidden_size=config.hidden_size,
                         num_layers=config.lstm_layers, batch_first=True, dropout=config.dropout_rate)
        # 这是一个全连接层
        self.linear = Linear(config.hidden_size, config.out_size)           
        

    def forward(self, x, hidden=None):
        # x = self.linear1(x)
        x = x.view(self.config.batch_size, self.config.time_step, self.config.input_size)
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(self.config.batch_size*self.config.time_step, self.config.hidden_size)
        linear_out = self.linear(lstm_out)
        pred = linear_out.view(self.config.batch_size, self.config.time_step,-1)
        pred = pred[:,-1,:]
        
        
        return pred, hidden


def train(config, logger, train_and_valid_data):
    if config.do_train_visualized:
        import visdom
        vis = visdom.Visdom(env='model_pytorch')
    
    train_data, valid_data = train_and_valid_data
    
    # Dataloader可自动生成可训练的batch数据 
    train_loader = DataLoader(train_data, batch_size=config.batch_size)
    valid_loader = DataLoader(valid_data, batch_size=config.batch_size)

    # 判断使用GPU还是CPU
    device = torch.device('cuda' if config.use_cuda and torch.cuda.is_available() else 'cpu')
    # 如果使用gpu训练， .to(device)会把模型/数据复制到gpu显存中
    model = Net(config).to(device)
    if config.add_train and os.path.exists(config.model_save_path + config.frame+'_'+config.stock_name+'_' +config.model_name):
        model.load_state_dict(torch.load(config.model_save_path + config.frame+'_'+config.stock_name+'_' +config.model_name))
    
    # 定义优化器和loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 正无穷
    valid_loss_min = float('inf')
    bad_epoch = 0
    global_step = 0

    train_loss = []
    valid_loss = []
    valid_ac = []
    for epoch in range(config.epoch):
        logger.info(f'Epoch  {epoch+1}/{config.epoch}')
        # 转换成训练模式
        model.train()

        train_loss_array = []
        
        hidden_train = None
        for _train_X, _train_Y in train_loader:
            _train_X, _train_Y = _train_X.to(device), _train_Y.to(device)
            
            # 这里走的就是向前计算forward函数
            pred_Y, hidden_train = model(_train_X, hidden_train)

            if not config.do_continue_train:
                hidden_train = None
            else:
                h_0, c_0 = hidden_train
                h_0.detach_(), c_0.detach_()
                hidden_train = (h_0, c_0)
            # 计算loss
            

            loss = criterion(pred_Y, _train_Y)
            # 训练前将梯度信息重置 0 防止重复计算
            optimizer.zero_grad()
            # 将loss反向传播
            loss.backward()
            # 用优化器更新参数
            optimizer.step()
            train_loss_array.append(loss.item())
            global_step += 1
            if config.do_train_visualized and global_step % 100 == 0:
                vis.line(X=np.array([global_step]), Y=np.array([loss.item()]), win='Train_Loss',
                update='append' if global_step > 0 else None, name='Train', opts=dict(showlegend=True))
            

        # 转换成测试模式
        model.eval()
        valid_loss_array = []
        hidden_valid = None
        for _valid_X, _valid_Y in valid_loader:
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            pred_Y, hidden_valid = model(_valid_X, hidden_valid)
            if not config.do_continue_train:
                hidden_valid = None
            else:
                h_0, c_0 = hidden_train
                h_0.detach_(), c_0.detach_()
                hidden_train = (h_0, c_0)
            
            # 验证过程只有向前计算，无反向传播
            
            loss = criterion(pred_Y, _valid_Y)
            valid_loss_array.append(loss.item())

        label = _valid_Y.squeeze()
        # 还原归一化后的数据
        label = label*config.std[0] + config.mean[0]
        label = label.detach().cpu().numpy()
        y = pred_Y.squeeze()
        y = y*config.std[0] + config.mean[0]
        y = y.detach().cpu().numpy()
        # 计算准确率
        err = np.diff(y,axis=0) * np.diff(label,axis=0)
        acc = sum(err > 0) * 1.0 / len(label)
        valid_ac.append(acc)


        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)
        train_loss.append(train_loss_cur)
        valid_loss.append(valid_loss_cur)
        logger.info(f'The train loss is {train_loss_cur}\tThe valid loss is {valid_loss_cur}')
        if config.do_train_visualized:
            vis.line(X=np.array([epoch]),Y=np.array([train_loss_cur]), win='Epoch_Loss',
                    update='append' if epoch > 0 else None, name='Train',opts=dict(showlegend=True))
            vis.line(X=np.array([epoch]),Y=np.array([valid_loss_cur]), win='Epoch_Loss',
                    update='append' if epoch > 0 else None, name='Eval',opts=dict(showlegend=True))

        # 以下是早停机制
        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            bad_epoch = 0
            torch.save(model.state_dict(), config.model_save_path + config.frame+'_'+config.stock_name+'_' +config.model_name)
        else:
            if config.patience == -1:
                continue
            bad_epoch += 1
            if bad_epoch >= config.patience:
                logger.info(f'The training stops early in epoch {epoch}')
                break
    
    np.save(f'{config.train_loss_path}', np.array(train_loss))
    np.save(f'{config.valid_loss_path}', np.array(valid_loss))
    np.save(f'{config.valid_ac_path}', np.array(valid_ac))
    
    return train_loss, valid_loss, valid_ac
    



def predict(config, logger,test_X):
    # 获取测试数据
    test_X = DataLoader(test_X, batch_size=1)
    # 加载模型
    device = torch.device('cuda' if config.use_cuda and torch.cuda.is_available() else 'cpu')
    model = Net(config).to(device)
    model.load_state_dict(torch.load(config.model_save_path + config.frame+'_'+config.stock_name+'_' +config.model_name))


    # 先定义一个ternsor保存结果
    result = torch.Tensor().to(device)
    label_h = torch.Tensor().to(device)
    # 预测过程
    model.eval()
    hidden_predict = None
    
    for _data in test_X:
        data_X = _data.to(device)
        
        pred_X, hidden_predict = model(data_X, hidden_predict)

        # 降维
        cur_pred = torch.squeeze(pred_X, dim=0)
        cur_pred = cur_pred * config.std[0] + config.mean[0]
        cur_label = data_X.squeeze()
        cur_label = torch.Tensor([cur_label.tolist()[-config.predict_day][0]]).to(device)
        cur_label = cur_label * config.std[0] + config.mean[0]
        label_h = torch.cat((label_h, cur_label), dim=0)
        # 拼接
        result = torch.cat((result, cur_pred), dim=0)

    result = result.detach().cpu().numpy()
    label_h = label_h.detach().cpu().numpy()
    label_,result_ = label_h[config.predict_day:],result[:-config.predict_day]
    label_p = label_[:-config.predict_day]
    label_n = label_[config.predict_day:]
    result_p = result_[:-config.predict_day]
    result_n = result_[config.predict_day:]
    label_updown = label_p - label_n
    result_updown = result_p - result_n
    label_updown = [1 if i < 0 else (-1 if i > 0 else 0)for i in label_updown]
    result_updown = [1 if i < 0 else (-1 if i > 0 else 0)for i in result_updown]

    err = np.array(label_updown) * np.array(result_updown)
    logger.info(f'Accuracy: {sum(err>0) * 1.0 / len(label_updown) * 100 :.2f}%')
    pred_day = result[-config.predict_day:]
    label_h = np.r_[label_h,pred_day]
    pred_day_ = label_h[-config.predict_day-1:]
    pred_day_p = pred_day_[:-1]
    pred_day_n = pred_day_[1:]
    pred_updown = pred_day_p - pred_day_n
    pred_updown = ['up' if i < 0 else('down' if i > 0 else '-')for i in pred_updown]
    logger.info(f"The predicted stock trend is: {pred_updown}")


    return cur_pred.item(), label_h[:-config.predict_day]


if __name__ == '__main__':
    pass