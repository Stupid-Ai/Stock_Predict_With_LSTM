# -*- coding: utf-8 -*-


import sys
import os
import time
import logging
import pandas as pd
from selenium import webdriver
from datetime import datetime
from selenium.common import exceptions as ex
from selenium.webdriver.common.by import By

from path_config import *



class Stock:
    def __init__(self) -> None:
        # 无浏览器界面化
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        self.brower = webdriver.Chrome(options=options)
        self.cfig = Config()


    # 获取所有基金信息
    def get_url(self):
        # 判断url.csv文件是否存在
        if os.path.exists(self.cfig.url_path):
            # 若存在则直接返回
            return

        # url.csv不存在则执行以下代码
        url = 'https://fund.eastmoney.com/' # 天天基金网首页
        self.brower.get(url)
        # 定位到‘点击查询全部基金净值’
        data = self.brower.find_element(By.XPATH,'//*[@id="jjjz"]/div[3]/table/tfoot/tr/td/a')
        # 获取全部基金网址
        data_all = data.get_attribute('href')

        time.sleep(2)
        # 跳转到全部基金
        self.brower.get(data_all)

        data_all_url = {}

        # 打开url.csv文件，没有会直接创建
        fp = open(self.cfig.url_path, 'w')

        '''
        因为每次切换页都只是刷新表单，网址不变，只能通过点击事件来更新表单类容
        所以通过循环所有页来获取所有的信息：抓取当前表单所有类容后通过点击事件点击下一页
        '''
        # 确定所有页数a，for循环a次
        for page in range(int(self.brower.find_element(By.XPATH,'//*[@id="pager"]/span[9]').text[1:-1])):
            try:
                
                # find_element寻找第一个 find_elements寻找所有的
                tags = self.brower.find_elements(By.XPATH,'//*[@id="oTable"]/tbody/tr')
                for i in tags:
                    name = i.find_element(By.XPATH,'./td[5]/nobr/a[1]').get_attribute('title') # 获取基金简称
                    num = i.find_element(By.XPATH,'./td[4]').text # 获取基金id
                    if name not in data_all_url.keys():
                        fp.write(f'{name},http://fundf10.eastmoney.com/jjjz_{num}.html\n') # 将获取到的基金简称和连接写入url.csv文件中
                        data_all_url.setdefault(name,f'http://fundf10.eastmoney.com/jjjz_{num}.html') # 将获取到的基金简称和连接添加到date_all_url字典中
                        logging.info(f'{name} -->  {data_all_url[name]}') # 在控制台显示写入信息提示
                
                if (page+1)/35==0:
                    time.sleep(60)
                # 表单循环结束后点击下一页
                self.brower.find_element(By.XPATH,'//*[@id="pager"]/span[8]').click()
                fp.flush()
                time.sleep(3)
                logging.info('-'*30+f'以获取{page+1}页'+'-'*30)
            except ex.WebDriverException as e:
                self.brower.find_element(By.XPATH,'//*[@id="pager"]/span[8]').click()
                time.sleep(3)
                logging.warning(str(e))
                
        self.brower.close()
        fp.close()


    # 获取上证指数等额外数据
    def overall_data(self,date,df):
        '''
        date: 基金数据记录的开始日期
        df: 基金数据的DataFrame数据

        '''
        try:
            # 获取现在的时间
            date_now = datetime.now().strftime("%Y-%m-%d")
            # 获取上证指数网址
            self.brower.get('https://q.stock.sohu.com/zs/000001/lshq.shtml')
            # 清空起始日期内的类容
            self.brower.find_element(By.XPATH,'//*[@id="BIZ_lshq_sd"]').clear()
            # 在起始日期内输入基金数据的起始日期
            self.brower.find_element(By.XPATH,'//*[@id="BIZ_lshq_sd"]').send_keys(date)
            # 清空结束日期内的类容
            self.brower.find_element(By.XPATH,'//*[@id="BIZ_lshq_ed"]').clear()
            # 在结束日期内输入现在的日期
            self.brower.find_element(By.XPATH,'//*[@id="BIZ_lshq_ed"]').send_keys(date_now)
            # 点击查询
            self.brower.find_element(By.XPATH,'/html/body/div[4]/div[2]/div[2]/div[2]/div/div[1]/form/input[7]').click()
            # 获取表单div的height值
            div_size = self.brower.find_element(By.XPATH,'//*[@id="BIZ_hq_historySearch"]/tbody').size
            div_h = div_size['height']
            # 将网页拉到底部 **
            self.brower.execute_script(f'scroll(0,{div_h})')
            # 获取数据表单
            form = self.brower.find_element(By.XPATH,'//*[@id="BIZ_hq_historySearch"]/tbody')
            time.sleep(5)
        except ex.WebDriverException as e:
            logging.warning(str(e))
        # 加载表单内容 ** 缺少这一步表单只会加载前100行数据
        form = form.text
        # 将获取到的文本从'\n'处分割成列表 开始遍历
        for i in form.split('\n'):
            # 获取日期 将获取到的文本按空格分割成列表，提取第一个元素
            data_t = i.split(' ')[0]
            # 获取涨跌值 将获取到的文本按空格分割成列表，提取第五个元素
            data_u = i.split(' ')[4]
            # 判断获取的日期与基金数据表内的‘净值日期’是否相等，相等则将获取的涨跌值填入到对应日期的‘上证涨跌’列
            df.loc[df['净值日期']==data_t,'上证涨跌'] = data_u
        
        self.brower.close()

        # 返回基金数据表
        return df
        
        

    # 获取查询数据
    def get_data(self,stk_nm):

        '''
        stk_nm: 基金简称
        '''
        self.cfig.stock_name = stk_nm

        # 防止重复爬取
        datapath = self.cfig.data_path + stk_nm + f'_{datetime.now().strftime("%Y%m%d")}.csv'
        if os.path.exists(datapath):
            # 设置读取文件的地址
            self.cfig.read_data_path = datapath
            return

        # 打开url.csv文件，编码格式是gbk 
        fp = open(self.cfig.url_path,encoding='gbk')
        # 使用pandas打开url.csv文件，并添加列名方便查询输入的基金是否被录入
        all_stock_url = pd.read_csv(fp,names=['基金简称','连接'])
        
        # 判断输入的基金是否被录入
        if stk_nm in list(all_stock_url['基金简称']):
            # 满足条件则获取输入基金对应的网址
            stk_url = list(all_stock_url.loc[all_stock_url['基金简称']==stk_nm]['连接'])[0]
        else:
            # 不满足提示；网址暂未收录
             raise Exception(f'网址暂未收录 {stk_url}')  
        logging.info(stk_url)

        
        # 获取网址
        self.brower.get(stk_url)
        next_page = 1

        # 获取总页数
        all_page = int(self.brower.find_element(By.XPATH, '//*[@id="pagebar"]/div[1]/label[7]').text)
        # 获取表单
        form = self.brower.find_element(By.XPATH,'//*[@id="jztable"]/table/thead')
        # 获取列名
        line_menu = [i for i in form.text.split(' ')][:6]
        

        # 数据处理
        df = pd.DataFrame()
        for i in line_menu:
            df[i] = pd.Series(dtype='float64')
        
        # 循环全部页，提取每页表单中的全部数据
        while next_page <= all_page:

            if next_page == 1:
                logging.info(f'正在获取  {stk_nm}  数据，请稍等...' )

            # 获取表单除列名以外所有的数据
            form = self.brower.find_elements(By.XPATH,'//div[@class="txt_in"]/div[2]/div/div[2]//tbody/tr')
            # tr = [i for j in form.text.split('\n') for i in j.split(' ')]
            # 循环提取表单中的每一行
            for tr in form:
                # 新建一个空列表用来存放提取出来的每一行的数据
                line_data = []
                # 循环提取一行中的所有数据
                for i in tr.find_elements(By.XPATH,'./td')[:6]:
                    # 将提取到的数据放到上面的列表中
                    line_data.append(i.text)
                    # 将提取到的这一行的数据 除最后一列 添加到数据表中
                if len(line_data) == len(line_menu):
                    df = df.append(pd.Series(line_data,index=line_menu), ignore_index=True)
            
            # 提取完一页后next_page + 1
            next_page += 1
            # 判断是否是最后一页
            if next_page <= all_page:
                # 如果不是最后一页则点击下一页
                self.brower.find_element(By.XPATH, f'//*[@id="pagebar"]/div[1]/label[@value="{next_page}"]').click()

            time.sleep(3)
            
            
            #  在控制面板显示进度 
            if next_page % 10 == 0 or next_page / all_page == 1:
                    logging.info(f'正在获取  {stk_nm}  数据，已完成:' + '▇' *
                     int(next_page/all_page*10) + f'{round(next_page / all_page * 100, 2)}' + '%')


        # self.brower.close()

        
        # 获取基金数据的开始日期
        date = df['净值日期'][-1:].values[0]
        # 添加‘上证涨跌’列
        df['上证涨跌'] = ''
        # 获取上证信息
        df = self.overall_data(date,df)

        # 将基金数据保存
        # datapath = self.cfig.data_path + stk_nm + f'_{datetime.now().strftime("%Y%m%d")}.csv'
        df.to_csv(datapath,encoding='gbk',index=False)


        # 设置读取文件的地址
        self.cfig.read_data_path = datapath
        # return self.cfig.read_data_path
        

if __name__ == '__main__':
    # cfig = Config()
    # cfig.load_logger()
    # s = Stock()
    # # s.get_url()
    # s.get_data('广发制造业精选混合A')
    # # s.overall_data('20210101')
    pass