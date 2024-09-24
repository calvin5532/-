import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import math
import re

'''
sell all
max_upper_bound(max(mean + 1.5*std,0))
sell half
upper_bound(max(mean +　0.5*std,0))
nothing
lower_bound(mean - 0.5*std)
buy half
max_lower_bound(mean - 1.5*std)
buy all

'''
per = 20
days = 10
period = "10y"
u_b = 0.3
l_b = 0
m_u_b = 0.8
m_l_b = -0.4
start_money = 10000


# perdict 剩下的資料
def predict(per,class_obj):
    predict_value = []
    predict_mean = []
    predict_data = class_obj.stock_data["Open"][per*class_obj.days:-1]
    if not len(predict_data):
        print("training period too long")
    else:
        tmp = 0
        stop = False
        while not stop:
            if tmp+class_obj.days >= len(predict_data):
                try:
                    tmp_list = calc(predict_data[tmp:-1])
                except:
                    pass
                stop = True
                predict_mean.append(tmp_list[1])
            else:
                tmp_list = calc(predict_data[tmp:tmp + class_obj.days])
                tmp += class_obj.days
                predict_mean.append(tmp_list[1])
                if tmp_list[0] > m_u_b:
                    class_obj.sell(class_obj.stock_amount,predict_data[tmp])
                elif m_u_b > tmp_list[0] > u_b:
                    class_obj.sell(class_obj.stock_amount/2, predict_data[tmp])
                elif m_l_b < tmp_list[0] < l_b:
                    class_obj.dump(class_obj.money / 2, predict_data[tmp])
                elif tmp_list[0] < m_l_b:
                    class_obj.dump(class_obj.stock_amount, predict_data[tmp])
                predict_value.append(class_obj.money+class_obj.stock_amount*predict_data[tmp])
        predict_x = np.arange(0,len(predict_value),1)
        predict_mean_x = np.arange(0,len(predict_mean),1)
        plt.figure()
        plt.title(f"{class_obj.stock_name} predict result for training {per} periods")
        plt.xlabel("periods")
        plt.ylabel("total value")
        plt.plot(predict_x,predict_value,linewidth=1)
        plt.grid()
        plt.figure()
        plt.title(f"{class_obj.stock_name} mean value")
        plt.xlabel("periods")
        plt.ylabel("mean value")
        plt.plot(predict_mean_x,predict_mean,linewidth=1)
        plt.grid()

# calc skew and mean value
def calc(data):
    n = len(data)
    niu = 0.0
    niu2 = 0.0
    niu3 = 0.0
    for a in data:
        niu += a
        niu2 += a**2
        niu3 += a**3
    niu /= n
    niu2 /= n
    niu3 /= n
    sigma = math.sqrt(niu2 - niu*niu)
    return [(niu3 - 3 * niu * sigma ** 2 - niu ** 3) / (sigma ** 3),niu]


class AI:
    def __init__(self,stock_name,period):
        self.stock_name = stock_name
        self.stock = yf.Ticker(stock_name)
        try:
            self.stock_data = self.stock.history(start = period[0],end =period[1])
        except:
            self.stock_data = self.stock.history(period)
        self.days = days
        self.money = start_money
        self.stock_amount = 0
        self.period = period
        try:
            self.year = int(re.split("y",period)[0])
        except:
            print("sdadaaaa")
            self.year = len(self.stock_data["Close"]) / 360
        self.period = period
        self.re = 0
        self.year_re = 0

    def dump(self,money,open_money):
        self.stock_amount += money/open_money
        self.money -= money

    def sell(self,stock_amount,open_money):
        self.money += stock_amount * open_money
        self.stock_amount -= stock_amount

    def train(self,times):
        draw_y = []
        draw_yy = []
        draw_yyy = []
        tmp = 0
        tmp_times = 0
        stock_arr = []
        train_data = np.array(self.stock_data["Open"][0:self.days*times+1])
        while tmp + self.days <= len(train_data) and tmp_times < times:
            tmp_arr = calc(train_data[tmp:tmp + self.days])
            stock_arr.append(tmp_arr[0])
            t_stock_arr.append(tmp_arr[0])
            draw_yy.append(tmp_arr[0])
            draw_yyy.append(tmp_arr[1])
            tmp += self.days
            tmp_times += 1
        for i in range(len(stock_arr)):
            if m_l_b > stock_arr[i]:
                self.dump(money=self.money,open_money=train_data[self.days*(i)])
            elif m_u_b > stock_arr[i] > u_b:
                self.sell(stock_amount=self.stock_amount / 2, open_money=train_data[self.days * (i)])
            elif stock_arr[i] > m_u_b:
                self.sell(stock_amount=self.stock_amount, open_money=train_data[self.days*(i)])
            elif m_l_b < stock_arr[i] < l_b:
                self.dump(money=self.money/2, open_money=train_data[self.days * (i)])
            draw_y.append(self.money+self.stock_amount*train_data[self.days*(i+1)])
            std = np.std(stock_arr[0:i+1])
            mean = np.mean(stock_arr[0:i+1])
            globals()["m_l_b"] = mean - 1.5*std
            globals()["m_u_b"] = max(mean + 1.5*std, 0)
            globals()["l_b"] = mean - 0.5*std
            globals()["u_b"] = max(mean + 0.5*std, 0)

        return [draw_y,draw_yy,draw_yyy]

    def cale_re(self,close_money):
        self.money = self.money + self.stock_amount * close_money
        self.re = (self.money - start_money) / start_money
        self.year_re = (1+self.re)**(1/self.year)-1
        print(f"{self.stock_name}_結算金額:", self.money, "元")
        print(f"{self.stock_name}_投資報酬率:", self.re * 100, "%")
        print(f"{self.stock_name}_年化報酬率:", self.year_re * 100, "%")

    def draw(self,draw_y,draw_yy,draw_yyy):
        normal_y = []
        normal_x = []
        draw_normal_y = []
        tmp = 0
        for i in range(len(self.stock_data["Close"])-1):
            normal_y.append(self.stock_data["Close"][i] - self.stock_data["Close"][i+1])
        sort_n_y = sorted(normal_y)
        for i in sort_n_y:
            if not normal_x:
                normal_x.append(round(i, 1))
                draw_normal_y.append(1)
            elif normal_x[tmp] == round(i, 1):
                draw_normal_y[tmp] += 1
            else:
                tmp += 1
                normal_x.append(round(i, 1))
                draw_normal_y.append(1)
        plt.figure()
        plt.grid()
        plt.title(f"{self.stock_name} Quote change sort and round")
        plt.xlabel("Quote change value")
        plt.ylabel("times")
        plt.plot(normal_x,draw_normal_y,linewidth=1)
        x = np.arange(0, len(draw_y), 1)
        xx = np.arange(0, len(draw_yy), 1)
        xxx = np.arange(0, len(draw_yyy), 1)
        plt.figure()
        plt.xlabel(f"time(per {self.days} days)")
        plt.ylabel("total value")
        plt.title(f"{self.stock_name} training time total value")
        plt.plot(x, draw_y, linewidth=1)
        plt.grid()
        plt.figure()
        plt.xlabel(f"time(per {self.days} days)")
        plt.ylabel("Skewness value")
        plt.title(f"{self.stock_name} training time Skewness value")
        plt.plot(xx, draw_yy, linewidth=1)
        plt.grid()
        plt.figure()
        plt.xlabel(f"time(per {self.days} days)")
        plt.ylabel("mean value per period")
        plt.title(f"{self.stock_name} training time mean value")
        plt.plot(xxx, draw_yyy, linewidth=1)
        plt.grid()

t_stock_arr = []
# tsmc
tsmc = AI('2330.TW',period)
tmp_x = 0
tmp_list = tsmc.train(per)
tsmc.draw(tmp_list[0],tmp_list[1],tmp_list[2])
predict(per, tsmc)
tsmc.cale_re(tsmc.stock_data["Close"][-1])

#apple
apple = AI("AAPL",period)
tmp_list = apple.train(per)
apple.draw(tmp_list[0],tmp_list[1],tmp_list[2])
predict(per, apple)
apple.cale_re(apple.stock_data["Close"][-1])

#yuanta
yuanta = AI("0050.TW",period)
tmp_list = yuanta.train(per)
yuanta.draw(tmp_list[0],tmp_list[1],tmp_list[2])
predict(per, yuanta)
yuanta.cale_re(yuanta.stock_data["Close"][-1])

#HPQ
hpq = AI("HPQ",period)
tmp_list = hpq.train(per)
hpq.draw(tmp_list[0],tmp_list[1],tmp_list[2])
predict(per, hpq)
hpq.cale_re(hpq.stock_data["Close"][-1])

# Dow Jones
dji = AI("^DJI",period)
tmp_list = dji.train(per)
dji.draw(tmp_list[0],tmp_list[1],tmp_list[2])
predict(per, dji)
dji.cale_re(dji.stock_data["Close"][-1])

'''
# TA training 
for i in range(10):
    test_tsmc = AI('2330.TW',period)
    test_tsmc.train(per*i)
    predict(per*i,test_tsmc)
    print(f"tsmc_第{i+1}週期訓練成果")
    test_tsmc.cale_re(test_tsmc.stock_data["Close"][-1])
    test_apple = AI('AAPL', period)
    test_apple.train(per * i)
    predict(per * i, test_apple)
    print(f"apple_第{i + 1}週期訓練成果")
    test_apple.cale_re(test_apple.stock_data["Close"][-1])
    test_yuanta = AI('0050.TW', period)
    test_yuanta.train(per * i)
    predict(per * i, test_yuanta)
    print(f"yuanta_第{i + 1}週期訓練成果")
    test_yuanta.cale_re(test_yuanta.stock_data["Close"][-1])
    test_hpq = AI('HPQ', period)
    test_hpq.train(per * i)
    predict(per * i, test_hpq)
    print(f"hqp_第{i + 1}週期訓練成果")
    test_hpq.cale_re(test_hpq.stock_data["Close"][-1])
'''

# test data
std = np.std(t_stock_arr)
mean = np.mean(t_stock_arr)
globals()["m_l_b"] = mean - 1.5*std
globals()["m_u_b"] = max(mean + 1.5*std, 0)
globals()["l_b"] = mean - 0.5*std
globals()["u_b"] = max(mean + 0.5*std, 0)
tlsa = AI("TLSA",period=['2017-01-01','2021-01-01'])
predict(0,tlsa)
tlsa.cale_re(tlsa.stock_data["Close"][-1])
gme = AI("GME",period=['2017-01-01','2021-01-01'])
predict(0,gme)
gme.cale_re(gme.stock_data["Close"][-1])
htc = AI("2498.TW",period=['2017-01-01','2021-01-01'])
predict(0,htc)
htc.cale_re(htc.stock_data["Close"][-1])
yuanki = AI("6443.TW",period=['2016-01-01','2020-01-01'])
predict(0,yuanki)
yuanki.cale_re(yuanki.stock_data["Close"][-1])
plt.show()