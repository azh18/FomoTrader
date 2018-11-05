import h5py
import pandas as pd
import numpy as np
import copy
import os
import sys
import operator

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# Change the working directory to your strategy folder.
# You should change this directory below on your own computer accordingly.
matplotlib.interactive(False)
current_dir = os.getcwd()
working_folder = current_dir + '/mytrade_avg'


# Write down your file paths for format 1 and format 2
# Note: You can test your strategy on different periods. Try to make your strategy profitable stably.
format1_dir = None
format2_dir = None
data_num = 3
if data_num == 1:
    # one laterial sudden drop
    format1_dir = current_dir + '/data/data_format1_20181007_20181014.h5'
    format2_dir = current_dir + '/data/data_format2_20181007_20181014.h5'
elif data_num == 2:
    # one laterial sudden up
    format1_dir = current_dir + '/data/data_format1_20181014_20181021.h5'
    format2_dir = current_dir + '/data/data_format2_20181014_20181021.h5'
elif data_num == 3:
    # one laterial sudden drop
    format1_dir = current_dir + '/data/data_format1_20180901_20180909.h5'
    format2_dir = current_dir + '/data/data_format2_20180901_20180909.h5'
elif data_num == 4:
    format1_dir = current_dir + '/data/data_format1_201807.h5'
    format2_dir = current_dir + '/data/data_format2_201807.h5'
elif data_num == 5:
    format1_dir = current_dir + '/data/data_format1_20181021_20181028.h5'
    format2_dir = current_dir + '/data/data_format2_20181021_20181028.h5'
elif data_num == 6:
    format1_dir = current_dir + '/data/data_format1_20181028_20181104.h5'
    format2_dir = current_dir + '/data/data_format2_20181028_20181104.h5'
# 

# The following code is for backtesting. DO NOT change it unless you want further exploration beyond the course project.
# import your handle_bar function
sys.path.append(working_folder)

# Run the main function in your demo.py to get your model and initial setup ready (if there is any)
os.chdir(working_folder)
os.system('python strategy.py')

from strategy import handle_bar
# from strategy import handle_bar


# Class of memory for data storage
class memory:
    def __init__(self):
        pass


class backTest:
    def __init__(self):
        # Initialize strategy memory with None. New memory will be updated every minute in backtest
        self.memory = memory()
        
        # Initial setting of backtest
        self.init_cash = 100000.
        self.cash_balance_lower_limit = 10000.
        self.commissionRatio = 0.0005
        
        # Data path
        self.data_format1_path = format1_dir
        self.data_format2_path = format2_dir
        
        # You can adjust the path variables below to train and test your own model
        self.train_data_path = ''
        self.test_data_path = ''
    
    def pnl_analyze(self, strategyDetail):
        balance = strategyDetail.total_balance
        balance_hourly = balance.resample("H").last()
        ret_hourly = balance_hourly.pct_change()
        ret_hourly[0] = balance_hourly[0] / self.init_cash - 1
        ret_hourly.fillna(0, inplace=True)

        balance_ratio = balance_hourly / 100000.
        price = strategyDetail["BTC-USD_price"]
        price_hourly = price.resample("H").last()
        price_first = price[0]
        price_ratio = price_hourly / price_first

        pos = strategyDetail['BTC-USD']

        balance_daily = balance.resample("D").last()
        ret_daily = balance_daily.pct_change()
        ret_daily[0] = balance_daily[0] / self.init_cash - 1
        ret_daily.fillna(0, inplace=True)

        total_ret = balance[-1] / balance[0] - 1
        daily_ret = ret_daily.mean()
        sharpe_ratio = np.sqrt(365) * ret_daily.mean() / ret_daily.std()
        max_drawdown = (balance / balance.cummax() - 1).min()

        print("Total Return: ", total_ret)
        print("Average Daily Return: ", daily_ret)
        print("Sharpe Ratio: ", sharpe_ratio)
        print("Maximum Drawdown: ", max_drawdown)


        # balance_hourly.plot(figsize=(12, 3), title='Balance Curve', grid=True)
        # plt.ioff()
        # matplotlib.pyplot.show(block=True)

        plt.subplot(2, 1, 1)
        balance_ratio.plot(figsize=(24,6), title='Balance/Price Curve', grid=True)
        price_ratio.plot(figsize=(24,6), title='Balance/Price Curve', grid=True)
        plt.ioff()

        #
        #
        plt.subplot(2, 1, 2)

        pos.plot(figsize=(24,6), title='position Curve', grid=True)
        plt.ioff()
        matplotlib.pyplot.show(block = True)

        # for i in range(0, 4):
        #     plt.subplot(4, 1, i+1)
        #     minute_prices = self.memory.timed_average_prices[0][i]
        #     xaxis = np.linspace(1, len(minute_prices), len(minute_prices))
        #     plt.plot(xaxis, minute_prices)
        # plt.show()

        watch_back_window = 35
        rsi_lookback_window = 15
        macd_lookback_window = 34
        bbands_lookback_window = 15
        stoch_lookback_window = 10
        atr_lookback_window = 15
        data_time_interval = [1, 5, 60]

        rows = 10
        cnt = 1
        plt.subplot(rows, 1, cnt)
        minute_prices = self.memory.timed_average_prices[1][1]
        xaxis = np.linspace(1, len(minute_prices), len(minute_prices))
        plt.plot(xaxis, minute_prices)


        cnt+=1
        plt.subplot(rows, 1, cnt)
        values = self.memory.timed_macds[2][1]
        zeros = np.zeros(macd_lookback_window)
        values = np.concatenate((zeros, values), axis=0)
        xaxis = np.linspace(1, len(values), len(values))
        plt.plot(xaxis, values)

        cnt +=1
        plt.subplot(rows, 1, cnt)
        values = self.memory.timed_macds[1][1]
        zeros = np.zeros(macd_lookback_window)
        values = np.concatenate((zeros, values), axis=0)
        xaxis = np.linspace(1, len(values), len(values))
        plt.plot(xaxis, values)

        cnt+=1
        plt.subplot(rows, 1, cnt)
        values = self.memory.timed_rsi[1][1]
        zeros = np.zeros(rsi_lookback_window)
        values = np.concatenate((zeros, values), axis=0)
        xaxis = np.linspace(1, len(values), len(values))
        plt.plot(xaxis, values)

        cnt+=1
        plt.subplot(rows, 1, cnt)
        values = self.memory.timed_rsi[2][1]
        zeros = np.zeros(rsi_lookback_window )
        values = np.concatenate((zeros, values), axis=0)
        xaxis = np.linspace(1, len(values), len(values))
        plt.plot(xaxis, values)

        cnt+=1
        plt.subplot(rows, 1, cnt)
        values = self.memory.timed_stochds[1][1]
        zeros = np.zeros(stoch_lookback_window)
        values = np.concatenate((zeros, values), axis=0)
        xaxis = np.linspace(1, len(values), len(values))
        plt.plot(xaxis, values)

        cnt+=1
        plt.subplot(rows, 1, cnt)
        values = self.memory.timed_stochks[1][1]
        zeros = np.zeros(stoch_lookback_window)
        values = np.concatenate((zeros, values), axis=0)
        xaxis = np.linspace(1, len(values), len(values))
        plt.plot(xaxis, values)

        cnt+=1
        plt.subplot(rows, 1, cnt)
        A = self.memory.timed_stochks[1][1]
        B = self.memory.timed_stochds[1][1]

        def cmp(x1, x2):
            if x1 >90 and x2 > 90:
                return 1
            elif x1 < 10 and x2 < 10:
                return -1
            return 0
        C = list(map(cmp, A, B))
        values =  C
        zeros = np.zeros(stoch_lookback_window)
        values = np.concatenate((zeros, values), axis=0)
        xaxis = np.linspace(1, len(values), len(values))
        plt.plot(xaxis, values)

        plt.grid()

        matplotlib.pyplot.show()
        # input()



        # Draw bbands graph
        # cnt = 0
        # time_index = 1
        #
        # minute_prices = self.memory.timed_average_prices[time_index][1]
        # minute_prices = minute_prices[bbands_lookback_window + 1:]
        # xaxis = np.linspace(1, len(minute_prices), len(minute_prices))
        # plt.plot(xaxis, minute_prices)
        #
        # values = self.memory.timed_lowerbs[time_index][1]
        # # zeros = np.zeros(bbands_lookback_window)
        # # values = np.concatenate((zeros, values), axis=0)
        # xaxis = np.linspace(1, len(values), len(values))
        # plt.plot(xaxis, values)
        #
        # values = self.memory.timed_higherbs[time_index][1]
        # # zeros = np.zeros(bbands_lookback_window)
        # # values = np.concatenate((zeros, values), axis=0)
        # xaxis = np.linspace(1, len(values), len(values))
        # plt.plot(xaxis, values)
        #
        # plt.show()

        #
        # values = self.memory.timed_lowerbs[2][1]
        # zeros = np.zeros(bbands_lookback_window)
        # values = np.concatenate((zeros, values), axis=0)
        # xaxis = np.linspace(1, len(values), len(values))
        # plt.plot(xaxis, values)
        #
        # values = self.memory.timed_higherbs[2][1]
        # zeros = np.zeros(bbands_lookback_window)
        # values = np.concatenate((zeros, values), axis=0)
        # xaxis = np.linspace(1, len(values), len(values))
        # plt.plot(xaxis, values)


        pass

    def backTest(self):

        ''' Function that used to do back-testing based on the strategy you give
        Params: None
        
        Notes: this back-test function will move on minute bar and generate your 
        strategy detail dataframe by using the position vectors your strategy gives
        each minute
        '''

        format1 = h5py.File(self.data_format1_path, mode='r')
        format2 = h5py.File(self.data_format2_path, mode='r')
        assets = list(format1.keys())
        keys = list(format2.keys())

        # limit = 1000
        # keys = keys[800:2000]

        for i in range(len(keys)):
            data_cur_min = format2[keys[i]][:]
            # 1. initialization
            if i == 0:
                total_balance = self.init_cash
                average_price_old = np.mean(data_cur_min[:,:4], axis=1)
                position_old = np.repeat(0., 4)
                position_new = np.repeat(0., 4)
                details = list()
                stop_signal = False

            # 2. calculate position & cash/crypto/total balance & transaction cost etc.
            position_change = position_new - position_old
            mask = np.abs(position_change) > .25*data_cur_min[:,4]
            position_change[mask] = (.25*data_cur_min[:,4]*np.sign(position_change))[mask]
            position_new = position_old + position_change
            average_price = np.mean(data_cur_min[:, :4], axis=1)
            transaction_cost = np.sum(np.abs(position_change)*average_price*self.commissionRatio)
            revenue = np.sum(position_old*(average_price - average_price_old)) - transaction_cost
            crypto_balance = np.sum(np.abs(position_new*average_price))
            total_balance = total_balance + revenue
            cash_balance = total_balance - crypto_balance
            detail = np.append(position_new, list(average_price) + [cash_balance, crypto_balance, revenue, total_balance, transaction_cost])
            details.append(copy.deepcopy(detail))


            position_old = copy.deepcopy(position_new)
            average_price_old = copy.deepcopy(average_price)

            # 3. check special cases
            # if cash balance is less than lower limit, the program will stop all trading actions in the future
            if (cash_balance < self.cash_balance_lower_limit) and (stop_signal == False):
                stop_signal = True
                print("Current cash balance is lower than", self.cash_balance_lower_limit)
                print("Your strategy is forced to stop")
                print("System will soon close all your positions (long and short) on crypto currencies")

            if stop_signal:
                position_new = np.repeat(0., 4)
                if '09:30:00' in keys[i]:
                    print(keys[i][:10])
                continue

            # Update position and memory
            [position_new, self.memory] = handle_bar(i,
                                                     keys[i],
                                                     data_cur_min,
                                                     self.init_cash,
                                                     self.commissionRatio,
                                                     cash_balance,
                                                     crypto_balance,
                                                     total_balance,
                                                     position_new,
                                                     self.memory)

            # Update position and timer
            if '09:30:00' in keys[i]:
                print(keys[i][:10])


        detailCol = assets + [s+'_price' for s in assets] + ["cash_balance", "crypto_balance", "revenue", "total_balance", "transaction_cost"]
        detailsDF = pd.DataFrame(details, index=pd.to_datetime(keys), columns=detailCol)


        format1.close()
        format2.close()
        return detailsDF


if __name__ == '__main__':
    ''' You can check the details of your strategy and do your own analyze by viewing 
    the strategyDetail dataframe
    '''
    bt = backTest()
    strategyDetail = bt.backTest()
    strategyDetail.to_csv(working_folder+"/backtest_details.csv")  # output backtest details to your working folder
    bt.pnl_analyze(strategyDetail)  # print performance summary, plot balance curve

