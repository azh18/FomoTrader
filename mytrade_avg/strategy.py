# Here you can
# 1. import necessary python packages for your strategy
# 2. Load your own facility files containing functions, trained models, extra data, etc for later use
# 3. Set some global constants
# Note:
# 1. You should put your facility files in the same folder as this strategy.py file
# 2. When load files, ALWAYS use relative path such as "data/facility.pickle"
# DO NOT use absolute path such as "C:/Users/Peter/Documents/project/data/facility.pickle"
import numpy as np
from copy import deepcopy
import  math
import talib
import logging
import sys
import pandas as pd
from pandas import Series
from talib import abstract

# from auxiliary import generate_bar, white_soider, black_craw  # auxiliary is a local py file containing some functions

my_cash_balance_lower_limit = 10000.  # Cutting-loss criterion
ALL_ASSET = [0, 1, 2, 3]
USE_ASSET_INDEX = [0, 1, 2, 3]

# USE_ASSET_INDEX = [1]
# buy_one = [0.01, 0.001, 0.02, 0.01]
# # buy_one = [0.1, 0.01, 0.2, 0.8]
# buy_factor = 2
# sell_factor = 2
# buy_one = list(map(lambda x: x * buy_factor, buy_one))
# sell_one = list(map(lambda x: x * sell_factor, buy_one))


buy_in_each_bitcoin = 0.1
buy_in_signal_dollar = 5
stop_loss_dollar = 10000
stop_profit_dollar = 10
borrow_in_signal_dollar = 5
borrow_each_bitcoin = 0.3

ticker_num = 4


watch_back_window = 35
rsi_lookback_window = 15
macd_lookback_window = 34
bbands_lookback_window = 15
stoch_lookback_window = 10
atr_lookback_window = 15

cols = ['counter', 'open', 'high', 'low', 'close', 'average', 'volume']

min_profit = 0.001


def get_avg_price(today_data):
    return np.mean(today_data[:, :4], axis=1)


def float_equal_int(float_num, int_num):
    return np.abs(float_num - int_num) < 0.000001


def d(x):
    logging.debug(x)


def get_volume(today_data):
    return today_data[:, 4]


def get_open(today_data):
    return today_data[:, 3]


def get_close(today_data):
    return today_data[:, 0]


def get_high(today_data):
    return today_data[:, 1]


def get_low(today_data):
    return today_data[:, 2]


def get_row_data(data):
    # return today data of all kind of tickers
    highs = get_high(data)
    lows = get_low(data)
    opens = get_open(data)
    closes = get_close(data)
    averages = get_avg_price(data)
    volumes = get_volume(data)
    row_data = pd.DataFrame([], columns=cols[1:])
    for i in range(0, ticker_num):
        row = [opens[i], highs[i], lows[i], closes[i], averages[i], volumes[i]]
        row_data.loc[i] = row
    # row_data = [opens, highs, lows, closes, averages, volumes]
    return row_data


def get_new_row(row_data, counter, i):
    # data: today data
    # counter: counter in hanlde bar para
    # i: index of asset

    row = [counter]
    # logging.debug(row)
    row.extend(list(row_data.iloc[i]))
    # logging.debug(row)
    return row


def fishers_inverse(series, smoothing = 0):
    """ Does a smoothed fishers inverse transformation.
        Can be used with any oscillator that goes from 0 to 100 like RSI or MFI """
    v1 = 0.1 * (series - 50)
    if smoothing > 0:
        v2 = talib.WMA(v1.values, timeperiod=smoothing)
    else:
        v2 = v1
    return (np.exp(2 * v2) - 1) / (np.exp(2 * v2) + 1)


def update_RSI(dataframe):
    df = deepcopy(dataframe)
    df['rsi'] = abstract.RSI(df)
    df['fisher_rsi'] = fishers_inverse(df['rsi'])
    df['fisher_rsi_norma'] = 50 * (df['fisher_rsi'] + 1)
    return df


# def get_RSI(dataframe):
#     df = deepcopy(dataframe[-rsi_lookback_window:])
#     df['rsi'] = abstract.RSI(df)
#     df['fisher_rsi'] = fishers_inverse(df['rsi'])
#     df['fisher_rsi_norma'] = 50 * (df['fisher_rsi'] + 1)
#     rsi = df['fisher_rsi_norma'].iloc[-1:]
#     # logging.debug(rsi)
#     return rsi
#
def get_RSI(prices):
    # price = prices[-rsi_lookback_window:]
    price = np.array(prices)
    rsi_list = talib.RSI(price, timeperiod=14)
    fisher_list = fishers_inverse(rsi_list)
    rsi_norma = 50 * (fisher_list + 1)
    rsi = rsi_norma[-1]
    # logging.debug(rsi)
    return rsi


def get_RSI_sig(rsi):
    if rsi > 70:
        return -1
    elif rsi < 30:
        return 1
    return 0

def get_long_macd(prices):

    price = np.array(prices)
    macd_raw, signal, hist = talib.MACD(price, fastperiod=12, slowperiod=26, signalperiod=9)
    macd = macd_raw[-1] - signal[-1]
    return macd

def get_MACD(prices):
    price = np.array(prices)
    macd_raw, signal, hist = talib.MACD(price, fastperiod=12, slowperiod=26, signalperiod=9)
    macd = macd_raw[-1] - signal[-1]
    return macd


def get_MACD_sig(macd):
    if macd > 0 :
        return 1
    elif macd < 0 :
        return -1
    return 0


def get_BBANDS(prices):
    price = np.array(prices)
    upper, middle, lower = talib.BBANDS(
        price,
        timeperiod=10,
        # number of non-biased standard deviations from the mean
        nbdevup=2,
        nbdevdn=2,
        # Moving average type: simple moving average here
        matype=0)
    return lower[-1], upper[-1], middle[-1]


def get_BBANDS_sig(price, lower, upper):
    if price < lower:
        return 1
    elif price > upper:
        return -1
    return 0


def get_STOCH(high_prices, low_prices, close_prices):
    high = np.array(high_prices)
    low = np.array(low_prices)
    close = np.array(close_prices)
    slowk, slowd = talib.STOCH(high, low, close,
                               fastk_period=5,
                               slowk_period=3,
                               slowk_matype=0,
                               slowd_period=3,
                               slowd_matype=0)
    slowk = slowk[-1]
    slowd = slowd[-1]
    return slowk, slowd


def get_STOCH_sig(slowk, slowd):
    if slowk < 10 or slowd < 10:
        return 1
    elif slowk > 90 or slowd > 90:
        return -1
    return 0


def get_ATR(high_prices, low_prices, close_prices):
    high = np.array(high_prices)
    low = np.array(low_prices)
    close = np.array(close_prices)
    atr = talib.ATR(high, low, close, timeperiod=14)[-1]
    return atr


def get_ATR_sig(price, prev_close, atr):
    upside_signal = price - (prev_close + atr)
    downside_signal = prev_close - (price + atr)
    if upside_signal > 0:
        return 1
    elif downside_signal > 0:
        return -1
    return 0


def get_transaction_cost(volume, trade_price, ratio):
    return volume * trade_price * ratio


factor_limit = 100000
def getSellFactor(pos, factor):
    if pos > 0:
        factor = factor * (pos/factor_limit+1)
    elif float_equal_int(pos, 0) :
        return factor
    else:
        factor = factor / (abs(pos)/factor_limit+1)
    # d(factor)
    return factor
def getBuyFactor(pos, factor):
    if pos < 0:
        factor = factor * (abs(pos)/factor_limit+1)
    elif float_equal_int(pos, 0):
        return factor
    else:
        factor = factor / (pos/factor_limit+1)
    # d(factor)
    return factor




minimal_roi = {
    "1440": 0.01,
    "80": 0.02,
    "40": 0.03,
    "20": 0.04,
    "0": 0.05
}
roi_rate = [0.01, 0.02, 0.03, 0.04, 0.05]
# roi_rate = [0.003, 0.005, 0.007, 0.01, 0.02]
# roi_time = [1440, 80, 40, 20, 0]
roi_time = [160, 80, 40, 20, 0]

data_time_interval = [1, 5, 60]

TRADE_INTERVAL = 5
# Here is your main strategy function
# Note:
# 1. DO NOT modify the function parameters (time, data, etc.)
# 2. The strategy function AWAYS returns two things - position and memory:
# 2.1 position is a np.array (length 4) indicating your desired position of four crypto currencies next minute
# 2.2 memory is a class containing the information you want to save currently for future use


def handle_bar(counter,  # a counter for number of minute bars that have already been tested
               time,  # current time in string format such as "2018-07-30 00:30:00"
               data,  # data for current minute bar (in format 2)
               init_cash,  # your initial cash, a constant
               transaction,  # transaction ratio, a constant
               cash_balance,  # your cash balance at current minute
               crypto_balance,  # your crpyto currency balance at current minute
               total_balance,  # your total balance at current minute
               position_current,  # your position for 4 crypto currencies at this minute
               memory  # a class, containing the information you saved so far
               ):
    # helper for logging
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)

    position_new = deepcopy(position_current)

    if counter == 0:
        memory.data = []
        # counters
        memory.trade_cnt = 0
        memory.buy_cnt = 0
        memory.borrow_cnt = 0
        memory.take_profit_cnt = 0


        memory.avg_cost = [0] * ticker_num

        memory.roi_timer = [None] * len(roi_rate)

        memory.expected_position = position_current
        memory.last_position = position_current

        memory.macd_gold_cross_cnt = 0
        memory.macd_dead_cross_cnt = 0
        memory.kd_gold_cross_cnt = 0
        memory.kd_dead_cross_cnt = 0



    # always do this, even coutner = 0
    memory.data.append(data)


    ## Deal with data change due to time interval change
    # global watch_back_window, rsi_lookback_window, macd_lookback_window, bbands_lookback_window, stoch_lookback_window, atr_lookback_window
    # watch_back_window  *=  time_interval
    # rsi_lookback_window *=  time_interval
    # macd_lookback_window *=  time_interval
    # bbands_lookback_window *=  time_interval
    # stoch_lookback_window *=  time_interval
    # atr_lookback_window *=  time_interval


    ## combine data in hour interval
    if counter == 0:
        # init
        memory.timed_data = []
        memory.timed_average_prices = []
        memory.timed_high_prices = []
        memory.timed_low_prices = []
        memory.timed_close_prices = []
        memory.timed_macds = []
        memory.timed_rsi = []
        memory.timed_stochks = []
        memory.timed_stochds = []
        memory.timed_position = []
        memory.timed_lowerbs = []
        memory.timed_higherbs = []
        memory.timed_middlebs = []
        for i in range(0, len(data_time_interval)):
            memory.timed_data.append([])
            memory.timed_average_prices.append([])
            memory.timed_high_prices.append([])
            memory.timed_low_prices.append([])
            memory.timed_close_prices.append([])
            memory.timed_macds.append([])
            memory.timed_rsi.append([])
            memory.timed_stochks.append([])
            memory.timed_stochds.append([])
            memory.timed_position.append([])
            memory.timed_lowerbs.append([])
            memory.timed_higherbs.append([])
            memory.timed_middlebs.append([])
            for j in range(0, ticker_num):
                memory.timed_data[i].append([])
                memory.timed_average_prices[i].append([])
                memory.timed_high_prices[i].append([])
                memory.timed_low_prices[i].append([])
                memory.timed_close_prices[i].append([])
                memory.timed_macds[i].append([])
                memory.timed_rsi[i].append([])
                memory.timed_stochks[i].append([])
                memory.timed_stochds[i].append([])
                memory.timed_position[i].append([])
                memory.timed_lowerbs[i].append([])
                memory.timed_higherbs[i].append([])
                memory.timed_middlebs[i].append([])
        # return position_current, memory

    if True:
        index = 0
        for interval in data_time_interval:
            if counter % interval != 0:
                pass
            else:
                if ((counter == 0 and interval == 1) or (counter !=0)) != True:
                    continue

                data_in_time = memory.data[-interval:]

                day_start = data_in_time[0]
                starts = [-1] * ticker_num
                for i in range(0, ticker_num):
                    starts[i] = day_start[i][3]
                day_last = data_in_time[interval - 1]
                closes = [-1] * ticker_num
                for i in range(0, ticker_num):
                    closes[i] = day_last[i][0]
                day_all = data_in_time
                maxs = [None] * ticker_num
                mins = [None] * ticker_num
                volumes = [None] * ticker_num
                for ticker_cnt in range(0, ticker_num):
                    highs = []
                    lows = []
                    vols = []
                    for day_index in range(0, interval):
                        highs.append(day_all[day_index][ticker_cnt][1])
                        lows.append(day_all[day_index][ticker_cnt][2])
                        vols.append(day_all[day_index][ticker_cnt][4])
                    maxs[ticker_cnt] = max(highs)
                    mins[ticker_cnt] = min(lows)
                    volumes[ticker_cnt] = sum(vols)
                # get aggregated data
                closes = np.array(closes)
                maxs = np.array(maxs)
                timed_data = np.concatenate((closes.reshape(-1, 1), maxs.reshape(-1, 1)), axis=1)
                mins = np.array(mins)
                timed_data = np.concatenate((data, mins.reshape(-1, 1)), axis=1)
                starts = np.array(starts)
                timed_data = np.concatenate((data, starts.reshape(-1, 1)), axis=1)
                volumes = np.array(volumes)
                timed_data = np.concatenate((data, volumes.reshape(-1, 1)), axis=1)

                memory.timed_data[index].append(timed_data)


                ## deal with data
                highs = get_high(timed_data)
                lows = get_low(timed_data)
                opens = get_open(timed_data)
                closes = get_close(timed_data)
                averages = get_avg_price(timed_data)
                volumes = get_volume(timed_data)

                for ticker_cnt in range(0, ticker_num):
                    memory.timed_average_prices[index][ticker_cnt].append(averages[ticker_cnt])
                    memory.timed_high_prices[index][ticker_cnt].append(highs[ticker_cnt])
                    memory.timed_low_prices[index][ticker_cnt].append(lows[ticker_cnt])
                    memory.timed_close_prices[index][ticker_cnt].append(closes[ticker_cnt])


                    avg_prices = memory.timed_average_prices[index][ticker_cnt]
                    if counter > rsi_lookback_window * data_time_interval[index]:
                        rsi = get_RSI(memory.timed_average_prices[index][ticker_cnt][-rsi_lookback_window:])
                        memory.timed_rsi[index][ticker_cnt].append(rsi)
                    if counter > stoch_lookback_window * data_time_interval[index]:
                        slowk, slowd = get_STOCH(memory.timed_high_prices[index][ticker_cnt][-stoch_lookback_window:],
                                                 memory.timed_low_prices[index][ticker_cnt][-stoch_lookback_window:],
                                                 memory.timed_close_prices[index][ticker_cnt][-stoch_lookback_window:])
                        memory.timed_stochks[index][ticker_cnt].append(slowk)
                        memory.timed_stochds[index][ticker_cnt].append(slowd)
                    if counter > macd_lookback_window * data_time_interval[index]:
                        macd_value = get_MACD(memory.timed_average_prices[index][ticker_cnt][-macd_lookback_window:])
                        memory.timed_macds[index][ticker_cnt].append(macd_value)
                    if counter > bbands_lookback_window * data_time_interval[index]:
                        lowerb, higherb, middleb = get_BBANDS(avg_prices[-bbands_lookback_window:])
                        memory.timed_lowerbs[index][ticker_cnt].append(lowerb)
                        memory.timed_higherbs[index][ticker_cnt].append(higherb)
                        memory.timed_middlebs[index][ticker_cnt].append(middleb)

            index += 1


    # deal with init watch back window
    basic_time_interval_index = 1
    interval = data_time_interval[basic_time_interval_index]
    if 1 <= counter <= watch_back_window * interval:
        return position_current, memory


    # phase 1: execute transaction in last minute using value in this minute
    # check transaction meta at last minute and execute it
    # we cannot execute it at last time
    # .. because the transaction volume of last second may exceed 0.25*max
    # .. and the price has changed compared to the last minute
    # SO we execute here after one minute
    expected_trans_volume = memory.expected_position - memory.last_position
    position_last = memory.last_position

    if (position_current == 0).sum() == len(position_current):
        pass
    elif (expected_trans_volume == 0).sum() == len(expected_trans_volume):
        # no trading at last time
        pass
    else:
        # justify whether trans are cut
        real_trans_volume = position_current - memory.last_position
        # calculate total cost and avg cost

        ##
        ##
        for asset_idx in USE_ASSET_INDEX:

            avg_cost = memory.avg_cost[asset_idx]
            last_pos = position_last[asset_idx]
            cur_pos = position_current[asset_idx]
            amount = abs(cur_pos)
            volume = abs(cur_pos - last_pos)


            if volume == 0:
                continue
            if float_equal_int(cur_pos, 0):
                avg_cost = 0
                continue


            last_avg_price = memory.timed_average_prices[0][asset_idx][-2]

            transactionFee = get_transaction_cost(volume, last_avg_price, transaction)
            # feeFactor can directly add to average cost
            feeFactor = transactionFee / amount

            if last_pos >= 0 and cur_pos >= 0:
                # normal sell
                if last_pos >= cur_pos:
                    avg_cost = avg_cost
                # normal buy
                else:
                    avg_cost = (avg_cost * (amount - volume) + last_avg_price * volume) / amount + feeFactor
            elif last_pos <= 0 and cur_pos <= 0:
                # future sell
                if last_pos <= cur_pos:
                    avg_cost = avg_cost
                # future buy
                else:
                    avg_cost = (avg_cost * (amount - volume) + last_avg_price * volume) / amount + feeFactor
            elif last_pos <= 0 and cur_pos >= 0:
                # sell future and then buy in
                avg_cost = last_avg_price
            elif last_pos >= 0 and cur_pos <= 0:
                # sell and then buy future
                avg_cost = last_avg_price
            else:
                d("wrong!!!")
            # d(asset_idx)
            # d(avg_cost)
            # d(last_avg_price)
            memory.avg_cost[asset_idx] = avg_cost + feeFactor


    # phase 2: build new transactions based on Cost_total and previous price

    buy_in, borrow_in = False, False
    for asset_idx in USE_ASSET_INDEX:
        #
        if counter % TRADE_INTERVAL == 0:
            continue


        # Strategy paramter
        STRAREGY_1 = True
        STRAREGY_1_ONLY = False
        TAKE_PROFIT = True
        CROSS_JUDGE = True

        STRAREGY_Trend_Judge = True

        sell_factor = 2
        buy_factor = 2

        # buy_one = [0.01, 0.01, 0.02, 0.01]
        buy_one = [0.01, 0.001, 0.1, 0.1]
        buy_one = list(map(lambda x: x * TRADE_INTERVAL, buy_one))
        sell_one = buy_one


        # set to 1 just to turn on the cross policy
        long_trend = 1
        if STRAREGY_Trend_Judge:

            # Init Paramter
            bull = 0
            bear = 0
            balance = 0

            # Indictor Judge Phase

            MACD_ON = True
            BBANDS_ON = True

            time_interval_index = 2

            interval = data_time_interval[time_interval_index]
            price = memory.timed_average_prices[0][asset_idx][-1]

            if BBANDS_ON == True:
                bbands_start_time = bbands_lookback_window * interval
                if counter > bbands_start_time + 66:
                    upper_bbands = memory.timed_higherbs[time_interval_index][asset_idx][-1]
                    lower_bbands = memory.timed_lowerbs[time_interval_index][asset_idx][-1]
                    middle_bbands = memory.timed_middlebs[time_interval_index][asset_idx][-1]
                    if price > middle_bbands:
                        bull += 1
                    elif price < middle_bbands:
                        bear += 1

            if MACD_ON == True:
                start_time = macd_lookback_window * interval
                if counter > start_time + 66:
                    # d(memory.timed_macds[2][asset_idx])
                    macd = memory.timed_macds[time_interval_index][asset_idx][-1]
                    if macd > 0:
                        bull += 1
                    else:
                        bear += 1

            ## Total Judge Phase

            long_trend = bull - bear
            if long_trend > 0:
                sell_factor = getSellFactor(position_current[asset_idx], 2)
                buy_factor = getBuyFactor(position_current[asset_idx], 4)
                buy_one = list(map(lambda x: x * buy_factor, buy_one))
                sell_one = list(map(lambda x: x * sell_factor, sell_one))
                # d("bullish market")
                # d(long_trend)
            elif long_trend < 0:
                sell_factor = getSellFactor(position_current[asset_idx], 4)
                buy_factor = getBuyFactor(position_current[asset_idx], 2)
                buy_one = list(map(lambda x: x * buy_factor, buy_one))
                sell_one = list(map(lambda x: x * sell_factor, sell_one))
                # d("bearish market")
                # d(long_trend)
            else:
                # maybe in still or shock
                sell_factor = getSellFactor(position_current[asset_idx], 2)
                buy_factor = getBuyFactor(position_current[asset_idx], 2)
                buy_one = list(map(lambda x: x * buy_factor, buy_one))
                sell_one = list(map(lambda x: x * sell_factor, sell_one))
                # d("shock or still market")





        if STRAREGY_1  == True:
            # time interval index = 1, means 5 min
            time_interval_index= 1

            local_time_interval = data_time_interval[time_interval_index]

            start_time = macd_lookback_window * local_time_interval
            if counter <= start_time:
                continue

            average_prices = memory.timed_average_prices[time_interval_index]
            high_prices = memory.timed_high_prices[time_interval_index]
            low_prices = memory.timed_low_prices[time_interval_index]
            close_prices = memory.timed_close_prices[time_interval_index]

            # d(average_prices)
            rsi_value = memory.timed_rsi[time_interval_index][asset_idx][-1]
            rsi = get_RSI_sig(rsi_value)

            macd_value =  memory.timed_macds[time_interval_index][asset_idx][-1]
            macd = get_MACD_sig(macd_value)


            lowerb = memory.timed_lowerbs[time_interval_index][asset_idx][-1]
            higherb = memory.timed_higherbs[time_interval_index][asset_idx][-1]
            bbands = get_BBANDS_sig(average_prices[asset_idx][-1], lowerb, higherb)


            slowk = memory.timed_stochks[time_interval_index][asset_idx][-1]
            slowd = memory.timed_stochds[time_interval_index][asset_idx][-1]
            stoch = get_STOCH_sig(slowk, slowd)

            # atr_value = get_ATR(high_prices[asset_idx][-atr_lookback_window:],
            #                     low_prices[asset_idx][-atr_lookback_window:],
            #                     close_prices[asset_idx][-atr_lookback_window:])
            # atr = get_ATR_sig(average_prices[asset_idx][-1], close_prices[asset_idx][-2], atr_value)

            votes = macd + rsi + bbands + stoch
            # votes = bbands
            # votes = stoch + macd
            # votes = macd
            # votes = stoch
            # votes = 0


            # cross
            # in shock market this strategy maybe will produce so many fake signals
            if long_trend != 0 and CROSS_JUDGE:
                cross_cnt = 2
                if counter > (macd_lookback_window * 2) * local_time_interval:
                    last_macd_value = memory.timed_macds[time_interval_index][asset_idx][-2]
                    if macd_value > 0 and last_macd_value < 0:
                        votes += cross_cnt
                        # if memory.macd_dead_cross_cnt == 1:
                        #     memory.macd_dead_cross_cnt = 0
                        # if memory.macd_gold_cross_cnt == 0:
                        #     memory.macd_gold_cross_cnt = 1
                        # if memory.macd_gold_cross_cnt == 1:
                        #     votes += cross_cnt
                        #     memory.macd_gold_cross_cnt = 0
                        # d("gold macd")
                        pass

                    elif macd_value < 0 and last_macd_value > 0:
                        votes -= cross_cnt
                        # if memory.macd_gold_cross_cnt == 1:
                        #     memory.macd_gold_cross_cnt = 0
                        # if memory.macd_dead_cross_cnt == 0:
                        #     memory.macd_dead_cross_cnt = 1
                        # if memory.macd_dead_cross_cnt == 1:
                        #     votes -= cross_cnt
                        #     memory.macd_dead_cross_cnt = 0
                        # d("down cross macd")
                        pass

                if counter > (stoch_lookback_window * 2) * local_time_interval:
                    last_slowk = memory.timed_stochks[time_interval_index][asset_idx][-2]
                    last_slowd = memory.timed_stochds[time_interval_index][asset_idx][-2]
                    if slowd > slowk and last_slowd < last_slowk and (slowd > 80 or slowk > 80):
                        votes -= cross_cnt
                        # if memory.kd_gold_cross_cnt == 1:
                        #     memory.kd_gold_cross_cnt = 0
                        # if memory.kd_dead_cross_cnt == 0:
                        #     memory.kd_dead_cross_cnt = 1
                        # if memory.kd_dead_cross_cnt == 1:
                        #     votes -= cross_cnt
                        #     memory.kd_dead_cross_cnt = 0
                        # d("kd down cross!")
                        pass
                    elif slowd < slowk and last_slowd > last_slowk and (slowd < 20 or slowk < 20):
                        votes += cross_cnt
                        # if memory.kd_dead_cross_cnt == 1:
                        #     memory.kd_dead_cross_cnt = 0
                        # if memory.kd_gold_cross_cnt == 0:
                        #     memory.kd_gold_cross_cnt = 1
                        # if memory.kd_gold_cross_cnt == 1:
                        #     votes += cross_cnt
                        #     memory.kd_gold_cross_cnt = 0
                        # # d("kd gold cross!")
                        pass

            # votes = 0
            # if bbands == 1:
            #     votes = 1
            # if (rsi == 1 and stoch == 1) or (bbands == 1 and rsi == 1):
            #     votes = 2
            # elif rsi + bbands + stoch <= -2:
            #     votes = -2
            # elif rsi == -1 or stoch == -1 or bbands == -1:
            #     votes = -1

            # d("votes")
            # d(votes)

            interval = 300
            buy_show_interval = 200
            if counter % interval == 0:
                d("cash: " + str(cash_balance))
                d("crypto: " + str(crypto_balance))
                d("total: " + str(total_balance))
            # votes = bbands
            if votes > 0:
                buy_in = True
                memory.trade_cnt += 1
                memory.buy_cnt += 1
                if memory.buy_cnt % buy_show_interval == 0:
                    d("buy: " + str(memory.buy_cnt))
                # d(buy_one[asset_idx])
                position_new[asset_idx] += buy_one[asset_idx] * votes

                # position_new[asset_idx] += cash_balance*buy_portion
                # position_new[asset_idx] += buy_in_each_bitcoin
            elif votes < 0:
                borrow_in = True
                memory.trade_cnt += 1
                memory.borrow_cnt += 1
                # d(sell_one[asset_idx])
                if memory.borrow_cnt % buy_show_interval == 0:
                    d("borrow: " + str(memory.borrow_cnt))

                # d("sell_one:")
                # d(sell_one[asset_idx])
                position_new[asset_idx] += sell_one[asset_idx] * votes
                # position_new[asset_idx] -= abs(cash_balance)*sell_portion
                # position_new[asset_idx] -= borrow_each_bitcoin

        if STRAREGY_1_ONLY == True:
            continue

        if TAKE_PROFIT == True:
            profit_unit = -1

            #TODO://is this right?
            if float_equal_int(memory.avg_cost[asset_idx], 0):
                continue


            if position_current[asset_idx] > 0:
                # positive position
                profit_unit = memory.timed_average_prices[0][asset_idx][-1] - memory.avg_cost[asset_idx]
            else:
                profit_unit = memory.avg_cost[asset_idx] - memory.timed_average_prices[0][asset_idx][-1]

            if position_new[asset_idx] != 0:
                profit_rate = profit_unit / abs(memory.avg_cost[asset_idx])
            else:
                profit_rate = 0
            # d(profit_rate)

            ROI_satisfied = False
            for i in range(0, len(roi_rate)):
                if profit_rate > roi_rate[i]:
                    memory.roi_timer[i] += 1
                    # d('has chance')
                    # d(memory.roi_timer[i])
                else:
                    memory.roi_timer[i] = 0

                if memory.roi_timer[i] > roi_time[i]:
                    ROI_satisfied = True
                    d("Take Profit")
                    d(roi_rate[i])
                    break
            if ROI_satisfied:
                position_new[asset_idx] = 0
                memory.take_profit_cnt += 1
                d('has taken profit')
                d(memory.take_profit_cnt)
                d(" times")
                for i in range(0, len(roi_rate)):
                    memory.roi_timer[i] = 0
                memory.avg_cost[asset_idx] = 0


    # if valid, execute buy-in transaction
    # if cash_balance_after_transaction > my_cash_balance_lower_limit:
    # ------ do not justify whether currency < 10000
    cash_after_transaction = cash_balance
    # safe_limit = 0
    # safe_buy_limit = 0
    # safe_sell_limit = 0
    # for asset_idx in USE_ASSET_INDEX:
    #     safe_limit = buy_one* memory.timed_average_prices[0][asset_idx][-1]
    for asset_idx in USE_ASSET_INDEX:
        # d("test start")
        # d(memory.timed_average_prices[0][asset_idx][-1])
        # d(position_current[asset_idx])
        # d(position_new[asset_idx])
        # d("test end")



        cash_after_transaction -= \
            ((np.abs(position_new[asset_idx]) - np.abs(position_current[asset_idx])) * memory.timed_average_prices[0][asset_idx][-1]) * (
                        1 + transaction)
    if cash_after_transaction > my_cash_balance_lower_limit + 5000:

        memory.expected_position = position_new
        memory.last_position = position_current
    else:
        position_new = position_current
        # d("jump out because cash limit 10000")
        # d(cash_after_transaction)
        # d(cash_balance)
        # d(crypto_balance)

    return position_new, memory
