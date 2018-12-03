import h5py
import pandas as pd
import numpy as np
import copy
import os
import sys
from copy import deepcopy
import talib
import logging
from talib import abstract
import time as timer

cols = ['counter', 'open', 'high', 'low', 'close', 'average', 'volume']

min_profit = 0.001
ticker_num = 4
data_time_interval = [1, 5, 60]

watch_back_window = 35
rsi_lookback_window = 15
macd_lookback_window = 34
bbands_lookback_window = 15
stoch_lookback_window = 10
atr_lookback_window = 15

max_lookback_window = max(watch_back_window, rsi_lookback_window, macd_lookback_window, bbands_lookback_window, stoch_lookback_window, atr_lookback_window)
max_time_interval = max(data_time_interval)

ALL_ASSET = [0, 1, 2, 3]
USE_ASSET_INDEX = [1]
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


def fishers_inverse(series, smoothing=0):
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
    if macd > 0:
        return 1
    elif macd < 0:
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


def extract_features_from_memory(memory_obj):
    interval_nslot = {
        1: 1,
        5: 1,
        60: 1,
    }

    features = np.array([])
    for i in range(0, len(data_time_interval)):
        n_slot = interval_nslot[data_time_interval[i]]
        for j in range(0, ticker_num):  # ticker_num
            # cleanup and copy (not include high low avg and close, they are cleaned according to lookback interval)
            memory_obj.timed_macds[i][j] = memory_obj.timed_macds[i][j][-n_slot:]
            v_macd = memory_obj.timed_macds[i][j]
            memory_obj.timed_rsi[i][j] = memory_obj.timed_rsi[i][j][-n_slot:]
            v_rsi = memory_obj.timed_rsi[i][j]
            memory_obj.timed_stochks[i][j] = memory_obj.timed_stochks[i][j][-n_slot:]
            v_hks = memory_obj.timed_stochks[i][j]
            memory_obj.timed_stochds[i][j] = memory_obj.timed_stochds[i][j][-n_slot:]
            v_hds = memory_obj.timed_stochds[i][j]
            memory_obj.timed_lowerbs[i][j] = memory_obj.timed_lowerbs[i][j][-n_slot:]
            v_lbs = memory_obj.timed_lowerbs[i][j]
            memory_obj.timed_higherbs[i][j] = memory_obj.timed_higherbs[i][j][-n_slot:]
            v_hbs = memory_obj.timed_higherbs[i][j]
            memory_obj.timed_middlebs[i][j] = memory_obj.timed_middlebs[i][j][-n_slot:]
            v_mbs = memory_obj.timed_middlebs[i][j]

            # high low avg and close is not clean here, because they are used in computing other metrics
            v_high = memory_obj.timed_high_prices[i][j][-n_slot:]
            v_low = memory_obj.timed_low_prices[i][j][-n_slot:]
            v_avg = memory_obj.timed_average_prices[i][j][-n_slot:]
            v_close = memory_obj.timed_close_prices[i][j][-n_slot:]

            v1 = np.concatenate((v_macd, v_rsi, v_hks, v_hds, v_lbs, v_hbs, v_mbs, v_high, v_low, v_avg, v_close))
            features = np.concatenate((features, v1))

    # print(features.shape)
    return features


def handle_bar(counter,  # a counter for number of minute bars that have already been tested
               time,  # current time in string format such as "2018-07-30 00:30:00"
               data,  # data for current minute bar (in format 2)
               memory,  # a class, containing the information you saved so far
               price_future = None, # price in future two days
               ):
    if counter == 0:
        memory.records = None

        memory.data = []
        # counters
        memory.trade_cnt = 0
        memory.buy_cnt = 0
        memory.borrow_cnt = 0
        memory.take_profit_cnt = 0

        memory.avg_cost = [0] * ticker_num

        memory.roi_timer = [None] * len(roi_rate)

        memory.macd_gold_cross_cnt = 0
        memory.macd_dead_cross_cnt = 0
        memory.kd_gold_cross_cnt = 0
        memory.kd_dead_cross_cnt = 0

    # always do this, even coutner = 0
    memory.data.append(data)
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

    index = 0
    for interval in data_time_interval:
        if counter % interval != 0:
            pass
        else:
            if ((counter == 0 and interval == 1) or (counter != 0)) != True:
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
                # clean data that will never be used
                max_lookback_interval = max_lookback_window*max_time_interval
                memory.timed_average_prices[index][ticker_cnt] = memory.timed_average_prices[index][ticker_cnt][-max_lookback_interval:]
                memory.timed_high_prices[index][ticker_cnt] = memory.timed_high_prices[index][ticker_cnt][-max_lookback_interval:]
                memory.timed_low_prices[index][ticker_cnt] = memory.timed_low_prices[index][ticker_cnt][-max_lookback_interval:]
                memory.timed_close_prices[index][ticker_cnt] = memory.timed_close_prices[index][ticker_cnt][-max_lookback_interval:]

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
    basic_time_interval_index = 1
    interval = data_time_interval[basic_time_interval_index]
    if 1 <= counter <= watch_back_window * interval:
        return [], memory
    if counter > 4500:
        timestamp = time
        #t1 = timer.time()
        features = extract_features_from_memory(memory)
        #print("p1:", timer.time()-t1)
        #t1 = timer.time()
        labels = get_avg_price(price_future)
        rec = pd.DataFrame([features], index=[pd.to_datetime(timestamp)])
        # print(labels)
        #print("p2:", timer.time()-t1)
        #t1 = timer.time()
        # feature_dict = {}
        # for i in range(len(features)):
        #     feature_dict["f"+str(i)] = features[i]

        # feature_dict["time"] = timestamp
        # lab_dict = {}
        for i in range(ticker_num):
            # feature_dict["l"+str(i)] = labels[i]
            rec["l"+str(i+1)] = [labels[i]]
            # lab_dict["l"+str(i)] = [labels[i]]
        # rec = pd.concat([rec, pd.DataFrame(lab_dict)], axis=1)
        # print(rec)
        # print("p3:", timer.time()-t1)
        # t1 = timer.time()
        if memory.records is None:
            memory.records = rec
        else:
            memory.records = pd.concat([memory.records, rec], axis=0)
        # print("p4:", timer.time()-t1)
        # t1 = timer.time()
        # print(memory.records)
        if counter % 100 == 0:
            print("finish: %d" % counter)

    # compute metrics
    # print(memory.timed_macds)
    return [], memory


class memory:
    def __init__(self):
        pass


data_collect = [
    "data_format2_201801.h5",
    "data_format2_201802.h5",
    "data_format2_201803.h5",
    "data_format2_201804.h5",
    "data_format2_201805.h5",
    "data_format2_201806.h5",
    "data_format2_201807.h5",
    "data_format2_201808.h5",
    "data_format2_20180901_20180909.h5",
    "data_format2_20180909_20180916.h5",
    "data_format2_20180916_20180923.h5",
    "data_format2_20180923_20180930.h5",
    "data_format2_20180930_20181007.h5",
    "data_format2_20181007_20181014.h5",
    "data_format2_20181014_20181021.h5",
    "data_format2_20181021_20181028.h5",
    "data_format2_20181028_20181104.h5",
    "data_format2_20181104_20181111.h5",
    "data_format2_20181111_20181118.h5",
    "data_format2_20181118_20181125.h5",
]

mem_obj = memory()
for data_idx in range(len(data_collect)):
    data_filename = data_collect[data_idx]
    data_path = '../data/' + data_filename
    data_block = h5py.File(data_path, mode='r')
    keys = list(data_block.keys())
    future_minutes = 5
    counter = 0
    for i in range(len(keys)):
        data_cur_min = data_block[keys[i]][:]
        # print(keys[i+future_minutes])
        if i+future_minutes < len(keys):
            data_fut_min = data_block[keys[i+future_minutes]][:]
            fut_time = keys[i+future_minutes]
        else:
            if data_idx+1 < len(data_collect):
                new_data_block = h5py.File('../data/' + data_collect[data_idx+1], mode='r')
                new_keys = list(new_data_block.keys())
                data_fut_min = new_data_block[new_keys[i+future_minutes-len(keys)]][:]
                fut_time = new_keys[i+future_minutes-len(keys)]
            else:
                break
        _, mem_obj = handle_bar(counter,
                               keys[i],
                               data_cur_min,
                               mem_obj,
                                data_fut_min
                               )
        counter += 1
        if i % 100 == 0:
            print(keys[i])
        if i/10000 > 1 and i % 10000 == 0:
            mem_obj.records.to_csv("feature_all.csv")
            # print(data_cur_min)


