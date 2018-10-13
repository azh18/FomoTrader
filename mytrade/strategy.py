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
# from auxiliary import generate_bar, white_soider, black_craw  # auxiliary is a local py file containing some functions

my_cash_balance_lower_limit = 20000.  # Cutting-loss criterion
ALL_ASSET = [0, 1, 2, 3]
USE_ASSET_INDEX = [1]


watch_back_window = 3
buy_in_each_bitcoin = 0.1
buy_in_signal_dollar = 5
stop_loss_dollar = 10000
stop_profit_dollar = 10
borrow_in_signal_dollar = 5
borrow_each_bitcoin = 0.1



def get_avg_price(today_data):
    return np.mean(today_data[:, :4], axis=1)


def get_volume(today_data):
    return today_data[:, 4]


def float_equal_int(float_num, int_num):
    return np.abs(float_num-int_num) < 0.000001


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
    # Here you should explain the idea of your strategy briefly in the form of Python comment.
    # You can also attach facility files such as text & image & table in your team folder to illustrate your idea

    # Get position of last minute
    position_new = deepcopy(position_current)

    if counter == 0:
        memory.data_list = list([data[:]])  # only store last M minutes
        memory.avg_price_list = list([get_avg_price(data)])
        memory.volume_list = list([get_volume(data)])
        # (0:asset_index, 1:buy_in_num, 2:buy_in_price, 3:trans_fee,
        # 4:time(minute), 5:balance_before, 6:balance_after)
        memory.expected_position = position_current
        memory.last_position = position_current
        memory.buy_in_asset_metadata = {}
        memory.borrow_asset_metadata = {}
        memory.will_operate_asset_data = {}
        for i in USE_ASSET_INDEX:
            memory.buy_in_asset_metadata[i] = list()
            memory.borrow_asset_metadata[i] = list()
            memory.will_operate_asset_data[i] = list()
        memory.last_trans_asset_metadata = {}
        memory.buy_in_asset_num = {}
        memory.borrow_asset_num = {}
        for i in USE_ASSET_INDEX:
            memory.buy_in_asset_num[i] = 0
            memory.borrow_asset_num[i] = 0
        return position_current, memory

    if 1 <= counter <= watch_back_window:
        memory.data_list.append(data[:])
        memory.avg_price_list.append(get_avg_price(data))
        memory.volume_list.append(get_volume(data))
        return position_current, memory

    # phase 1: execute transaction in last minute using value in this minute
    # check transaction meta at last minute and execute it
    # we cannot execute it at last time
    # .. because the transaction volume of last second may exceed 0.25*max
    # .. and the price has changed compared to the last minute
    # SO we execute here after one minute

    # (0:asset_index, 1:buy_in_num(storage), 2:buy_in_price, 3:avg_cost,
    # 4:time(minute), 5:cash_balance_before, 6:cash_balance_after)
    position_delta = memory.expected_position - position_current
    expected_trans_volume = memory.expected_position - memory.last_position

    if (position_current == 0).sum() == len(position_current):
        # clear all positions
        for i in ALL_ASSET:
            memory.buy_in_asset_metadata[i] = list()
            memory.borrow_asset_metadata[i] = list()
    elif (expected_trans_volume == 0).sum() == len(expected_trans_volume):
        # no trading at last time
        pass
    else:
        # justify whether trans are cut
        real_trans_volume = position_current - memory.last_position
        # success proportion:
        proportion = real_trans_volume / expected_trans_volume
        for i in range(len(proportion)):
            if np.isnan(proportion[i]):
                proportion[i] = 1.0
        for asset_idx in USE_ASSET_INDEX:
            cur_price = get_avg_price(data)[asset_idx]
            # justify whether a trans occurred in last minute
            if asset_idx in memory.last_trans_asset_metadata:
                for trans in memory.last_trans_asset_metadata[asset_idx]:
                    trans_volume = trans[1]
                    trans_volume = trans_volume * proportion[asset_idx]
                    # if volume > 0 (buy), record this operate as a buy
                    # else (borrow), re-compute params of operated assets
                    trans[1] = trans_volume
                    trans[2] = cur_price
                    trans[3] = cur_price
                    trans[4] = counter
                    if trans_volume > 0:
                        memory.buy_in_asset_metadata[asset_idx].append(trans)
                    else:
                        memory.borrow_asset_metadata[asset_idx].append(trans)
            if asset_idx in memory.will_operate_asset_data:
                for trans_tuple in memory.will_operate_asset_data[asset_idx]:
                    trans_volume = trans_tuple[0][1]
                    # all done. clean them in metadata record.
                    # else, clean then append new one with less volume regarding to the proportion
                    if trans_volume > 0:
                        memory.buy_in_asset_metadata[asset_idx].remove(trans_tuple[0])
                    else:
                        memory.borrow_asset_metadata[asset_idx].remove(trans_tuple[0])

                    if not float_equal_int(proportion[asset_idx], 1):
                        new_trans = trans_tuple[0]
                        new_trans[1] *= (1-proportion[asset_idx])
                        if trans_volume > 0:
                            memory.buy_in_asset_metadata[asset_idx].append(new_trans)
                        else:
                            memory.borrow_asset_metadata[asset_idx].append(new_trans)
    # clean out-dated memory
    memory.last_trans_asset_metadata.clear()
    memory.will_operate_asset_data.clear()
    for i in USE_ASSET_INDEX:
        memory.buy_in_asset_num[i] = 0
        memory.borrow_asset_num[i] = 0



    # phase 2: build new transactions

    # buy-in strategy: check rise magnitude
    avg_price = get_avg_price(data)
    buy_in = np.array([0.0] * len(ALL_ASSET))
    borrow = np.array([0.0] * len(ALL_ASSET))
    length_list = len(memory.avg_price_list)
    for i in USE_ASSET_INDEX:
        for diff_time in range(1, watch_back_window+1):
            if avg_price[i] - memory.avg_price_list[length_list-diff_time][i] >= buy_in_signal_dollar:
                buy_in[i] += buy_in_each_bitcoin
                break

    # to be design: borrow strategy
    ###
    for i in USE_ASSET_INDEX:
        for diff_time in range(1, watch_back_window+1):
            if avg_price[i] - memory.avg_price_list[length_list-diff_time][i] <= -borrow_in_signal_dollar:
                borrow[i] += borrow_each_bitcoin
                break
    ###

    # sell strategy: cut loss and cut profit
    will_operated_asset_metadata = {}
    for asset_idx in USE_ASSET_INDEX:
        will_operated_asset_metadata[asset_idx] = []
        # positive asset
        for idx in range(len(memory.buy_in_asset_metadata[asset_idx])):
            tran = memory.buy_in_asset_metadata[asset_idx][idx]
            asset_idx = tran[0]
            asset_num = tran[1]
            buy_in_price = tran[2]

            # stop profit
            if avg_price[asset_idx] - buy_in_price > stop_profit_dollar:
                # if more information need to add, can add val with tran in tuple
                will_operated_asset_metadata[asset_idx].append((tran, ))
                position_new[asset_idx] -= asset_num
                continue

            # stop loss
            if avg_price[asset_idx] - buy_in_price < -stop_loss_dollar:
                # if more information need to add, can add val with tran in tuple
                will_operated_asset_metadata[asset_idx].append((tran, ))
                position_new[asset_idx] -= asset_num
                continue

        # negative asset
        for idx in range(len(memory.borrow_asset_metadata[asset_idx])):
            tran = memory.borrow_asset_metadata[asset_idx][idx]
            asset_idx = tran[0]
            asset_num = -tran[1]
            borrow_price = tran[2]

            # stop profit
            if borrow_price - avg_price[asset_idx] > stop_profit_dollar:
                # if more information need to add, can add val with tran in tuple
                will_operated_asset_metadata[asset_idx].append((tran, ))
                position_new[asset_idx] += asset_num
                continue

            # stop loss
            if borrow_price - avg_price[asset_idx] < -stop_loss_dollar:
                # if more information need to add, can add val with tran in tuple
                will_operated_asset_metadata[asset_idx].append((tran, ))
                position_new[asset_idx] += asset_num
                continue

    # try to execute buy-in & borrow & sell
    # (0:asset_index, 1:buy_in_num, 2:buy_in_price, 3:trans_fee,
    # 4:time(minute), 5:cash_balance_before, 6:cash_balance_after)

    asset_metadata_will_valid = {}
    for asset_idx in USE_ASSET_INDEX:
        asset_metadata_will_valid[asset_idx] = list()

    # buy-in
    cash_balance_after_transaction = cash_balance
    for asset_idx in USE_ASSET_INDEX:
        buy_in_num = buy_in[asset_idx]
        if float_equal_int(buy_in_num, 0):
            continue
        buy_in_price = avg_price[asset_idx]
        trans_fee = buy_in_price * np.abs(buy_in_num) * transaction
        time = counter
        cash_balance_before = cash_balance
        cash_balance_after = cash_balance - buy_in_num*buy_in_price - trans_fee
        cash_balance_after_transaction = cash_balance_after
        asset_metadata_will_valid[asset_idx].append([asset_idx, buy_in_num, -1.0, -1.0, time,
                                                cash_balance_before, -1.0])
        position_new[asset_idx] += buy_in_num

    # borrow
    for asset_idx in USE_ASSET_INDEX:
        borrow_num = borrow[asset_idx]
        if float_equal_int(borrow_num, 0):
            continue
        borrow_price = avg_price[asset_idx]
        trans_fee = borrow_price * np.abs(borrow_num) * transaction
        time = counter
        cash_balance_before = cash_balance_after_transaction
        cash_balance_after_transaction -= (borrow_num * borrow_price + trans_fee)
        asset_metadata_will_valid[asset_idx].append([asset_idx, -borrow_num, -1.0, -1.0, time,
                                                cash_balance_before, -1.0])
        position_new[asset_idx] -= borrow_num

    # if valid, execute buy-in transaction
    # if cash_balance_after_transaction > my_cash_balance_lower_limit:
    # ------ do not justify whether currency < 10000
    if True:
        memory.expected_position = position_new
        memory.last_position = position_current
        # do not process meta to be invalid, process at next minute
        memory.last_trans_asset_metadata = asset_metadata_will_valid
        memory.will_operate_asset_data = will_operated_asset_metadata

    # update memory
    memory.data_list.append(data[:])
    memory.avg_price_list.append(avg_price)

    return position_new, memory
