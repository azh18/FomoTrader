# Here you can
# 1. import necessary python packages for your strategy
# 2. Load your own facility files containing functions, trained models, extra data, etc for later use
# 3. Set some global constants
# Note:
# 1. You should put your facility files in the same folder as this strategy.py file
# 2. When load files, ALWAYS use relative path such as "data/facility.pickle"
# DO NOT use absolute path such as "C:/Users/Peter/Documents/project/data/facility.pickle"
import numpy as np
# from auxiliary import generate_bar, white_soider, black_craw  # auxiliary is a local py file containing some functions

my_cash_balance_lower_limit = 20000.  # Cutting-loss criterion
ALL_ASSET = [0, 1, 2, 3]
USE_ASSET_INDEX = [1]


watch_back_window = 5
buy_in_each_bitcoin = 1
buy_in_signal_dollar = 5
stop_loss_dollar = 3
stop_profit_dollar = 10


def get_avg_price(today_data):
    return np.mean(today_data[:, :4], axis=1)


def get_volume(today_data):
    return today_data[:, 4]


def float_equal_0(float_num):
    return np.abs(float_num) < 0.000001


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
    position_new = position_current

    if counter == 0:
        memory.data_list = list([data[:]])  # only store last M minutes
        memory.avg_price_list = list([get_avg_price(data)])
        memory.volume_list = list([get_volume(data)])
        # (0:asset_index, 1:buy_in_num, 2:buy_in_price, 3:trans_fee,
        # 4:time(minute), 5:balance_before, 6:balance_after)
        memory.asset_metadata = {}
        for i in USE_ASSET_INDEX:
            memory.asset_metadata[i] = list()
        memory.last_trans_asset_metadata = {}
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
    # 4:time(minute), 5:cash_balance_before, 6:cash_balance_after,
    # 7:asset_meta_to_operate_list)
    for asset_idx in USE_ASSET_INDEX:
        cur_volume = get_volume(data)[asset_idx]
        cur_price = get_avg_price(data)[asset_idx]
        # justify whether a trans occurred in last minute
        if asset_idx in memory.last_trans_asset_metadata:
            trans_volume = memory.last_trans_asset_metadata[asset_idx][1]
            is_over_buy = np.abs(trans_volume) > cur_volume*0.25
            trans_volume = np.sign(trans_volume)*cur_volume*0.25 if is_over_buy else trans_volume
            # if volume > 0 (buy), record this operate as a buy
            # else (sell), re-compute params of operated assets
            if trans_volume > 0:
                memory.last_trans_asset_metadata[asset_idx][1] = trans_volume
                memory.last_trans_asset_metadata[asset_idx][2] = cur_price
                memory.last_trans_asset_metadata[asset_idx][3] = (np.abs(trans_volume)*cur_price*(1.0+transaction))\
                                                          / trans_volume
                memory.last_trans_asset_metadata[asset_idx][6] = cash_balance
                memory.asset_metadata[asset_idx].append(memory.last_trans_asset_metadata[asset_idx])
            else:
                proportion = []
                is_one_asset_with_zero_storage = False
                for tran_tuple in memory.last_trans_asset_metadata[asset_idx][7]:
                    proportion.append(tran_tuple[0][1])
                proportion = np.array(proportion, dtype=np.float64)/np.sum(proportion)
                sell_volume = proportion * np.abs(trans_volume)
                idx = 0
                for tran_tuple in memory.last_trans_asset_metadata[asset_idx][7]:
                    tran = tran_tuple[0]
                    # delete storage volume by proportion
                    tran[1] -= sell_volume[idx]
                    if float_equal_0(tran[1]):
                        is_one_asset_with_zero_storage = True
                    idx += 1

                # there exists an asset with zero storage, find and delete it
                if is_one_asset_with_zero_storage:
                    memory.asset_metadata[asset_idx] = list(filter(lambda x: (not float_equal_0(x[1])),
                                                              memory.asset_metadata[asset_idx]))
            # clear it after finishing process it
            del memory.last_trans_asset_metadata[asset_idx]

    # phase 2: build new transactions

    # buy-in strategy: check rise magnitude
    avg_price = get_avg_price(data)
    buy_in = np.array([0.0] * len(ALL_ASSET))
    length_list = len(memory.avg_price_list)
    for i in USE_ASSET_INDEX:
        for diff_time in range(1, watch_back_window+1):
            if avg_price[i] - memory.avg_price_list[length_list-diff_time][i] >= buy_in_signal_dollar:
                buy_in[i] += buy_in_each_bitcoin
                break

    # sell strategy: cut loss and cut profit
    will_operated_asset_metadata = {}
    for asset_idx in USE_ASSET_INDEX:
        will_operated_asset_metadata[asset_idx] = []
        for idx in range(len(memory.asset_metadata[asset_idx])):
            tran = memory.asset_metadata[asset_idx][idx]
            asset_idx = tran[0]
            asset_num = tran[1]
            buy_in_price = tran[2]
            # stop profit
            if avg_price[asset_idx] - buy_in_price > stop_profit_dollar:
                buy_in[asset_idx] -= asset_num
                # if more information need to add, can add val with tran in tuple
                will_operated_asset_metadata[asset_idx].append((tran, ))
                continue

            # stop loss
            if avg_price[asset_idx] - buy_in_price < -stop_loss_dollar:
                buy_in[asset_idx] -= asset_num
                # if more information need to add, can add val with tran in tuple
                will_operated_asset_metadata[asset_idx].append((tran, ))
                continue

    # try to execute buy-in
    # (0:asset_index, 1:buy_in_num, 2:buy_in_price, 3:trans_fee,
    # 4:time(minute), 5:cash_balance_before, 6:cash_balance_after,
    # 7:asset_meta_to_operate_list)
    asset_metadata_will_valid = {}
    cash_balance_after_transaction = 0.0
    for asset_idx in USE_ASSET_INDEX:
        buy_in_num = buy_in[asset_idx]
        if float_equal_0(buy_in_num):
            continue
        buy_in_price = avg_price[asset_idx]
        trans_fee = buy_in_price * np.abs(buy_in_num) * transaction
        time = counter
        cash_balance_before = cash_balance
        cash_balance_after = cash_balance - buy_in_num*buy_in_price - trans_fee
        cash_balance_after_transaction = cash_balance_after
        asset_metadata_will_valid[asset_idx] = [asset_idx, buy_in_num, -1.0, -1.0, time,
                                                cash_balance_before, -1.0, will_operated_asset_metadata[asset_idx]]

    # if valid, execute buy-in transaction
    if cash_balance_after_transaction > my_cash_balance_lower_limit:
        position_new = position_new + buy_in
        # do not process meta to be invalid, process at next minute
        memory.last_trans_asset_metadata = asset_metadata_will_valid

    # update memory
    memory.data_list.append(data[:])
    memory.avg_price_list.append(avg_price)

    return position_new, memory
