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

my_cash_balance_lower_limit = 10000.  # Cutting-loss criterion
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
    return np.abs(float_num - int_num) < 0.000001


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

    # TODO: to be cleaned
    if counter == 0:
        memory.data_list = list([data[:]])  # only store last M minutes
        memory.avg_price_list = list([get_avg_price(data)])
        memory.volume_list = list([get_volume(data)])
        memory.Cost_total = np.array([0, 0, 0, 0])  # total cost, which is used to calculate avg cost

        # (0:asset_index, 1:buy_in_num, 2:buy_in_price, 3:trans_fee,
        # 4:time(minute), 5:balance_before, 6:balance_after)
        memory.expected_position = position_current
        memory.last_position = position_current
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
            cur_price = get_avg_price(data)[asset_idx]
            if position_current[asset_idx] == position_last[asset_idx]:
                # no transaction at all
                continue
            if np.sign(position_current[asset_idx]) != np.sign(position_last[asset_idx]):
                trans_cost = np.abs(position_current[asset_idx]) * cur_price * (1 + transaction)
                memory.Cost_total[asset_idx] = trans_cost
            else:
                trans_cost = np.abs(position_current[asset_idx] - position_last[asset_idx]) * cur_price * transaction
                if np.sign(position_current[asset_idx]) == np.sign(1):
                    if position_current[asset_idx] > position_last[asset_idx]:
                        memory.Cost_total[asset_idx] += (
                                    (position_current[asset_idx] - position_last[asset_idx]) * cur_price + trans_cost)
                    else:
                        avg_cost = memory.Cost_total[asset_idx] / position_last[asset_idx]
                        memory.Cost_total[asset_idx] += (
                                    (position_current[asset_idx] - position_last[asset_idx]) * avg_cost + trans_cost)
                else:
                    if position_current[asset_idx] < position_last[asset_idx]:
                        memory.Cost_total[asset_idx] += (np.abs(
                            position_current[asset_idx] - position_last[asset_idx]) * cur_price - trans_cost)
                    else:
                        avg_cost = memory.Cost_total[asset_idx] / position_last[asset_idx]
                        memory.Cost_total[asset_idx] += (np.abs(
                            position_current[asset_idx] - position_last[asset_idx]) * avg_cost - trans_cost)

    # get avg_cost for each asset
    avg_cost = np.array([0., 0., 0., 0.])
    for asset_idx in USE_ASSET_INDEX:
        avg_cost[asset_idx] = 0 if float_equal_int(position_current[asset_idx], 0) else memory.Cost_total[asset_idx] / \
                                                                                        position_current[asset_idx]

    # phase 2: build new transactions based on Cost_total and previous price
    avg_price = get_avg_price(data)
    buy_in, borrow_in = False, False
    length_list = len(memory.avg_price_list)
    for asset_idx in USE_ASSET_INDEX:
        # buy-in strategy: check rise magnitude
        for diff_time in range(1, watch_back_window + 1):
            if avg_price[asset_idx] - memory.avg_price_list[length_list - diff_time][asset_idx] >= buy_in_signal_dollar:
                buy_in = True
                position_new[asset_idx] += buy_in_each_bitcoin
                break
        if buy_in:
            continue
        # borrow strategy
        ###
        for diff_time in range(1, watch_back_window + 1):
            if avg_price[asset_idx] - memory.avg_price_list[length_list - diff_time][asset_idx] <= -borrow_in_signal_dollar:
                borrow_in = True
                position_new[asset_idx] -= borrow_each_bitcoin
                break
        if borrow_in:
            continue

        # sell strategy based on avg cost
        ###
        stop_profit_proportion = 0.2
        stop_loss_proportion = 0.2
        # sell strategy: cut loss and cut profit
        if position_current[asset_idx] > 0:
            # positive position
            profit_unit = avg_price[asset_idx] - avg_cost[asset_idx]
            if profit_unit > stop_profit_dollar:
                position_new[asset_idx] -= position_current[asset_idx] * stop_profit_proportion
            elif profit_unit < -stop_loss_dollar:
                position_new[asset_idx] -= position_current[asset_idx] * stop_loss_proportion
        else:
            # negative position
            profit_unit = avg_cost[asset_idx] - avg_price[asset_idx]
            if profit_unit > stop_profit_dollar:
                position_new[asset_idx] -= position_current[asset_idx] * stop_profit_proportion
            elif profit_unit < -stop_loss_dollar:
                position_new[asset_idx] -= position_current[asset_idx] * stop_loss_proportion

    # if valid, execute buy-in transaction
    # if cash_balance_after_transaction > my_cash_balance_lower_limit:
    # ------ do not justify whether currency < 10000
    cash_after_transaction = cash_balance
    for asset_idx in USE_ASSET_INDEX:
        cash_after_transaction -= \
            ((np.abs(position_new[asset_idx]) - np.abs(position_current[asset_idx])) * avg_price[asset_idx]) * (1 + transaction)
    if cash_after_transaction > my_cash_balance_lower_limit + 5000:
        memory.expected_position = position_new
        memory.last_position = position_current
    else:
        position_new = position_current

    # update memory
    memory.data_list.append(data[:])
    memory.avg_price_list.append(avg_price)

    return position_new, memory
