import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler

BATCH_SIZE = 64

LSTM_HIDDEN_SIZE = 128


FEATURE_NUM = 22

NUM_STEPS = 48

USE_GPU = False

LEARNING_RATE = 0.001

Labels = ["test"]

class LSTM_model(object):
    def __init__(self, inputs, batch_size=BATCH_SIZE, state_size=LSTM_HIDDEN_SIZE, layer_num=2):
        self.cur_batch = 0

        if USE_GPU:
            # time major for cudnn_rnn
            self.inputs = tf.transpose(inputs, [1, 0, 2])

            self.cell = tf.contrib.cudnn_rnn.CudnnLSTM(layer_num, state_size)

            outputs, _ = self.cell(self.inputs)
            outputs = tf.transpose(outputs, [1, 0, 2])
            self.output = outputs[:, -1, :]


        else:
            self.inputs = inputs
            self.stacked_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(state_size) \
                                                             for _ in range(layer_num)])

            self.initial_state = self.stacked_cell.zero_state(batch_size, dtype=tf.float32)

            outputs, _ = tf.nn.dynamic_rnn(self.stacked_cell,
                                           self.inputs, initial_state=self.initial_state, dtype=tf.float32)

            self.output = outputs[:, -1, :]

class FModel(object):
    def __init__(self, is_training):
        if is_training:
            self.targets = tf.placeholder(tf.float32, [BATCH_SIZE, 1])


        # Build Model

        self.lstm_inputs = tf.placeholder(tf.float32, [BATCH_SIZE, NUM_STEPS, FEATURE_NUM])

        with tf.variable_scope("air_lstm") as scope:
            self.lstm = LSTM_model(self.lstm_inputs)

        with tf.variable_scope("fc"):
            fc_output = fc(self.lstm.output)

            self.results = predict_layer(fc_output)

        if not is_training:
            return

        self.mse_losses = tf.square(tf.subtract(self.targets, self.results))
        self.losses = self.mse_losses

        self.cost_pure = tf.div(tf.reduce_sum(self.losses), BATCH_SIZE)
        self.cost = self.cost_pure


        # self.train_op = tf.contrib.layers.optimize_loss(
        #     loss, tf.train.get_global_step(), optimizer="Adam", learning_rate=0.01)


        self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
        self.merged_summary = tf.summary.merge_all()


# inputs shape is [batch_size, hidden_size]
# output shape is [batch_size, label_size]
def predict_layer(inputs):
    with tf.variable_scope("predict_layer"):
        inputs_length = inputs.get_shape()[1]
        inputs_size = inputs.get_shape()[0]
        weight = tf.get_variable("vector_weight", [inputs_length, len(Labels)])
        bias = tf.get_variable("bias", [len(Labels), ])
        # outputs = tf.tensordot(inputs, weight, [[1], [0]])
        outputs = tf.matmul(inputs, weight)
        outputs = tf.add(outputs, bias)
        outputs = tf.nn.relu(outputs)
    return outputs

def fc(inputs, hidden_size = 100):
    with tf.variable_scope("location_fc"):
        # local locations will be changed after a long batch period in datasets
        location_fc_output = tf.contrib.layers.fully_connected(inputs, hidden_size)
    return location_fc_output
# Notice this will also return actuall count
def getBatchCount(total_count, batch_size = BATCH_SIZE):
    take_count = total_count - 1
    sequence_count = (take_count - batch_size + 1)
    batch_count = sequence_count // batch_size
    actuall_count = batch_count * batch_size

    return batch_count, actuall_count
# inputs will be shape [time_count, feature_number]
def generate_lstm_data(inputs, num_steps = NUM_STEPS, hasLabel = False, stop_before = 0, data_scalar=None):
    # params:
    # stop_before: stop sequence before stop_before days before sequence end
    columns = list(inputs)
    feature_num = len(columns)

    row_count = inputs.shape[0]



    # Notcie weather data longer than air quality two days
    # row_count = row_count - NUM_STEPS
    # inputs = inputs[:-NUM_STEPS]
    if stop_before != 0:
        row_count = row_count - stop_before
        inputs = inputs[:-stop_before]

    take_count = row_count - 1



    # Adjust to make Y one hour later than X inherently

    # The last time data will be one of our predictions, so will not include in our input_X
    # To make input X has proper shape, I start the index from n-1
    input_X = inputs.iloc[- take_count - 1: - 1]

    # we only need the label data to be our labels
    if hasLabel:
        data_Y = inputs.iloc[:]
    # predict data will be start after lstm feedforwad through the first num steps data
    # Y start will be one hour later than X to be our label
        Y = data_Y.iloc[- take_count:]

    # # Normalized Data
    if len(list(input_X)) == 9999:
        input_X = np.array(input_X).reshape(-1, 1)
    if data_scalar == None:
        data_scalar = StandardScaler()
        input_X = data_scalar.fit_transform(input_X)
    else:
        input_X = data_scalar.transform(input_X)

    #     Arrange X into sequence list
    sequence_X = []

    if hasLabel:
        sequence_Y = []

    sequence_count = row_count
    for i in range(take_count):
        if i > take_count - num_steps:
            sequence_count = i
            break
        sequence_X.append(input_X[i:i + num_steps])
        if hasLabel:
        # Y start already earlier than X one hour, so here we should minus 1, it will get the correspond Y for X
            sequence_Y.append(Y.values[i + num_steps - 1])

    # Make Batch
    # batch_size is the actuall training records' batch size
    batch_count, actuall_count = getBatchCount(sequence_count)
    # print(batch_count)

    # clip the margin data

    sequence_X = sequence_X[-actuall_count:]
    if hasLabel:
        sequence_Y = sequence_Y[-actuall_count:]

    X_batches = np.split(np.array(sequence_X), batch_count)
    if hasLabel:
        Y_batches = np.split(np.array(sequence_Y), batch_count)
    if hasLabel:
        return X_batches, Y_batches
    else:
        return  X_batches

EPOCH = 5

scaler = StandardScaler()
def main():
    with open("5013_dataset/feature_names.pkl", 'rb') as f:
        names = pickle.load(f)
    model = FModel(True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=100)
        for epoch in range(5):
            for i in range(36, 46):
                filename = "5013_dataset/price_feature_dataset_partial_" + str(i + 1) + '.pkl'
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                    targets = data[20]

                    data1 = data.iloc[:,55:66]
                    data2 = data.iloc[:, 99:110]
                    data = pd.concat([data1, data2], axis=1 )




                    X_batches = generate_lstm_data(data)
                    # targets = np.expand_dims(targets, 0)



                    _, Y_batches = generate_lstm_data(targets, hasLabel=True)

                    Y_batches = np.expand_dims(Y_batches, 2)


                    batch_count  = len(X_batches)
                    print(batch_count)
                    for j in range(EPOCH):
                        print("epoch ", str(j))
                        for batch_idx in range(batch_count):
                            cost, _, output, losses = sess.run(
                                [model.cost, model.train_op, model.results, model.losses],
                                {model.lstm_inputs: X_batches[batch_idx], model.targets: Y_batches[batch_idx]})

                    print(cost)
                    saver.save(sess, './my_model_' + str(epoch) + '_' + str(j))

def predict(eval_model, X_batch):



        model = eval_model
        train_op = tf.no_op()
        model_name = "./my_model_2_4"


        X_batch = np.tile(np.expand_dims(X_batch, 0), (BATCH_SIZE, 1, 1))




        with tf.Session() as sess:
            saver = tf.train.Saver()

            saver.restore(sess, model_name)

            _, output = sess.run(
                [train_op, model.results],
                {model.lstm_inputs: X_batch})

            result = output[-1]


        return result



