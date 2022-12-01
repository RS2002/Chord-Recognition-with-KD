import numpy as np
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.framework import ops
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

class Student(object):
    def __init__(self,
                 T=3,
                 frequency_size=24,
                 segment_width=21,
                 n_steps=100,
                 n_classes=26,
                 initial_learning_rate=1e-4,
                 dropout_rate=0.5,
                 batch_size=60,
                 lambda_loss_c=1,
                 lambda_L2=1e-3,
                 training_steps=50000,
                 alpha=1.5,
                 beta=0.5
                 ):
        self.T=T #temprature
        self._frequency_size = frequency_size
        self._segment_width = segment_width
        self._feature_size = frequency_size * segment_width  # input size (feature size)
        self._n_steps = n_steps
        self._n_classes = n_classes
        self._session = None
        self._graph = None
        self._lambda_loss_c = lambda_loss_c
        self._lambda_L2 = lambda_L2
        self._dropout_rate = dropout_rate
        self._batch_size = batch_size
        self._initial_learning_rate = initial_learning_rate
        self._training_steps = training_steps
        self._alpha=alpha #soft
        self._beta=beta   #hard


    def load_data(self):
        fileDir = 'preprocessed_data/Billboard_data_mirex_Mm_model_input_final.npz'
        with np.load(fileDir, allow_pickle=True) as input_data:
            x_train = input_data['x_train']
            TC_train = input_data['TC_train']
            y_train = input_data['y_train']
            y_cc_train = input_data['y_cc_train']
            y_len_train = input_data['y_len_train']
            x_valid = input_data['x_valid']
            TC_valid = input_data['TC_valid']
            y_valid = input_data['y_valid']
            y_cc_valid = input_data['y_cc_valid']
            y_len_valid = input_data['y_len_valid']
            split_sets = input_data['split_sets']
        split_sets = split_sets.item()
        fileDir = 'preprocessed_data/soft_target.npz'
        with np.load(fileDir, allow_pickle=True) as input_data:
            logits_train = input_data['soft_target']
            logits_valid =input_data['soft_target_test']

        return x_train, TC_train, y_train, y_cc_train, y_len_train, \
               x_valid, TC_valid, y_valid, y_cc_valid, y_len_valid, \
               split_sets, logits_train, logits_valid

    def _normalize(self, inputs, epsilon=1e-8, scope="ln", reuse=tf.AUTO_REUSE):
        '''Applies layer normalization.'''
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
            gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta
        return outputs

    '''def DFN(self, inputs,  dropout_rate, is_training=True):
        n_hidden1 = 250
        n_hidden2 = 80

        inputs_reshape = tf.reshape(inputs, shape=[-1, self._frequency_size, self._segment_width])
        inputs_reshape = self._feedforward(inputs_reshape, n_units=[self._segment_width * 4, self._segment_width],dropout_rate=dropout_rate,is_training=is_training)  # [batch_size*n_steps, tonal_size, segment_width]
        #restore shape
        inputs_reshape = tf.reshape(inputs_reshape, shape=[-1, self._frequency_size * self._segment_width])  # [batch_size*n_steps, tonal_size*segment_width]

        inputs_drop = tf.layers.dropout(inputs_reshape, dropout_rate, training=is_training)
        hidden1 = self.FFN(inputs_drop, n_hidden1, name="hidden1", activation="relu")

        hidden1 = self._normalize(hidden1, scope='hidden1')

        hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=is_training)
        hidden2 = self.FFN(hidden1_drop, n_hidden2, name="hidden2", activation="relu")

        hidden2 = self._normalize(hidden2, scope='hidden2')

        hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=is_training)
        output = self.FFN(hidden2_drop, self._n_classes, name="output")

        output = self._normalize(output, scope='output')

        logits = output
        logits = tf.reshape(logits,shape=[-1, self._n_steps, self._n_classes])  # shape = [batch_size, n_steps, n_classes]
        chord_predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)  # shape = [batch_size, n_steps]
        return logits, chord_predictions'''

    def DFN(self, inputs, dropout_rate, is_training=None):
        n_hidden1 = 250
        n_hidden2 = 80
        inputs_reshape = tf.reshape(inputs, shape=[-1,self._frequency_size * self._segment_width])  # [batch_size*n_steps, tonal_size* segment_width]
        inputs_reshape = self._normalize(inputs_reshape,scope='input')

        #inputs_reshape = tf.reshape(inputs, shape=[-1, self._frequency_size, self._segment_width])
        #inputs_reshape = self._feedforward(inputs_reshape, n_units=[self._segment_width * 4, self._segment_width],dropout_rate=dropout_rate,is_training=is_training)  # [batch_size*n_steps, tonal_size, segment_width]
        # restore shape
        #inputs_reshape = tf.reshape(inputs_reshape, shape=[-1, self._frequency_size * self._segment_width])  # [batch_size*n_steps, tonal_size*segment_width]

        inputs_drop = tf.layers.dropout(inputs_reshape, dropout_rate, training=is_training)
        hidden1 = self.FFN(inputs_drop, n_hidden1, name="hidden1", activation="relu")

        hidden1 = self._normalize(hidden1,scope='hidden1')

        hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=is_training)
        hidden2 = self.FFN(hidden1_drop, n_hidden2, name="hidden2", activation="relu")

        hidden2 = self._normalize(hidden2,scope='hidden2')

        hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=is_training)
        output = self.FFN(hidden2_drop, self._n_classes, name="output")

        output = self._normalize(output,scope='output')

        logits = output
        logits = tf.reshape(logits,shape=[-1, self._n_steps, self._n_classes])  # shape = [batch_size, n_steps, n_classes]
        chord_predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)  # shape = [batch_size, n_steps]
        return logits, chord_predictions


    def FFN(self,inputs,n_outputs,activation=None,name="FFN"):
            with tf.name_scope(name):
                n_inputs = int(inputs.get_shape()[1])
                init = tf.truncated_normal((n_inputs, n_outputs), stddev=2/np.sqrt(n_inputs))
                W = tf.Variable(init, name="weights")
                b = tf.Variable(tf.zeros([n_outputs]), name="biases")
                z = tf.matmul(inputs, W) + b
                if activation=="relu":
                    z = tf.nn.relu(z)
                elif activation=='elu':
                    z = tf.nn.elu(z)
                elif activation=="sigmoid":
                    z = tf.nn.sigmoid(z)
                return z


    def _feedforward(self, inputs, n_units=[2048, 512], activation_function=tf.nn.relu, dropout_rate=0, is_training=True, scope="feedforward", reuse=None):
        '''Point-wise feed forward net.'''
        with tf.variable_scope(scope, reuse=reuse):
            # Inner layer
            params = {"inputs": inputs, "filters": n_units[0], "kernel_size": 1, "activation": activation_function, "use_bias": True}
            outputs = tf.layers.conv1d(**params)

            # Readout layer
            params = {"inputs": outputs, "filters": n_units[1], "kernel_size": 1, "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)

            # Dropout
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)

            # Residual connection
            outputs += inputs

            # Normalize
            outputs = self._normalize(outputs)

        return outputs



    def train(self, save_model=False, load_model=False, save_path="save/student/model.ckpt",load_path="save/student/model.ckpt", train=True, test=False, distill=True):
        if not distill:
            self._beta = 1
            self._alpha = 0
            self.T=1
        print("load input data...")
        x_train, TC_train, y_train, y_cc_train, y_len_train, \
        x_valid, TC_valid, y_valid, y_cc_valid, y_len_valid, \
        split_sets, logits_train, logits_valid = self.load_data()

        '''print(logits_train[0][0])
        print(np.sum(logits_train[0][0]))'''

        #logits_train_origin=logits_train.copy()

        logits_train/=self.T
        logits_train -= np.max(logits_train, axis=-1, keepdims=True)
        soft_target = np.exp(logits_train) / np.sum(np.exp(logits_train),axis=-1,keepdims=True)

        logits_valid /= self.T
        logits_valid -= np.max(logits_valid, axis=-1, keepdims=True)
        soft_target_valid = np.exp(logits_valid) / np.sum(np.exp(logits_valid), axis=-1, keepdims=True)

        num_examples_train = x_train.shape[0]

        '''print(soft_target[0][0])
        print(np.sum(soft_target[0][0]))
        print(y_train[0][0])'''

        # Define placeholders
        print("build model...")
        x = tf.placeholder(tf.float32, [None, self._n_steps, self._feature_size],name='encoder_inputs')  # shape = [batch_size, n_steps, n_inputs]
        y = tf.placeholder(tf.int32, [None, self._n_steps],name='chord_labels')  # ground_truth, shape = [batch_size, n_steps]
        y_soft = tf.placeholder(tf.float32, [None, self._n_steps, self._n_classes], name='soft_target')
        dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
        is_training = tf.placeholder(tf.bool, name='is_training')
        global_step = tf.placeholder(tf.int32, name='global_step')
        stochastic_tensor = tf.placeholder(tf.bool, name='stochastic_tensor')

        logits, chord_predictions = self.DFN(x, dropout_rate, is_training)

        # Define loss
        with tf.name_scope('loss'):
            loss_c = self._lambda_loss_c * tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y, depth=self._n_classes), logits=logits, label_smoothing=0.1)
            #loss_c_soft = self._lambda_loss_c * tf.losses.softmax_cross_entropy(onehot_labels=y_soft, logits=logits/self.T, label_smoothing=0.1)

            param=1
            loss_c_soft = self._lambda_loss_c * tf.losses.mean_squared_error(param*tf.nn.softmax(logits/self.T),param*y_soft)
            # loss_c_soft = self._lambda_loss_c * tf.losses.mean_squared_error(logits ,y_soft)

            # L2 norm regularization
            vars = tf.trainable_variables()
            L2_regularizer = self._lambda_L2 * tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name])
            # loss
            loss = self._alpha * loss_c_soft + self._beta* loss_c + L2_regularizer

        with tf.name_scope('optimization'):
            # apply learning rate decay
            learning_rate = tf.train.exponential_decay(learning_rate=self._initial_learning_rate,
                                                       global_step=global_step,
                                                       decay_steps=(x_train.shape[0] // self._batch_size),
                                                       decay_rate=0.96,
                                                       staircase=True)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                               #learning_rate=self._initial_learning_rate,
                                               beta1=0.9,
                                               beta2=0.98,
                                               epsilon=1e-9)

        # Apply gradient clipping
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) if grad is not None else (grad, var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)

        # Define accuracy
        with tf.name_scope('accuracy'):
            label_mask = tf.less(y, 24)  # where label != 24('X)' and label != 25('pad')
            correct_predictions = tf.equal(chord_predictions, y)
            correct_predictions_mask = tf.boolean_mask(tensor=correct_predictions, mask=label_mask)
            accuracy = tf.reduce_mean(tf.cast(correct_predictions_mask, tf.float32))
            #accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))




        # Training
        print('train the model...')
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if load_model:
                saver.restore(sess, load_path)
            if train:
                epoch = num_examples_train // self._batch_size  # steps per epoch
                for step in range(self._training_steps):

                    if step % epoch == 0:
                        # shuffle trianing set
                        indices = random.sample(range(num_examples_train), num_examples_train)
                        batch_indices = [indices[x:x + self._batch_size] for x in range(0, len(indices), self._batch_size)]

                    # training
                    batch = (x_train[batch_indices[step % len(batch_indices)]],
                             y_train[batch_indices[step % len(batch_indices)]],
                             soft_target[batch_indices[step % len(batch_indices)]])

                    '''batch = (x_train[batch_indices[step % len(batch_indices)]],
                             y_train[batch_indices[step % len(batch_indices)]],
                             logits_train_origin[batch_indices[step % len(batch_indices)]])'''



                    train_run_list = [train_op, loss, loss_c, loss_c_soft, L2_regularizer, chord_predictions, accuracy]
                    train_feed_fict = {x: batch[0],
                                       y: batch[1],
                                       y_soft: batch[2],
                                       dropout_rate: self._dropout_rate,
                                       is_training: True,
                                       global_step: step + 1,
                                       stochastic_tensor: True}
                    _, train_loss,  train_loss_c, train_loss_c_soft, train_L2, train_c_pred, train_acc = sess.run(train_run_list, feed_dict=train_feed_fict)

                    if step % (epoch // 2) == 0:
                        print("------ step %d: train_loss %.4f (c %.4f, c_soft %.4f, L2 %.4f), train_accuracy %.4f ------" % (step, train_loss, train_loss_c, train_loss_c_soft, train_L2, train_acc))
                        if save_model:
                            saver_path = saver.save(sess, save_path)
                            print("Model saved in file:", saver_path)

            if test:
                for i in range(1):
                    T1 = time.time()
                    test_run_list = [loss, loss_c, loss_c_soft, L2_regularizer, chord_predictions, accuracy]
                    test_feed_fict = {x: x_valid,
                                       y: y_valid,
                                       y_soft: soft_target_valid,
                                       dropout_rate: self._dropout_rate,
                                       is_training: False,
                                       global_step: self._training_steps,
                                       stochastic_tensor: True}
                    test_loss, test_loss_c, test_loss_c_soft, test_L2, test_c_pred, test_acc = sess.run(test_run_list, feed_dict=test_feed_fict)
                    T2 = time.time()
                    cost = T2 - T1
                    print("\n------ test: test_loss %.4f (c %.4f, c_soft %.4f, L2 %.4f), test_accuracy %.4f ------" % (test_loss, test_loss_c, test_loss_c_soft, test_L2, test_acc))
                    print("compute time: " + str(cost) + "s")


if __name__ == '__main__':
    model = Student()
    model.train(save_model=True,train=True,test=True,distill=True)
    #model.train(save_model=False, train=True, test=True, distill=True)