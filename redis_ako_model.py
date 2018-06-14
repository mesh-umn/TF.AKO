import tensorflow as tf

params = dict()


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def change_gradent_format(new_values, previous_grads):
    modified_grads = list()
    for v, ref in zip(new_values, previous_grads):
        modified_grads.append((v, ref[1]))
    return modified_grads

def init_parameters():
    # For parameter passing
    params["weights"] = dict()
    params["data"] = dict()
    params["gradient"] = list()
    params["new_g"] = dict()
    params["loss"] = list()
    params["optimizer"] = list()


def build_model(cfg):
    init_parameters()
    with tf.device("/job:%s/task:%d" % (cfg.job_name, cfg.nID)):
        with tf.variable_scope("%s%d" % (cfg.job_name, cfg.nID)):

            # Trainable variables
            W_conv1 = weight_variable([5, 5, 3, 32])
            b_conv1 = bias_variable([32])
            W_conv2 = weight_variable([3, 3, 32, 64])
            b_conv2 = bias_variable([64])
            W_conv3 = weight_variable([3, 3, 64, 64])
            b_conv3 = bias_variable([64])
            W_fc1 = weight_variable([8 * 8 * 64, 1024])
            b_fc1 = bias_variable([1024])
            W_fc2 = weight_variable([1024, 10])
            b_fc2 = bias_variable([10])

            # Model
            x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
            y = tf.placeholder(tf.float32, shape=[None, 10])
            new_g_W_conv1 = tf.placeholder(tf.float32, shape=[5, 5, 3, 32])
            new_g_b_conv1 = tf.placeholder(tf.float32, shape=[32])
            new_g_W_conv2 = tf.placeholder(tf.float32, shape=[3, 3, 32, 64])
            new_g_b_conv2 = tf.placeholder(tf.float32, shape=[64])
            new_g_W_conv3 = tf.placeholder(tf.float32, shape=[3, 3, 64, 64])
            new_g_b_conv3 = tf.placeholder(tf.float32, shape=[64])
            new_g_W_fc1 = tf.placeholder(tf.float32, shape=[8 * 8 * 64, 1024])
            new_g_b_fc1 = tf.placeholder(tf.float32, shape=[1024])
            new_g_W_fc2 = tf.placeholder(tf.float32, shape=[1024, 10])
            new_g_b_fc2 = tf.placeholder(tf.float32, shape=[10])

            layer2 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
            layer3 = max_pool_2x2(layer2)
            layer4 = tf.nn.relu(conv2d(layer3, W_conv2) + b_conv2)
            layer5 = tf.nn.relu(conv2d(layer4, W_conv3) + b_conv3)
            layer6 = max_pool_2x2(layer5)
            layer6_flat = tf.reshape(layer6, [-1, 8 * 8 * 64])
            layer7 = tf.nn.relu(tf.matmul(layer6_flat, W_fc1) + b_fc1)
            keep_prob = tf.placeholder(tf.float32)
            layer8 = tf.nn.dropout(layer7, keep_prob)
            layer9 = tf.matmul(layer8, W_fc2) + b_fc2

            # Loss function
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=layer9))
            optimizer = tf.train.AdamOptimizer(0.001)
            trainable_vars = [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2]
            grads = optimizer.compute_gradients(loss, trainable_vars)
            new_grads = [new_g_W_conv1, new_g_b_conv1, new_g_W_conv2, new_g_b_conv2, new_g_W_conv3,
                         new_g_b_conv3, new_g_W_fc1, new_g_b_fc1, new_g_W_fc2, new_g_b_fc2]
            modified_grads = change_gradent_format(new_grads, grads)
            train_op = optimizer.apply_gradients(modified_grads)

            # Accuracy
            softmax_y = tf.nn.softmax(layer9)
            correct_pred = tf.equal(tf.argmax(softmax_y, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            # For parameter passing
            params["weights"]["W_conv1"] = W_conv1
            params["weights"]["b_conv1"] = b_conv1
            params["weights"]["W_conv2"] = W_conv2
            params["weights"]["b_conv2"] = b_conv2
            params["weights"]["W_conv3"] = W_conv3
            params["weights"]["b_conv3"] = b_conv3
            params["weights"]["W_fc1"] = W_fc1
            params["weights"]["b_fc1"] = b_fc1
            params["weights"]["W_fc2"] = W_fc2
            params["weights"]["b_fc2"] = b_fc2
            params["new_g"]["W_conv1"] = new_g_W_conv1
            params["new_g"]["b_conv1"] = new_g_b_conv1
            params["new_g"]["W_conv2"] = new_g_W_conv2
            params["new_g"]["b_conv2"] = new_g_b_conv2
            params["new_g"]["W_conv3"] = new_g_W_conv3
            params["new_g"]["b_conv3"] = new_g_b_conv3
            params["new_g"]["W_fc1"] = new_g_W_fc1
            params["new_g"]["b_fc1"] = new_g_b_fc1
            params["new_g"]["W_fc2"] = new_g_W_fc2
            params["new_g"]["b_fc2"] = new_g_b_fc2
            params["data"]["x"] = x
            params["data"]["y"] = y
            params["keep_prob"] = keep_prob
            params["gradient"] = grads
            params["loss"] = loss
            params["optimizer"] = train_op
            params["softmax_y"] = softmax_y
            params["correct_pred"] = correct_pred
            params["accuracy"] = accuracy

    return params

