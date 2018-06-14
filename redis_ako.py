import os
import sys
import time

import numpy as np
import tensorflow as tf
from tflearn.data_utils import to_categorical
from tflearn.datasets import cifar10

import redis_ako_config
from redis_ako_cluster import build_cluster
from redis_ako_model import build_model
from redis_ako_queue import GradientExchange

# Application parameters
job_name = sys.argv[1]
nID = int(sys.argv[2])

# Execute in local machine
cfg = redis_ako_config.Config(job_name=job_name, nID=nID)

# Make a cluster, create queues, and build a model
cluster, server, workers, term_cmd = build_cluster(cfg)
params = build_model(cfg)

# Data loading
(x_image, Y), (X_test, Y_test) = cifar10.load_data()
y_test_vector = to_categorical(Y_test, 10)
y_vector = to_categorical(Y, 10)
y_features = to_categorical(np.arange(10), 10)
print "Image data: cifar10_asynch.load_data (50000)"

# Each nodes executes the following codes
with tf.Session("grpc://" + workers[nID]) as mySess:
    mySess.run(tf.global_variables_initializer())
    myQueue = GradientExchange(mySess, cfg)

    # Ensure all workers launch redis server and load data
    myQueue.send_ready()
    myQueue.check_all_ready()
    myQueue.receive_go_sign()

    if cfg.synchronous_training:
        if nID == 0:
            myQueue.set_pongs()

    # Model Training
    accuracies = list()
    elapsed_time = 0.0
    iteration = -1
    flag_stop_training = False

    # Train
    for i in range(cfg.training_epochs):
        print "*** epoch %d ***" % (i + 1)
        for j in range(cfg.num_batches):
            if (j % cfg.num_workers) == nID:
                start_time = time.time()
                iteration += 1
                idxfrom = j * cfg.batch_size
                idxto = idxfrom + cfg.batch_size

                # Calculate gradients
                _grads = mySess.run(params["gradient"],
                                    feed_dict={params["data"]["x"]: x_image[idxfrom:idxto],
                                               params["data"]["y"]: y_vector[idxfrom:idxto],
                                               params["keep_prob"]: 0.5})

                myQueue.enqueue(_grads, iteration)

                if myQueue.get_stop() == "True":
                    flag_stop_training = True
                    break

                if cfg.synchronous_training:
                    myQueue.receive_pong()

                total_grads = myQueue.get_others_grads()

                for w in range(len(cfg.weights)):
                    total_grads[w] = np.add(total_grads[w], _grads[w][0])

                _ = mySess.run(params["optimizer"],
                               feed_dict={params["new_g"]["W_conv1"]: total_grads[cfg.weights["W_conv1"]["wid"]],
                                          params["new_g"]["b_conv1"]: total_grads[cfg.weights["b_conv1"]["wid"]],
                                          params["new_g"]["W_conv2"]: total_grads[cfg.weights["W_conv2"]["wid"]],
                                          params["new_g"]["b_conv2"]: total_grads[cfg.weights["b_conv2"]["wid"]],
                                          params["new_g"]["W_conv3"]: total_grads[cfg.weights["W_conv3"]["wid"]],
                                          params["new_g"]["b_conv3"]: total_grads[cfg.weights["b_conv3"]["wid"]],
                                          params["new_g"]["W_fc1"]: total_grads[cfg.weights["W_fc1"]["wid"]],
                                          params["new_g"]["b_fc1"]: total_grads[cfg.weights["b_fc1"]["wid"]],
                                          params["new_g"]["W_fc2"]: total_grads[cfg.weights["W_fc2"]["wid"]],
                                          params["new_g"]["b_fc2"]: total_grads[cfg.weights["b_fc2"]["wid"]]})

                _loss = mySess.run(params["loss"],
                                   feed_dict={params["data"]["x"]: x_image[idxfrom:idxto],
                                              params["data"]["y"]: y_vector[idxfrom:idxto],
                                              params["keep_prob"]: 0.5})

                print "[Node ID: %d] iter: %d, loss: %f, batch %d - %d" % \
                      (nID, iteration, _loss, idxfrom, idxto)

                if cfg.testing:
                    if iteration == cfg.testing_iteration:
                        break

                elapsed_time += (time.time() - start_time)

                if cfg.train_until_fixed_accuracy:
                    if iteration % cfg.iteration_to_check_accuracy == 0:
                        test_accuracy = mySess.run(params["accuracy"],
                                                   feed_dict={params["data"]["x"]: X_test,
                                                              params["data"]["y"]: y_test_vector,
                                                              params["keep_prob"]: 1.0})
                        accuracies.append(test_accuracy)
                        print "[epoch %d][iter %d] Execution Time: %d seconds" % ((i+1), iteration, elapsed_time)
                        print "[epoch %d][iter %d] Test Accuracy %g" % ((i+1), iteration, test_accuracy)

                        if test_accuracy >= cfg.target_accuracy:
                            flag_stop_training = True
                            myQueue.set_stop()
                            break

                if elapsed_time >= cfg.stop_time:
                    flag_stop_training = True
                    myQueue.set_stop()
                    break
                else:
                    if myQueue.get_stop() == "True":
                        flag_stop_training = True
                        break

        if cfg.testing is False:
            test_accuracy = mySess.run(params["accuracy"],
                                     feed_dict={params["data"]["x"]: X_test,
                                                params["data"]["y"]: y_test_vector,
                                                params["keep_prob"]: 1.0})
            accuracies.append(test_accuracy)
            print "[epoch %d][iter %d] Execution Time: %d seconds" % ((i+1), iteration, elapsed_time)
            print "[epoch %d][iter %d] Test Accuracy %g" % ((i+1), iteration, test_accuracy)
            print "Total Test Accuracy"
            print accuracies

        if flag_stop_training:
            break

    # Terminate all threads
    myQueue.terminate_threads()

    # Ensure everybody finishes their tasks
    myQueue.send_ready()
    myQueue.check_all_ready()
    myQueue.receive_go_sign()

    # Stop redis-server
    os.system(term_cmd)

    print "Terminating server" + str(nID)

