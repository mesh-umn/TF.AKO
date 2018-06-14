import numpy as np
import redis
import threading
import copy
import time

others_grads = list()
cnt_msgs_received = list()
cur_iter = 0


class Clock_thread(threading.Thread):
    def __init__(self, msgQs, ping, num_workers, synch_max_diff):
        threading.Thread.__init__(self)
        self.msgQs = msgQs
        self.ping_pubsub = ping
        self.num_workers = num_workers
        self.synch_max_diff = synch_max_diff
        self.clocks = [0] * self.num_workers
        self.min_step = 0

    def receive_ping(self, item):
        sender = int(item["data"])
        self.clocks[sender] += 1
        new_min_step = np.min(self.clocks)
        if self.min_step < new_min_step:
            self.min_step = new_min_step
            self.send_pong()

    def send_pong(self):
        for i in range(self.num_workers):
            self.msgQs[i].publish("pong", "pong#")

    def run(self):
        for item in self.ping_pubsub.listen():
            if type(item["data"]) is not long:
                if item["channel"] == "done":
                    self.ping_pubsub.unsubscribe()
                    break
                else:
                    self.receive_ping(item)

# Dequeue threading
class Dequeue_thread(threading.Thread):
    def __init__(self, msgQs, channels, cmr_init, cfg):
        threading.Thread.__init__(self)
        self.redis = msgQs[cfg.nID]
        self.ping_channel = msgQs[0]
        self.channels = channels
        self.cmr_init = cmr_init
        self.cfg = cfg
        self.pubsub = self.redis.pubsub()
        self.pubsub.subscribe(self.channels)

    def count_msgs(self, wid):
        global cnt_msgs_received
        global cur_iter

        lidx = cur_iter % max(self.cfg.p)
        cnt_msgs_received[lidx][wid] -= 1
        if np.sum(cnt_msgs_received[lidx]) <= 0:
            cur_iter += 1
            cnt_msgs_received[lidx] = np.add(cnt_msgs_received[lidx], self.cmr_init[lidx])
            self.ping_channel.publish("ping", str(self.cfg.nID))

    def work(self, item):
        global others_grads

        key = item["channel"]
        keyinfo = key.split("@")
        if len(keyinfo) == 1:
            # normal weights
            # e.g. = keyinfo = ["W_conv1"] = [weight_name]
            wid = self.cfg.weights[key]["wid"]
            data = np.fromstring(item["data"], dtype="float32").reshape(self.cfg.weights[key]["shape"])
            others_grads[wid] = np.add(others_grads[wid], data)
            if self.cfg.synchronous_training:
                self.count_msgs(self.cfg.weights[key]["wid"])
        else:
            # fine-grained weights
            # e.g. = keyinfo = ["1", "W_conv1"] = [part#, weight_name]
            part = int(keyinfo[0])
            parent = keyinfo[1]
            wid = self.cfg.weights[parent]["wid"]
            fromidx = self.cfg.weights[parent]["range"][part - 1]
            toidx = self.cfg.weights[parent]["range"][part]
            subdata = np.fromstring(item["data"], dtype="float32").reshape(self.cfg.subweights[key]["shape"])
            data = np.zeros(self.cfg.weights[parent]["shape"], dtype="float32")
            data[fromidx:toidx] = subdata
            others_grads[wid] = np.add(others_grads[wid], data)
            if self.cfg.synchronous_training:
                self.count_msgs(self.cfg.subweights[key]["wid"])

    def run(self):
        for item in self.pubsub.listen():
            if type(item["data"]) is not long:
                if item["channel"] == "done":
                    self.pubsub.unsubscribe()
                    break
                else:
                    self.work(item)


class GradientExchange:
    def __init__(self, mySess, cfg):
        self.mySess = mySess
        self.cfg = cfg
        self.keys_weights = self.cfg.weights.keys()
        self.keys_subweights = self.cfg.subweights.keys()
        self.num_weights = len(self.keys_weights)
        self.prev_grads = list()
        self.accum_grads = [None] * self.num_weights
        self.cmr_init = list()
        self.msgQs = []
        self.ping = None
        self.pong = None
        self.ready = None
        self.go = None
        self.clockThread = None
        self.threads = list()
        self.init_grads_related_variables()
        self.init_cnt_msgs_received()
        self.init_msgQs_N_synch_channels()
        self.start_threads()
        time.sleep(3)

    def init_grads_related_variables(self):
        global others_grads
        # Initialize accumulated gradients (accum_grads) & others gradients variables (others_grads)
        others_grads = [None] * self.num_weights
        for key in self.keys_weights:
            wid = self.cfg.weights[key]["wid"]
            shape = self.cfg.weights[key]["shape"]
            self.accum_grads[wid] = np.zeros(shape, dtype="float32")
            others_grads[wid] = np.zeros(shape, dtype="float32")

        # Initialize previous gradients variable (prev_grads)
        for pi in range(self.cfg.p[self.cfg.nID]):
            self.prev_grads.append([None] * self.num_weights)
            for key in self.keys_weights:
                self.prev_grads[pi][self.cfg.weights[key]["wid"]] = np.zeros(self.cfg.weights[key]["shape"], dtype="float32")

    def init_cnt_msgs_received(self):
        global cnt_msgs_received
        # Setup their own initial values on cnt_msgs_received
        # cnt_msgs_received = max(p) X len(all_topics)
        all_topics = self.cfg.weights.keys() + self.cfg.subweights.keys()
        max_p = max(self.cfg.p)
        for i in range(max_p):
            self.cmr_init.append([0] * len(all_topics))
        for i in range(self.cfg.num_workers):
            if i is not self.cfg.nID:
                other_p = self.cfg.p[i]
                for j in range(max_p):
                    other_topic = self.cfg.partitions[other_p][j % other_p]
                    for t in other_topic:
                        if t in self.cfg.weights:
                            wid = self.cfg.weights[t]["wid"]
                        else:
                            wid = self.cfg.subweights[t]["wid"]
                        self.cmr_init[j][wid] += 1

        self.cmr_init = np.asarray(self.cmr_init)
        cnt_msgs_received = copy.deepcopy(self.cmr_init)

    def init_msgQs_N_synch_channels(self):
        for q in range(self.cfg.num_workers):
            if self.cfg.remote is None:
                self.msgQs.append(redis.Redis(host="localhost", port=self.cfg.redis_port + q))
            else:
                self.msgQs.append(redis.Redis(host=self.cfg.remote_ip[self.cfg.remote[q]],
                                              port=self.cfg.redis_port + q))
        for q in range(self.cfg.num_workers):
            self.msgQs[q].set("stop", "False")

        # Increase output buffer limit of Redis Pub/Sub
        self.msgQs[self.cfg.nID].config_set("client-output-buffer-limit", "normal 0 0 0 slave 268435456 67108864 60 pubsub 0 0 0")
        print self.msgQs[self.cfg.nID].config_get("client-output-buffer-limit")

        # Create ping/pong synchronization channels
        # Only worker 0 (chef node) has ping & ready channels
        if self.cfg.nID == 0:
            self.ping = self.msgQs[self.cfg.nID].pubsub()
            self.ping.subscribe(["ping", "done"])

            self.ready = self.msgQs[self.cfg.nID].pubsub()
            self.ready.subscribe("ready")

        # Every worker has pong & go channels
        self.pong = self.msgQs[self.cfg.nID].pubsub()
        self.pong.subscribe("pong")

        self.go = self.msgQs[self.cfg.nID].pubsub()
        self.go.subscribe("go")


    def start_threads(self):
        # Start required threads
        if self.cfg.nID == 0:
            # Start clock thread
            self.clockThread = Clock_thread(self.msgQs, self.ping, self.cfg.num_workers, self.cfg.synch_max_diff)
            self.clockThread.start()

        # Start dequeue threads
        channels = self.keys_weights + self.keys_subweights
        for i in range(self.cfg.num_dqthreads):
            topics = list()
            for idx, ch in enumerate(channels):
                if idx % self.cfg.num_dqthreads == i:
                    topics.append(ch)
            topics.append("done")
            dqThread = Dequeue_thread(self.msgQs, topics, self.cmr_init, self.cfg)
            dqThread.start()
            self.threads.append(dqThread)

    def set_pongs(self):
        # Allow asynchrony to some extent
        for i in range(self.cfg.num_workers):
            for j in range(self.cfg.synch_max_diff):
                self.msgQs[i].publish("pong", "pong" + str(j))

    def receive_pong(self):
        for item in self.pong.listen():
            if type(item["data"]) is not long:
                break

    # Ready/Go for worker synchronization
    def send_ready(self):
        self.msgQs[0].publish("ready", str(self.cfg.nID))

    def check_all_ready(self):
        if self.cfg.nID == 0:
            cnt = 0
            for item in self.ready.listen():
                if type(item["data"]) is not long:
                    cnt += 1
                    if cnt == self.cfg.num_workers:
                        for i in range(self.cfg.num_workers):
                            self.msgQs[i].publish("go", "go")
                        break

    def receive_go_sign(self):
        for item in self.go.listen():
            if type(item["data"]) is not long:
                break

    def set_stop(self):
        for q in range(self.cfg.num_workers):
            self.msgQs[q].set("stop", "True")

    def get_stop(self):
        return self.msgQs[self.cfg.nID].get("stop")

    def terminate_threads(self):
        for q in range(self.cfg.num_workers):
            self.msgQs[q].publish("done", "done")
        if self.cfg.nID == 0:
            self.msgQs[0].publish("done", "done")
            self.clockThread.join()
        for t in self.threads:
            t.join()

    def get_others_grads(self):
        global others_grads
        curr_others_grads = others_grads
        others_grads = np.subtract(others_grads, curr_others_grads)
        return curr_others_grads

    def toString(self, data):
        return data.ravel().tostring()

    def enqueue(self, _grads, iteration):
        # Accumulate p previous grads
        pidx = iteration % self.cfg.p[self.cfg.nID]
        for i in range(self.num_weights):
            self.accum_grads[i] = np.subtract(self.accum_grads[i], self.prev_grads[pidx][i])
            self.prev_grads[pidx][i] = _grads[i][0]
            self.accum_grads[i] = np.add(self.accum_grads[i], self.prev_grads[pidx][i])

        # Get partition infomation
        sub_channels = self.cfg.partitions[self.cfg.p[self.cfg.nID]][pidx]

        # Enqueue data
        for key in sub_channels:
            keyinfo = key.split("@")
            if len(keyinfo) == 1:
                # normal weights
                # e.g. = keyinfo = ["W_conv1"] = [weight_name]
                data = self.accum_grads[self.cfg.weights[key]["wid"]]
            else:
                # fine-grained weights
                # e.g. = keyinfo = ["1", "W_conv1"] = [part#, weight_name]
                part = int(keyinfo[0])
                parent = keyinfo[1]
                fromidx = self.cfg.weights[parent]["range"][part - 1]
                toidx = self.cfg.weights[parent]["range"][part]
                data = self.accum_grads[self.cfg.weights[parent]["wid"]][fromidx:toidx]

            for q in range(self.cfg.num_workers):
                if q is not self.cfg.nID:
                    self.msgQs[q].publish(key, self.toString(data))
