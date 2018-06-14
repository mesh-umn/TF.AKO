class Config():
    def __init__(self, **kwargs):

        ####################################
        #### Basic Configurations ##########
        ####################################

        # job_name and nID are set by parameters of execution command
        self.job_name = kwargs.get("job_name", "wk")
        self.nID = kwargs.get("nID", 0)

        # When DL model is trained in local machine
        self.local_ip = "localhost"
        # When DL model is trained in remote machines,
        # Their IP addresses.
        self.remote_ip = ["111.222.333.444", "111.222.333.555", "111.222.333.666", "111.222.333.777", "111.222.333.888"]
        # When DL model is trained in remote machines,
        # Indices of the remote machines that you will use for training
        self.remote = [0, 2, 3]
        # When DL model is trained in local machine
        # self.remote = None


        # Number of machines you will use for training
        self.num_workers = 2
        # Workers' port number
        self.worker_port = 2222
        # Each worker has their own redis server.
        # Redis server's port number
        self.redis_port = 6380
        self.training_epochs = 10
        self.batch_size = 104
        # Number of batches per epoch
        self.num_batches = 480


        ####################################
        #### Ways to train a model #########
        ####################################

        # True : Synchronous DL
        # False : Asynchronous DL
        self.synchronous_training = False
        # True: train a model until it reaches to the fixed accuracy
        # False: train a model for a fixed training time
        self.train_until_fixed_accuracy = False
        # True: Testing mode: training for several iterations for testing purpose
        # False: No testing mode
        self.testing = False

        if self.train_until_fixed_accuracy:
            # self.train_until_fixed_accuracy = True
            # Train a model until it reaches to the fixed accuracy

            # Target accuracy
            self.target_accuracy = 0.4
            # Check model's accuracy every this iteration
            self.iteration_to_check_accuracy = 5
            # Stop training at this second even though it doesn't reach the target accuracy
            self.stop_time = 3000

        else:
            # self.train_until_fixed_accuracy = False
            # Train a model for a fixed training time

            # Traing a model for this seconds
            self.stop_time = 300

        if self.testing:
            # self.testing = True
            # Testing mode

            # Train a mode for this epoch & this iteration
            self.training_epochs = 1
            self.testing_iteration = 10



        ####################################
        #### Ako Specific configuations ####
        ####################################

        # P values of each workers
        self.p = kwargs.get("p", [4, 4])
        # Staleness bound (Number of iterations) for SSP(Stale Synchronization Parallel)
        # 0 : the strongest synchronization
        self.synch_max_diff = 0
        # When partitioning gradients,
        # False: Partition gradients layer-by-layer
        # True: Partition even a layer to multiple partitions
        self.fine_grained_partition = False
        # Number of threads used for receiving gradients from other workers
        self.num_dqthreads = 2


        # Trainable variables and its queue name and indices
        # It should be modified according to your model
        # wid is incremental
        # num_parts is number of partitions of the layer
        # shape is the shape of the layer
        # range is the indices of first dimension of the layer's partitions
        self.weights = dict()
        self.weights["W_conv1"] = {"wid": 0, "num_parts": 1, "shape": (5, 5, 3, 32), "range": [0, 5]}
        self.weights["b_conv1"] = {"wid": 1, "num_parts": 1, "shape": (32), "range": [0, 32]}
        self.weights["W_conv2"] = {"wid": 2, "num_parts": 1, "shape": (3, 3, 32, 64), "range": [0, 3]}
        self.weights["b_conv2"] = {"wid": 3, "num_parts": 1, "shape": (64), "range": [0, 5]}
        self.weights["W_conv3"] = {"wid": 4, "num_parts": 1, "shape": (3, 3, 64, 64), "range": [0, 3]}
        self.weights["b_conv3"] = {"wid": 5, "num_parts": 1, "shape": (64), "range": [0, 64]}
        self.weights["W_fc1"] = {"wid": 6, "num_parts": 4, "shape": (4096, 1024), "range": [0, 1024, 2048, 3072, 4096]}
        self.weights["b_fc1"] = {"wid": 7, "num_parts": 1, "shape": (1024), "range": [0, 1024]}
        self.weights["W_fc2"] = {"wid": 8, "num_parts": 1, "shape": (1024, 10), "range": [0, 1024]}
        self.weights["b_fc2"] = {"wid": 9, "num_parts": 1, "shape": (10), "range": [0, 10]}

        # For fine-grained partition
        # When self.fine_grained_partition = True

        # When self.weights has num_parts greater than 1, its subweights information should be stated here
        # wid is incremental
        # part is the indices of partitions of the layer
        # shape is the shape of the subweights
        self.subweights = dict()
        self.subweights["1@W_fc1"] = {"wid": 10, "part": 1, "shape": (1024, 1024)}
        self.subweights["2@W_fc1"] = {"wid": 11, "part": 2, "shape": (1024, 1024)}
        self.subweights["3@W_fc1"] = {"wid": 12, "part": 3, "shape": (1024, 1024)}
        self.subweights["4@W_fc1"] = {"wid": 13, "part": 4, "shape": (1024, 1024)}


        # State how partitions of gradients are divided as per the P value
        if self.fine_grained_partition:
            # For fine-grained partition
            # When self.fine_grained_partition = True
            self.partitions = dict()
            self.partitions[1] = [["W_conv1", "b_conv1", "W_conv2", "b_conv2", "W_conv3", "b_conv3", "W_fc1", "b_fc1", "W_fc2", "b_fc2"]]
            self.partitions[2] = [["W_conv1", "b_conv1", "W_conv2", "b_conv2", "1@W_fc1", "2@W_fc1", "b_fc1"], ["W_conv3", "b_conv3", "W_fc2", "b_fc2", "3@W_fc1", "4@W_fc1"]]
            self.partitions[4] = [["W_conv1", "b_conv1", "1@W_fc1", "b_fc1"], ["W_conv2", "b_conv2", "2@W_fc1"], ["W_conv3", "b_conv3", "3@W_fc1"], ["W_fc2", "b_fc2", "4@W_fc1"]]

        else:
            # For layer-by-layer partition
            # When self.fine_grained_partition = True
            self.partitions = dict()
            self.partitions[1] = [["W_conv1", "b_conv1", "W_conv2", "b_conv2", "W_conv3", "b_conv3", "W_fc1", "b_fc1", "W_fc2", "b_fc2"]]
            self.partitions[2] = [["W_conv1", "b_conv1", "W_conv2", "b_conv2", "W_conv3", "b_conv3", "W_fc2", "b_fc2"], ["W_fc1", "b_fc1"]]
            self.partitions[4] = [["W_conv1", "b_conv1", "W_fc2", "b_fc2"], ["W_conv2", "b_conv2"], ["W_conv3", "b_conv3"], ["W_fc1", "b_fc1"]]
