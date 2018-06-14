# Decentralized Distributed Deep Learning (DL) in TensorFlow

This is a TensorFlow implementation of [Ako](https://lsds.doc.ic.ac.uk/sites/default/files/ako-socc16.pdf) (Ako: Decentralised Deep Learning with Partial Gradient Exchange). You can train any DNNs in a decentralized manner without parameter servers. Workers exchange partitioned gradients directly with each other without help of parameter servers and update their own local weights. Please refer [Ako](https://lsds.doc.ic.ac.uk/sites/default/files/ako-socc16.pdf) paper for more details. 

### Installation
 - Environments  
    - ubuntu 16.04 
    - Python 2.7
    - Tensorflow 1.4
    
 - Prerequisites 
    - redis-server & redis-client 
    - tflearn (only for loading CIFAR10 dataset)
        ```sh
        $ sudo apt-get update
        $ sudo apt-get install redis-server -y
        $ sudo pip install redis
        $ sudo pip install tflearn
        ```

### How to run
1. Build your model in **redis_ako_model.py**
2. Write your session and load your dataset in **redis_ako.py**
3. Change your configurations in **redis_ako_config.py**
    - Basic configurations: Cluster IP/Port, Redis port, Synchronous training, Training epochs, Batch size, Number of batches
    - Ways to train models: training a few iterations, training for a fixed time, training until a fixed accuracy 
    - Ako specific configurations: P values, partition details, SSP interation bound, Number of queue threads
4. Execute it 
    ```sh
    # When 3 workers are clustered and used for decentralized DL
    # At worker 0
    $ python redis_ako.py wk 0 
    # At worker 1
    $ python redis_ako.py wk 1
    # At worker 2
    $ python redis_ako.py wk 2
    ```


### Setting for experiments
 - Comparison between Centralized DL and Decentralized DL
 
      <img src="/img/centralizedDL.JPG" width="360">
      
      - **Centralized DL**: There are parameter servers to maintain and update model weights. Workers get the latest weights from parameter servers and calculate gradients locally and send them to parameter servers.    
       
      <img src="/img/decentralizedDL.JPG" width="400">

     - **Decentralized DL**: There are only workers. Parameter servers do not exist in this setting. Workers maintain and update model weights locally, calculate and share their gradients with other workers. 
     
 - DL Model: naive 3 convolutional + 2 fully-connected layers
 
     <img src="/img/DNmodel.JPG" width="400">
  
 - Dataset: CIFAR10
    - 10 classes
    - 32 by 32 color images
    - 50,000 training images and 10,000 test images
    
 - Testbed: 5 local servers 
    - 6 Intel Xeon CPU E5-2620 v3 CPUs /server
    - 45G memory /server
    - 1Gbps ethernet card /server
    
 - Metrics
    - **Training time (= execution time)**: the elapsed time in seconds from training start to training stop
    - **Accuracy**: the percentage of the number of test images correctly predicted over 10,000 test images
    
 - LAN & WAN networks emulation
    - **Uniform LAN**: every link among workers has 800 Mbps
    - **Heterogeneous LAN**: 800/400/200 Mbps are assigned to links
    - **Uniform WAN**: every link among workers has 40 Mbps
    - **Heterogenous WAN**: 40/20/10 Mbps are assigned to links
    
    
### Experiment results
 - Centralized DL vs Decentralized DL training for 300 seconds in various network environments
     <img src="/img/experimentLAN.JPG" width="600">
     
     <img src="/img/experimentWAN.JPG" width="600">
     
     - As the number of workers increase, the accuracy slightly decreases because the amount of training data unseen every iteration is proportional to the number of worker. 
     - When network resource is enough to exchange gradients (uniform LAN case), Centralized DL and Decentralized DL get similar accuracy. Since network bandwidth is much greater than the size of gradients, both ways can get their best accuracies. 
     - When network resource is not enough (other network settings), Decentralized DL with partial gradient exchange (P > 1) outperforms Centralized DL in most cases. Centralized DL suffers from network bottleneck issue at parameter servers since all workers sent their whole gradients to parameter servers every iteration. On the other hand, Decentralized DL can reduce the data size exchanged among workers every iteration by partitioning the gradients. 
     - Partial gradient exchange works well in such limited network bandwidth cases. The setting with larger P values leads the best accuracy in those network settings. Partial gradient exchange does not affect the accuracy, but drametically reduce the network trasmission time. 
     - Decentralized DL with various P values (each worker has different P value) obtain the best or similar accuracy in most cases. However, Decentralized DL with P = 1 (all-to-all case: exchanging whole gradients) shows the worst accuracy in most cases. 
  
 - Centralized DL vs Decentralized DL training for 300 seconds in Uniform LAN & WAN
     <img src="/img/LANvsWAN.JPG" width="600">
    
     - When network resource is enough to exchange gradients (LAN cases), Decentralized DL requires extra iterations because the efficiency of single iteration is less than the one of Centralized DL due to the partial gradient exchange. 
     - When network resource is not enough (WAN cases), Centralized DL required more iterations than Decentralized DL because Centralized DL acts like a all-to-all case (P = 1) of Decentralized DL due to scarse network capacity especially at parameter servers.
     - When network resource is scarse, Decentralized DL with large P value reaches a certain accuracy faster than Centralized DL. Since Decentralized DL requires less network bandwidth, it iterates much faster.
     - Accuracy flutuation over time of Centralized DL is greater than Decentralized. Since workers of Centralized DL get synchronized weight values from parameter servers every iteration, it gives workers a room to explore wider gradient space. On the other hand, workers of Decentralized DL do not have any weight synchronization among worker. They locally update their weight values based on gradients received from other workers. It causes smoother accuracy change over time. 
    
### Future work
We will implement another decentralized distributed deep learning system by using ***Horovod*** (https://github.com/uber/horovod) instead of using redis to share gradients among workers.
