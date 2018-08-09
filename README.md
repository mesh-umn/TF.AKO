# Decentralized Distributed Deep Learning (DL) in TensorFlow

This is a TensorFlow implementation of [Ako](https://lsds.doc.ic.ac.uk/sites/default/files/ako-socc16.pdf) (Ako: Decentralised Deep Learning with Partial Gradient Exchange). You can train any DNNs in a decentralized manner without parameter servers. Workers exchange partitioned gradients directly with each other without help of parameter servers and update their own local weights. Please refer the original paper [Ako](https://lsds.doc.ic.ac.uk/sites/default/files/ako-socc16.pdf) or our [project home](https://www-users.cs.umn.edu/~chandra/tfako/home.html) for more details. 

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

