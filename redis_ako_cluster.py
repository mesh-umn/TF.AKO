import tensorflow as tf
import subprocess

def build_cluster(cfg):
    # Cluster configuration
    workers = list()
    if cfg.remote is None:
        # Build a cluster in local machine
        for i in range(cfg.num_workers):
            ipport = cfg.local_ip + ":" + str(cfg.worker_port + i)
            workers.append(ipport)
    else:
        # Build a cluster in remote servers
        for i in range(cfg.num_workers):
            ipport = cfg.remote_ip[cfg.remote[i]] + ":" + str(cfg.worker_port)
            workers.append(ipport)

    cluster = tf.train.ClusterSpec({"wk": workers})
    server = tf.train.Server(cluster, job_name=cfg.job_name, task_index=cfg.nID)
    print("Starting server /job:{}/task:{}".format(cfg.job_name, cfg.nID))

    # Start Redis-server
    redis_start_cmd = "redis-server --port %s &" % str(cfg.redis_port + cfg.nID)
    redis_process = subprocess.Popen(redis_start_cmd, shell=True)
    term_cmd = "kill -9 %s" % str(redis_process.pid + 1)

    return cluster, server, workers, term_cmd

