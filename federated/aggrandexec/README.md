
## Dashboard

We have integrated Tensorboad for the visualization of experiment results. To track the experiment with ```[log_path]``` (e.g., ```./federated/benchmark/logs/cifar10/0209_141336```), please try ```tensorboard --logdir=[log_path] --bind_all```, and all the results will be available at: ```http://[ip_of_coordinator]:6006/```.

## Logs and Metrics

Meanwhile, all logs are dumped to ```log_path``` (specified in the config file) on each node. 
```testing_perf``` locates at the master node under this path, and the user can load it with ```pickle``` to check the time-to-accuracy performance. The user can also check ```/benchmark/[job_name]_logging``` to see whether the job is moving on.
