

# 集群连接方式
张家口可用区的horovod-dev001~horovod-dev003。 登录方式:开启horovod-dev001后, ssh 公网ip，密码:(). 利用horovod-dev001当做跳板机可以直接ssh连接至其他节点.

# 测试数据说明
测试数据按照时间整理，每一天的所有测试数据放置在同一文件夹下。

如`20-12-08`文件夹下放置为2020年12月8日的测试数据。

在日期文件夹下，每次测试为一个文件夹。

文件夹下名如vgg16-19，代表测试的模型为vgg16，测试的时间为当天的19点(大体时间)。

文件夹下名如vgg16-09-fusion128-cycle20, 则代表--fusion-threshold-mb 参数设置为128， --cycle-time-ms 参数设置为20

# 默认测试参数
如测试文件夹内无特殊说明，则默认训练的epoch为10，batch为10，batch_size为32

# 集群间延迟
节点间的延迟如下,第一列为发送ping命令的节点rank:

```
0 1  avg_time=0.149ms
0 2  avg_time0.158ms
0 3  avg_time0.134ms

1 0  avg_time0.135ms
1 2  avg_time0.148ms
1 3  avg_time0.122ms

2 0  avg_time0.154ms
2 1  avg_time0.147ms
2 3  avg_time0.143ms

3 0  avg_time0.130ms
3 1  avg_time0.131ms
3 2  avg_time0.140ms
```



