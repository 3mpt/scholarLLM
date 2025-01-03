#!/bin/bash

# 设置远程服务器的地址
REMOTE_SERVER="root@ssh.intern-ai.org.cn"
PORT=47089

# 创建批量转发的端口映射
ssh -CNg -L 7474:127.0.0.1:7474 -p $PORT $REMOTE_SERVER &
ssh -CNg -L 7687:127.0.0.1:7687 -p $PORT $REMOTE_SERVER &
# 可以在这里添加更多的端口映射
# ssh -CNg -L <local_port>:<remote_host>:<remote_port> -p $PORT $REMOTE_SERVER &

# 等待所有后台进程结束
wait
