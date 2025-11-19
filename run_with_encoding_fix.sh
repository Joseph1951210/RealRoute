#!/bin/bash
# 包装脚本：设置正确的编码环境变量后运行程序

# 设置所有编码相关的环境变量
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LC_CTYPE=en_US.UTF-8

# 确保所有 HTTP 相关的环境变量不包含非 ASCII 字符
# 清理可能包含非 ASCII 字符的环境变量
unset HTTP_PROXY
unset HTTPS_PROXY
unset http_proxy
unset https_proxy

# 运行 Python 脚本，传递所有参数
cd /common/home/mg1998/THUIR/DeepSieve
conda activate deepsieve
python "$@"

