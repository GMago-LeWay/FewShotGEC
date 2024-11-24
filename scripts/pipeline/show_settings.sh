#!/bin/bash
if [ $# -lt 1 ]; then
    echo "Usage: $0 VAR1=VALUE1 VAR2=VALUE2 ..."
    exit 1
fi

# 遍历所有的参数
for var in "$@"; do
    name=$(echo $var | cut -d '=' -f 1)
    value=$(echo $var | cut -d '=' -f 2)
    echo "$name=$value"
done