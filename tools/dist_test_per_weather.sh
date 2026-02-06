#!/usr/bin/env bash

# CONFIG=$1
# CHECKPOINT=$2
GPUS=$1
IDX=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-65332}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
OUTPUT_FILE="test_per_weather_output.log"

# 定义配置文件路径
declare -A CONFIG_FILES=(
    [dense_fog]="configs/lidarweather_minkunet/sj+lpd+minkunet_semantickitti2semanticstf_dense_fog.py"
    [light_fog]="configs/lidarweather_minkunet/sj+lpd+minkunet_semantickitti2semanticstf_light_fog.py"
    [rain]="configs/lidarweather_minkunet/sj+lpd+minkunet_semantickitti2semanticstf_rain.py"
    [snow]="configs/lidarweather_minkunet/sj+lpd+minkunet_semantickitti2semanticstf_snow.py"
)

# 函数：运行测试并保存输出
run_test() {
    local config_file=$1
    shift
    (
        PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
        CUDA_VISIBLE_DEVICES=$IDX \
        python -m torch.distributed.launch \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --nproc_per_node=$GPUS \
            --master_port=$PORT \
            $(dirname "$0")/test.py \
            $config_file \
            # --tta \
            --launcher pytorch \
            ${@:3}
    ) 2>&1 | tee /dev/tty | 
    awk '/mmengine - WARNING - The prefix is not set in metric class SegMetric./{flag=1; next} /\/home\/dongsh\/anaconda3\/envs\/lidar_weather\//{flag=0} flag' >> $OUTPUT_FILE
}

# 执行测试
for config_key in "${!CONFIG_FILES[@]}"; do
    echo "Running test with ${config_key} configuration..." | tee -a $OUTPUT_FILE
    run_test "${CONFIG_FILES[$config_key]}"
done