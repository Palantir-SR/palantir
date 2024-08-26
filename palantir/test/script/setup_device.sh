#!/bin/bash

source ${PALANTIR_CODE_ROOT}/palantir/script/common_funcs.sh

function _usage()
{
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-c CONTENT] [-q QUALITY] [-r INPUT_RESOLUTION] [-a ALGORITHM] [-d DEVICE_ID] [-t TRAIN_METHOD]
EOF
}

function _set_conda(){
    # >>> conda initialize >>>
    # !! Contents within this block are managed by 'conda init' !!
    __conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
            . "/opt/conda/etc/profile.d/conda.sh"
        else
            export PATH="/opt/conda/bin:$PATH"
        fi
    fi
    unset __conda_setup
    # <<< conda initialize <<<
    conda deactivate
    conda activate palantir_py3.6
}

[[ ($# -ge 1)  ]] || { echo "[ERROR] Invalid number of arguments. See -h for help."; exit 1;  }

while getopts ":c:q:r:a:d:l:n:o:C:p:i:W:H:h" opt; do
    case $opt in
        h) _usage; exit 0;;
        a) algorithm="$OPTARG";;
        c) content=("$OPTARG");;
        q) quality=("$OPTARG");;
        r) input_resolution=("$OPTARG");;
        d) device_id="$OPTARG";;
        l) limit="$OPTARG";;
        o) output_resolution="$OPTARG";;
        C) checkpoint_ready_flag="$OPTARG";;
        p) profile_name="$OPTARG";;
        i) chunk_idx="$OPTARG";;
        W) patch_width="$OPTARG";;
        H) patch_height="$OPTARG";;
        \?) exit 1;
    esac
done

if [ -z "${content+x}" ]; then
    echo "[ERROR] content is not set"
    exit 1;
fi

if [ -z "${patch_width+x}" ]; then
    echo "[WARNING] patch_width is not set; the default value (0) is used"
    patch_width=0
fi

if [ -z "${patch_height+x}" ]; then
    echo "[WARNING] patch_height is not set; the default value (0) is used"
    patch_height=0
fi

if [ -z "${algorithm+x}" ]; then
    echo "[ERROR] algorithm is not set"
    exit 1;
fi

if [ -z "${device_id+x}" ]; then
    echo "[ERROR] device_id is not set"
    exit 1;
fi

if [ -z "${quality+x}" ]; then
    echo "[ERROR] quality is not set"
    exit 1;
fi

if [ -z "${input_resolution+x}" ]; then
    echo "[ERROR] input_resolution is not set"
    exit 1;
fi

if [ -z "${train_method+x}" ]; then
    echo "[WARNING] train_method is not set, so the default value ('anchor_block_sr') is used"
    train_method="anchor_block_sr"
fi

if [ -z "${output_resolution+x}" ]; then
    echo "[ERROR] output_resolution is not set"
    exit 1;
fi

if [ -z "${profile_name+x}" ]; then
    echo "[ERROR] profile_name is not set"
    exit 1;
fi

py_checkpoint_ready_flag=""
if [ ! -z "${checkpoint_ready_flag+x}" ]; then
    py_checkpoint_ready_flag="--checkpoint_ready"
fi

limit_flag=""
if [ ! -z "${limit+x}" ]; then
    limit_flag="--limit ${limit}"
fi

chunk_idx_flag=""
if [ ! -z "${chunk_idx+x}" ]; then
    chunk_idx_flag="--chunk_idx ${chunk_idx}"
fi

_set_conda
_set_output_size ${output_resolution}
_set_bitrate ${input_resolution}
_set_num_blocks ${input_resolution} ${quality}
_set_num_filters ${input_resolution} ${quality}
_set_lr_video_name
CUDA_VISIBLE_DEVICES=${gpu_index} python ${PALANTIR_CODE_ROOT}/palantir/test/setup_device.py --data_dir ${PALANTIR_DATA_ROOT} --content ${content} --video_name ${lr_video_name} --lib_dir ${PALANTIR_CODE_ROOT}/palantir/test/libs/arm64-v8a --num_blocks ${num_blocks} --num_filters ${num_filters} --algorithm ${algorithm} --device_id ${device_id} --patch_width ${patch_width} --patch_height ${patch_height}  --output_width ${output_width} --output_height ${output_height} ${py_checkpoint_ready_flag} ${chunk_idx_flag} --profile_name ${profile_name} ${limit_flag}