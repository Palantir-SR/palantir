#!/bin/bash

source ${PALANTIR_CODE_ROOT}/palantir/script/common_funcs.sh

function _usage()
{
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-g GPU_INDEX] [-c CONTENT] [-q QUALITY] [-i INPUT_RESOLUTION] [-a ALGORITHM] [-o OUTPUT_RESOLUTION] [-W PATCH_WIDTH] [-H PATCH_HEIGHT] [-m MAXIMUM_ANCHOR] [-n NUM_CHUNKS]
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

while getopts ":a:c:q:i:o:W:H:m:n:h" opt; do
    case $opt in
        h) _usage; exit 0;;
        a) algorithm="$OPTARG";;
        c) content="$OPTARG";;
        q) quality="$OPTARG";;
        i) input_resolution="$OPTARG";;
        o) output_resolution="$OPTARG";;
        W) patch_width="$OPTARG";;
        H) patch_height="$OPTARG";;
        m) maximum_anchor="$OPTARG";;
        n) num_chunks="$OPTARG";;
        \?) exit 1;
    esac
done

if [ -z "${content+x}" ]; then
    echo "[ERROR] content is not set"
    exit 1;
fi

if [ -z "${algorithm+x}" ]; then
    echo "[ERROR] algorithm is not set"
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

if [ -z "${output_resolution+x}" ]; then
    echo "[ERROR] output_resolution is not set"
    exit 1;
fi

if [ -z "${patch_width+x}" ]; then
    echo "[ERROR] patch_width is not set"
    exit 1;
fi

if [ -z "${patch_height+x}" ]; then
    echo "[ERROR] patch_height is not set"
    exit 1;
fi

if [ -z "${maximum_anchor+x}" ]; then
    echo "[ERROR] maximum_anchor is not set"
    exit 1;
fi

if [ -z "${num_chunks+x}" ]; then
    echo "[ERROR] num_chunks is not set"
    exit 1;
fi

_set_conda
_set_output_size ${output_resolution}
_set_input_size ${input_resolution}
_set_num_blocks ${input_resolution} ${quality}
_set_num_filters ${input_resolution} ${quality}
_set_lr_video_name
_set_hr_video_name

python ${PALANTIR_CODE_ROOT}/palantir/cache_profile/aggregate.py --data_dir ${PALANTIR_DATA_ROOT} --content ${content} --lr_video_name ${lr_video_name} --num_blocks ${num_blocks} --num_filters ${num_filters} --algorithm ${algorithm} --lr_height ${input_resolution} --output_height ${output_height} --patch_width ${patch_width} --patch_height ${patch_height} --maximum_anchor ${maximum_anchor} --num_chunks ${num_chunks}