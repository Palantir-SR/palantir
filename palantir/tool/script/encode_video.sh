#!/bin/bash

source ${PALANTIR_CODE_ROOT}/palantir/script/common_funcs.sh

function _usage()
{
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-c CONTENTS] [-g gop]

mandatory arguments:
-c CONTENTS                 Specifies indices of contents (e.g., 1)
-g gop                      Specifies the gop (i.e., the group of pictures)

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

function _transcode()
{
    #cut and encode (2160p)
    _set_output_size 2160
    _set_hr_video_name
    #python $PALANTIR_CODE_ROOT/palantir/tool/encode_video.py --output_video_dir $PALANTIR_DATA_ROOT/$1/video --input_video_path $PALANTIR_DATA_ROOT/video/$1.webm --start 0 --duration 300 --bitrate $output_bitrate --mode cut_and_resize_and_encode --output_width $output_width --output_height $output_height --gop $2

    #encode (480p)
    _set_input_size 480
    python $PALANTIR_CODE_ROOT/palantir/tool/encode_video.py --output_video_dir $PALANTIR_DATA_ROOT/$1/video --input_video_path $PALANTIR_DATA_ROOT/$1/video/$hr_video_name --bitrate $input_bitrate --output_width $input_width --output_height $input_height --start 0 --duration 300 --mode resize_and_encode --gop $2
}

[[ ($# -ge 1)  ]] || { echo "[ERROR] Invalid number of arguments. See -h for help."; exit 1;  }

while getopts "c:g:h" opt; do
    case $opt in
        h) _usage; exit 0;;
        c) contents+=("$OPTARG");;
        g) gop=("$OPTARG");;
        \?) exit 1;
    esac
done

if [ -z "${contents}" ] ; then
    echo "[ERROR] contents is not set"
    exit 1;
fi

if [ -z "${gop}" ] ; then
    echo "[ERROR] gop is not set"
    exit 1;
fi

_set_conda
for content in "${contents[@]}"; do
    _transcode $content $gop
done
