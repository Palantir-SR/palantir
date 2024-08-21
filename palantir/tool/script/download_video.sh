#!/bin/bash

function _usage()
{
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-c CONTENTS]

mandatory arguments:
-c CONTENTS                 Specifies indices of contents (e.g., 1)

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

while getopts ":c:h" opt; do
    case $opt in
        h) _usage; exit 0;;
        c) contents+=("$OPTARG");;
        \?) exit 1;
    esac
done

if [ -z "${contents}" ] ; then
    echo "[ERROR] contents is not set"
    exit 1;
fi

_set_conda
for content in "${contents[@]}"
do
    python ${PALANTIR_CODE_ROOT}/palantir/tool/download_video.py --video_dir ${PALANTIR_DATA_ROOT}/video --content ${content}
done
