#!/bin/bash

function _set_output_size(){
    if [ "$1" == 2160 ];then
        output_width=3840
        output_height=2160
        output_bitrate=12000
    fi
}

function _set_input_size(){
    if [ "$1" == 480 ];then
        input_width=854
        input_height=480
        input_bitrate=1800
    fi
}

function _set_num_blocks(){
    if [ "$1" == 480 ];then
        if [ "$2" == "low" ];then
            num_blocks=4
        elif [ "$2" == "medium" ];then
            num_blocks=4
        elif [ "$2" == "high" ];then
            num_blocks=8
        fi
    fi
}

function _set_num_filters(){
    if [ "$1" == 480 ];then
        if [ "$2" == "low" ];then
            num_filters=8
        elif [ "$2" == "medium" ];then
            num_filters=24
        elif [ "$2" == "high" ];then
            num_filters=48
        fi
    fi
}

function _set_lr_video_name(){
    lr_video_name="${input_height}p_${input_bitrate}kbps_s0_d300.webm"
}

function _set_hr_video_name(){
    hr_video_name="${output_height}p_${output_bitrate}kbps_s0_d300.webm"
}