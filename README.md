# Palantir

This is an official Github repository for the paper "Palantir: Towards Efficient Super Resolution for Ultra-high-definition Live Streaming". This project is built upon Google libvpx, Android Exoplayer, and Qualcomm SNPE and consists of C/C++/Java/Python.   
[[Project homepage]](https://palantir-sr.github.io/) [[Paper]](https://www.arxiv.org/abs/2408.06152)

If you use our work for research, please cite it.
```
@misc{jin2024palantir,
      title={Palantir: Towards Efficient Super Resolution for Ultra-high-definition Live Streaming}, 
      author={Xinqi Jin and Zhui Zhu and Xikai Sun and Fan Dang and Jiangchuan Liu and Jingao Xu and Kebin Liu and Xinlei Chen and Yunhao Liu},
      year={2024},
      eprint={2408.06152},
      archivePrefix={arXiv},
      primaryClass={cs.MM},
      url={https://arxiv.org/abs/2408.06152}, 
}
```
Lastly, Palantir is currently protected under the patent and is retricted to be used for the commercial usage.  
* `BY-NC-SA` – [Attribution-NonCommercial-ShareAlike](https://github.com/idleberg/Creative-Commons-Markdown/blob/master/4.0/by-nc-sa.markdown)

## Project structure
```
./palantir
├── test                  # Python: Test the performance of Palantir
├── dnn                    # Python: DNN trainer/converter
├── cache_profile          # Python: Anchor point selector
├── player                 # Java, C/C++: Android video player built upon Exoplayer and the SR-integrated codec
|—— tool                   # Python: Tools used by Palantir
./third_party
├── libvpx                 # C/C++: SR-integrated codec
```

## Prerequisites

* OS: Ubuntu 16.04 or higher versions
* HW: NVIDIA GPU
* Docker: https://docs.docker.com/install/
* NVIDIA docker: https://github.com/NVIDIA/nvidia-docker
* Qualcomm SNPE SDK (v1.68.0): https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk/tools
  (We cannot provide this due to the Qualcom license policy.)
* Android Studio Hedgehog | 2023.1.1 (Build #AI-231.9392.1.2311.11076708, built on November 10, 2023) + Gradle 7.5 + Android Gradle Plugin Version 7.4.1 + Java 8 + API 29 ("Q"; Android 10.0) + Build Tools 29.0.2.

## Guide

We provide a step-by-step guide with a single video (whose index is 1).

All the commands for the Step 2-7 must be executed inside the docker.

### 1. Setup
* Clone the Palantir docker repository
```
git clone https://github.com/Palantir-SR/palantir-docker.git
```
* Build the docker image 
```
cd ${HOME}/palantir-docker
./build.sh
```
* Run & Attach to the docker
```
cd ${HOME}/palantir-docker
./run.sh
```
* Clone the Palantir main repository
```
git clone --recurse-submodules https://github.com/Palantir-SR/palantir ${PALANTIR_CODE_ROOT}
```
* Download/Setup the Qualcomm SNPE SDK as follow:
```
./palantir
├── third_party
    ├── snpe
        ├── benchmarks
        ├── bin
        ├── include
        ├── lib
        ├── models
        ├── share
        ...
```

### 2. Prepare videos

* Download a Youtube video
```
$PALANTIR_CODE_ROOT/palantir/tool/script/download_video.sh -c 1
```

* Transcode the video 
```
$PALANTIR_CODE_ROOT/palantir/tool/script/encode_video.sh -c 1 -g 60
```

[Details are described in this file.](palantir/tool/README.md)

### 3. Prepare DNNs 

* Train a DNN
```
$PALANTIR_CODE_ROOT/palantir/dnn/script/train_video.sh -g 0 -c 1 -q high -i 480 -o 2160
```

[Details are described in this file.](palantir/dnn/README.md)

### 4. Compute PSNR under different settings 

* Setup: Build the SR-integrated codec (x86_64)
```
$PALANTIR_CODE_ROOT/palantir/cache_profile/script/setup.sh
```

#### 4.1 NeuroScaler (Baseline #1)

* Generate the cache profiles for every chunk
```
$PALANTIR_CODE_ROOT/palantir/cache_profile/script/select_anchor_points.sh -g 0 -c 1 -q high -i 480 -o 2160 -a neuroscaler -W 854 -H 480 -m 5 -p generate_profile
```

* Measure PSNR for all the generated cache profiles
```
$PALANTIR_CODE_ROOT/palantir/cache_profile/script/select_anchor_points.sh -g 0 -c 1 -q high -i 480 -o 2160 -a neuroscaler -W 854 -H 480 -m 5 -p measure_profile
```

* Aggregate the cache profiles across all chunks to obtain the average PSNR
```
$PALANTIR_CODE_ROOT/palantir/cache_profile/script/aggregate.sh -a neuroscaler -c 1 -q high -i 480 -o 2160 -W 854 -H 480 -m 5 -n 150
```

#### 4.2 Key+Uniform (Baseline #2)

* Generate the cache profiles for every chunk
```
$PALANTIR_CODE_ROOT/palantir/cache_profile/script/select_anchor_points.sh -g 0 -c 1 -q high -i 480 -o 2160 -a key_uniform -W 170 -H 160 -m 75 -p generate_profile
```

* Measure PSNR for all the generated cache profiles
```
$PALANTIR_CODE_ROOT/palantir/cache_profile/script/select_anchor_points.sh -g 0 -c 1 -q high -i 480 -o 2160 -a key_uniform -W 170 -H 160 -m 75 -p measure_profile
```

* Aggregate the cache profiles across all chunks to obtain the average PSNR
```
$PALANTIR_CODE_ROOT/palantir/cache_profile/script/aggregate.sh -a key_uniform -c 1 -q high -i 480 -o 2160 -W 170 -H 160 -m 75 -n 150
```

#### 4.3 Palantir

* Generate the cache profiles for every chunk
```
$PALANTIR_CODE_ROOT/palantir/cache_profile/script/select_anchor_points.sh -g 0 -c 1 -q high -i 480 -o 2160 -a palantir -W 170 -H 160 -m 75 -p generate_profile
```

* Measure PSNR for all the generated cache profiles
```
$PALANTIR_CODE_ROOT/palantir/cache_profile/script/select_anchor_points.sh -g 0 -c 1 -q high -i 480 -o 2160 -a palantir -W 170 -H 160 -m 75 -p measure_profile
```

* Aggregate the cache profiles across all chunks to obtain the average PSNR
```
$PALANTIR_CODE_ROOT/palantir/cache_profile/script/aggregate.sh -a palantir -c 1 -q high -i 480 -o 2160 -W 170 -H 160 -m 75 -n 150
```

#### 4.4 (Optional) Applying bilinear interpolation on all the frames

* Save the average PSNR value
```
TODO
```

#### 4.5 (Optional) Applying DNN-based SR on all the frames

* Save the average PSNR value
```
TODO
```

### 5. Execute in Android adb shell 

* Setup: Build the SR-integrated codec (arm64-v8)
```
# Execute the following command in the docker:
$PALANTIR_CODE_ROOT/palantir/test/script/setup_local.sh 
```

* Copy data (including the DNN model, the profile, the low-resolution video, and the SR-integrated libvpx decoder) to mobile devices (use `adb devices` in the container to obtain the  `device_id`)
```
$PALANTIR_CODE_ROOT/palantir/test/script/setup_device.sh -c 1 -q high -r 480 -o 2160 -a [algorithm_name] -p [profile_name] -d [device_id] -W 170 -H 160 -g 60
(e.g., $PALANTIR_CODE_ROOT/palantir/test/script/setup_device.sh -c 1 -q high -r 480 -o 2160 -a palantir -p nchunks_150_palantir_75_w170_h160_nanchors_15 -d [device_id] -W 170 -H 160 -g 60)
(e.g., $PALANTIR_CODE_ROOT/palantir/test/script/setup_device.sh -c 1 -q high -r 480 -o 2160 -a neuroscaler -p nchunks_150_neuroscaler_5_w854_h480_nanchors_1 -d [device_id] -W 854 -H 480 -g 60)
```

* Execute the decoder through the adb shell 

### 6 (Optional) Preliminary experiment (see Sec. 4.1 of the paper)

* TODO
```
TODO
```

### 7 (Optional) Ablation study (see Sec. 6.4 of the paper)

#### 7.1 Palantir w/o weight

* Generate the cache profiles for every chunk
```
$PALANTIR_CODE_ROOT/palantir/cache_profile/script/select_anchor_points.sh -g 0 -c 1 -q high -i 480 -o 2160 -a palantir_wo_weight -W 170 -H 160 -m 75 -p generate_profile
```

* Measure PSNR for all the generated cache profiles
```
$PALANTIR_CODE_ROOT/palantir/cache_profile/script/select_anchor_points.sh -g 0 -c 1 -q high -i 480 -o 2160 -a palantir_wo_weight -W 170 -H 160 -m 75 -p measure_profile
```

* Aggregate the cache profiles across all chunks to obtain the average PSNR
```
$PALANTIR_CODE_ROOT/palantir/cache_profile/script/aggregate.sh -a palantir_wo_weight -c 1 -q high -i 480 -o 2160 -W 170 -H 160 -m 75 -n 150
```

#### 7.2 Palantir w/o TC

* Generate the cache profiles for every chunk
```
$PALANTIR_CODE_ROOT/palantir/cache_profile/script/select_anchor_points.sh -g 0 -c 1 -q high -i 480 -o 2160 -a palantir_wo_tc -W 170 -H 160 -m 75 -p generate_profile
```

* Measure PSNR for all the generated cache profiles
```
$PALANTIR_CODE_ROOT/palantir/cache_profile/script/select_anchor_points.sh -g 0 -c 1 -q high -i 480 -o 2160 -a palantir -W 170 -H 160 -m 75 -p measure_profile
```

* Aggregate the cache profiles across all chunks to obtain the average PSNR
```
$PALANTIR_CODE_ROOT/palantir/cache_profile/script/aggregate.sh -a palantir -c 1 -q high -i 480 -o 2160 -W 170 -H 160 -m 75 -n 150
```

#### 7.3 Vanilla Palantir

* Generate the cache profile (the generated profile should be the same as [Palantir](#43-palantir) but the generation thoughput is much lower)
```
$PALANTIR_CODE_ROOT/palantir/cache_profile/script/select_anchor_points.sh -g 0 -c 1 -q high -i 480 -o 2160 -a vanilla_palantir -W 170 -H 160 -m 75 -p generate_profile
```

#### 7.4 Partially optimized Palantir

* Generate the cache profile (the generated profile should be the same as [Palantir](#43-palantir) but the generation thoughput is much lower)
```
$PALANTIR_CODE_ROOT/palantir/cache_profile/script/select_anchor_points.sh -g 0 -c 1 -q high -i 480 -o 2160 -a partially_optimized_palantir -W 170 -H 160 -m 75 -p generate_profile
```

### 8. (Optional) Build and install the SR-integrated Exoplayer Android app

* Setup: Copy data to mobile devices
```
$PALANTIR_CODE_ROOT/palantir/player/script/setup_device.sh -c 1 -q high -r 480 -o 2160 -a [algorithm_name] -p [cache_profile_name] -d ${deviceId} -W 170 -H 160 -g 60
(e.g., $PALANTIR_CODE_ROOT/palantir/player/script/setup_device.sh -c 1 -q high -r 480 -o 2160 -a palantir -p nchunks_150_palantir_75_w170_h160_nanchors_15 -d [device_id] -W 170 -H 160 -g 60)
(e.g., $PALANTIR_CODE_ROOT/palantir/player/script/setup_device.sh -c 1 -q high -r 480 -o 2160 -a neuroscaler -p nchunks_150_neuroscaler_5_w854_h480_nanchors_1 -d [device_id] -W 854 -H 480 -g 60)
```

* Build and install the player app
1. Enable USB debugging and USB install at mobiles.
2. Outside the docker container: open `$HOME/docker/palantir/palantir/palantir/player` using Android Studio
3. Inside the docker container: 
  
```
cp -r /workspace/palantir/third_party/libvpx /workspace/palantir/palantir/player/ExoPlayer/extensions/vp9/src/main/jni && cp /workspace/palantir/third_party/snpe/lib/aarch64-android-clang8.0/libc++_shared.so /android-ndk-r14b/sources/cxx-stl/llvm-libc++/libs/arm64-v8a
cd /workspace/palantir/palantir/player/ExoPlayer/extensions/vp9/src/main/jni/libvpx
make distclean
cd ..
```
4. Inside the docker container: build vp9 extension according to the [documentation](./palantir/player/ExoPlayer/extensions/vp9/README.md)
5. Inside the docker container:
```
cp -r /android-ndk-r14b /workspace/android-ndk-r14b
```
6. Outside the docker container:
```
sudo apt-get install libncurses5 make
```
7. Outside the docker container: set ndk path in local.properties as `ndk.dir=/home/[user_name]/docker/palantir/android-ndk-r14b`, using Android Studio.
8. Outside the docker container: build & install it with Android studio.