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

## Guide
We provide a step-by-step guide with a single video (whose index is 1).


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

TODO

### 4. Generate a cache profile

TODO

### 5. Compare Palantir vs. baselines

TODO

### 6. Play Palantir in Android smartphones 

TODO