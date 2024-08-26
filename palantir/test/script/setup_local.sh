#!/bin/bash

python ${PALANTIR_CODE_ROOT}/palantir/test/setup_local.py --libvpx_dir ${PALANTIR_CODE_ROOT}/third_party/libvpx --binary_dir ${PALANTIR_CODE_ROOT}/palantir/cache_profile/bin --jni_dir ${PALANTIR_CODE_ROOT}/palantir/test/jni --ndk_dir /android-ndk-r14b --snpe_dir ${PALANTIR_CODE_ROOT}/third_party/snpe
