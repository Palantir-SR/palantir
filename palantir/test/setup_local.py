import os
import shutil
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--libvpx_dir', type=str, required=True)
    parser.add_argument('--snpe_dir', type=str, required=True)
    parser.add_argument('--jni_dir', type=str, required=True)
    parser.add_argument('--binary_dir', type=str, required=True)
    parser.add_argument('--ndk_dir', type=str, required=True)
    args = parser.parse_args()

    # libvpx link
    src = args.libvpx_dir
    dest = os.path.join(args.jni_dir, 'libvpx')
    if not os.path.exists(dest):
        cmd = 'ln -s {} {}'.format(src, dest)
        os.system(cmd)

    # snpe link
    src = args.snpe_dir
    dest = os.path.join(args.jni_dir, 'snpe')
    if not os.path.exists(dest):
        cmd = 'ln -s {} {}'.format(src, dest)
        os.system(cmd)
    
    # switch to libc++_shared.so shipped with snpe
    dest_common_path = os.path.join(args.ndk_dir, "sources/cxx-stl/llvm-libc++/libs")
    # arm64-v8a
    dest = os.path.join(os.path.join(dest_common_path, "arm64-v8a"), "libc++_shared.so")
    os.remove(dest)
    src = os.path.join(args.snpe_dir, "lib/aarch64-android-clang8.0/libc++_shared.so")
    shutil.copy(src, dest)
    '''
    # armeabi-v7a
    dest = os.path.join(os.path.join(dest_common_path, "armeabi-v7a"), "libc++_shared.so")
    os.remove(dest)
    src = os.path.join(args.snpe_dir, "lib/arm-android-clang8.0/libc++_shared.so")
    shutil.copy(src, dest)
    '''

    # configure
    cmd = 'cd {} && make distclean'.format(args.libvpx_dir)
    os.system(cmd)
    cmd = 'cd {} && ./configure {}'.format(args.jni_dir, args.ndk_dir)
    os.system(cmd)

    # build
    build_dir = os.path.join(args.jni_dir, '..')
    cmd = 'cd {} && ndk-build clean && ndk-build NDK_TOOLCHAIN_VERSION=clang APP_STL=c++_shared NDK_DEBUG=0'.format(build_dir)
    os.system(cmd)
