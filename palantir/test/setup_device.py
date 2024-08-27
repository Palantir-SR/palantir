import argparse
import sys
import math

from palantir.tool.adb import *
from palantir.tool.snpe import *
from palantir.tool.video import get_video_profile
from palantir.dnn.utility import build_model
from palantir.tool.libvpx  import get_num_threads


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #path
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--video_name', type=str, required=True)
    parser.add_argument('--lib_dir', type=str, required=True)
    parser.add_argument('--num_chunks', type=int, default=75)

    #model
    parser.add_argument('--model_type', type=str, default='palantir_s')
    parser.add_argument('--num_filters', type=int)
    parser.add_argument('--num_blocks', type=int)
    parser.add_argument('--upsample_type', type=str, default='deconv')
    parser.add_argument('--patch_width', type=int, default=0)
    parser.add_argument('--patch_height', type=int, default=0)
    parser.add_argument('--checkpoint_ready', action='store_true')

    #anchor point selector
    parser.add_argument('--algorithm', type=str, required=True, choices=['palantir', 'palantir_wo_weight', 'palantir_wo_tc', 'vanilla_palantir', 'partially_optimized_palantir', 'neuroscaler', 'key_uniform'])
    parser.add_argument('--gop', type=int, default=60)
    parser.add_argument('--profile_name', type=str, required=True)

    #device
    parser.add_argument('--device_id', type=str, required=True)

    #codec
    parser.add_argument('--output_width', type=int, default=1920)
    parser.add_argument('--output_height', type=int, default=1080)
    parser.add_argument('--chunk_idx', type=int, default=None)
    parser.add_argument('--limit', type=int, default=None)

    args = parser.parse_args()

    #setup directory
    device_root_dir = os.path.join('/data/local/tmp', args.content)
    device_video_dir = os.path.join(device_root_dir, 'video')
    device_lib_dir = os.path.join('/data/local/tmp', 'libs')
    device_bin_dir = os.path.join('/data/local/tmp', 'bin')
    adb_mkdir(device_video_dir, args.device_id)
    adb_mkdir(device_bin_dir, args.device_id)
    adb_mkdir(device_lib_dir, args.device_id)

    #setup vpxdec
    vpxdec_path = os.path.join(args.lib_dir, 'vpxdec_palantir_ver2')
    print(vpxdec_path)
    adb_push(device_bin_dir, vpxdec_path, args.device_id)
    
    
    #setup library
    c_path = os.path.join(args.lib_dir, 'libc++_shared.so')
    snpe_path = os.path.join(args.lib_dir, 'libSNPE.so')
    libvpx_path = os.path.join(args.lib_dir, 'libvpx.so')
    adb_push(device_lib_dir, c_path, args.device_id)
    adb_push(device_lib_dir, snpe_path, args.device_id)
    adb_push(device_lib_dir, libvpx_path, args.device_id)
    
    #setup videos
    video_path = os.path.join(args.data_dir, args.content, 'video', args.video_name)
    adb_push(device_video_dir, video_path, args.device_id)
    video_profile = get_video_profile(video_path)
    input_height = video_profile['height']
    algorithm = args.algorithm
    if algorithm in ['palantir', 'palantir_wo_weight', 'palantir_wo_tc', 'vanilla_palantir', 'partially_optimized_palantir', 'key_uniform']:
        decode_mode="decode_block_cache"
        assert args.patch_width != 0
        assert args.patch_height != 0
        num_block_per_row = video_profile['width'] // args.patch_width
        num_block_per_column = video_profile['height'] // args.patch_height
    elif algorithm in ['neuroscaler']:
        decode_mode="decode_cache"
        num_block_per_row = 1
        num_block_per_column = 1
    else:
        raise NotImplementedError()
    scale = args.output_height // video_profile['height']

    #setup a dnn
    model = build_model(args.model_type, args.num_blocks, args.num_filters, scale, args.upsample_type, apply_clip=True)
    if not args.checkpoint_ready:
        checkpoint_dir = os.path.join(args.data_dir, args.content, 'checkpoint', args.video_name, model.name)
        device_checkpoint_dir = os.path.join(device_root_dir, 'checkpoint', args.video_name)
        adb_mkdir(device_checkpoint_dir, args.device_id)
        input_shape_list = [([1, video_profile['height'], video_profile['width'], 3], "_frame")]
        if algorithm in ['palantir', 'palantir_wo_weight', 'palantir_wo_tc', 'vanilla_palantir', 'partially_optimized_palantir', 'key_uniform']:
            assert args.patch_width > 0 and args.patch_height > 0
            input_shape_list.append(([1, args.patch_height, args.patch_width, 3], f"_block_{args.patch_width}_{args.patch_height}"))
        for input_shape, postfix in input_shape_list:
            snpe_convert_model(model, input_shape, checkpoint_dir, postfix)
            dlc_path = os.path.join(checkpoint_dir, '{}{}.dlc'.format(model.name, postfix))
            adb_push(device_checkpoint_dir, dlc_path, args.device_id)

    #setup a cache profile
    device_cache_profile_dir = os.path.join(device_root_dir, 'profile', args.video_name, model.name)
    adb_mkdir(device_cache_profile_dir, args.device_id)
    cache_profile_path = os.path.join(args.data_dir, args.content, 'profile', args.video_name, model.name, 'aggregated' if args.chunk_idx is None else "chunk{:04d}".format(args.chunk_idx), '{}.profile'.format(args.profile_name))
    adb_push(device_cache_profile_dir, cache_profile_path, args.device_id)

    # limit and skip
    limit = ""
    skip = ""
    if args.chunk_idx is not None:
        num_total_frames = int(round(video_profile['frame_rate'], 3) * round(video_profile['duration']))
        num_left_frames = num_total_frames - args.chunk_idx * args.gop
        num_skipped_frames = args.chunk_idx * args.gop
        num_decoded_frames = args.gop if num_left_frames >= args.gop else num_left_frames
        limit = f" --limit={num_decoded_frames}"
        skip = f" --skip={num_skipped_frames}"
    if args.limit is not None:
        limit = f" --limit={args.limit}"

    #setup scripts (setup.sh, offline_dnn.sh, online_dnn.sh)
    script_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.script')
    os.makedirs(script_dir, exist_ok=True)
    
    #case 1: No SR
    device_script_dir = os.path.join(device_root_dir, 'script', args.video_name)
    adb_mkdir(device_script_dir, args.device_id)
    cmds = ['#!/system/bin/sh',
            'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{}'.format(device_lib_dir),
            'cd {}'.format(device_root_dir),
            '{} --codec=vp9  --noblit --threads={} --frame-buffers=50 {} {} --dataset-dir={} --input-video-name={} --save-latency --save-metadata'.format(os.path.join(device_bin_dir, 'vpxdec_palantir_ver2'), get_num_threads(input_height), limit, skip, device_root_dir, args.video_name),
            'exit']
    cmd_script_path = os.path.join(script_dir, 'measure_decode_latency.sh')
    with open(cmd_script_path, 'w') as cmd_script:
        for ln in cmds:
            cmd_script.write(ln + '\n')
    adb_push(device_script_dir, cmd_script_path, args.device_id)
    os.system('adb -s {} shell "chmod +x {}"'.format(args.device_id, os.path.join(device_script_dir, '*.sh')))

    #case 2: Per-frame SR
    device_script_dir = os.path.join(device_root_dir, 'script', args.video_name, model.name)
    adb_mkdir(device_script_dir, args.device_id)
    cmds = ['#!/system/bin/sh',
            'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{}'.format(device_lib_dir),
            'cd {}'.format(device_root_dir),
            '{} --codec=vp9  --noblit --threads={} --frame-buffers=50 {} {} --dataset-dir={} --input-video-name={}  --decode-mode=decode_sr --dnn-mode=online_dnn --dnn-runtime=gpu_float16 --dnn-name={} --dnn-scale={} --save-latency --save-metadata'.format(os.path.join(device_bin_dir, 'vpxdec_palantir_ver2'), get_num_threads(input_height), limit, skip, device_root_dir, args.video_name, model.name, scale),
            'exit']
    cmd_script_path = os.path.join(script_dir, 'measure_per_frame_sr_latency.sh')
    with open(cmd_script_path, 'w') as cmd_script:
        for ln in cmds:
            cmd_script.write(ln + '\n')
    adb_push(device_script_dir, cmd_script_path, args.device_id)
    os.system('adb -s {} shell "chmod +x {}"'.format(args.device_id, os.path.join(device_script_dir, '*.sh')))
    
    #case 3: cache
    device_script_dir = os.path.join(device_root_dir, 'script', args.video_name, model.name, algorithm)
    adb_mkdir(device_script_dir, args.device_id)

    cmds = ['#!/system/bin/sh',
        'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{}'.format(device_lib_dir),
        'cd {}'.format(device_root_dir),
        '{} --codec=vp9  --noblit --threads={} --frame-buffers=50 --dataset-dir={} --input-video-name={} --decode-mode={} --dnn-mode=online_dnn --dnn-runtime=gpu_float16 --cache-mode=profile_cache --dnn-name={} --dnn-scale={} --cache-profile-name={} --save-latency --save-metadata --patch-width={} --patch-height={} --npatches-per-row={} --npatches-per-column={} --gop={} {} {}'.format(os.path.join(device_bin_dir, 'vpxdec_palantir_ver2'), get_num_threads(input_height), device_root_dir, args.video_name, decode_mode, model.name, scale, args.profile_name, args.patch_width, args.patch_height, num_block_per_row, num_block_per_column, args.gop, limit, skip),
            'exit']
    cmd_script_path = os.path.join(script_dir, f'measure_{algorithm}_latency.sh')
    with open(cmd_script_path, 'w') as cmd_script:
        for ln in cmds:
            cmd_script.write(ln + '\n')
    adb_push(device_script_dir, cmd_script_path, args.device_id)
    os.system('adb -s {} shell "chmod +x {}"'.format(args.device_id, os.path.join(device_script_dir, '*.sh')))
