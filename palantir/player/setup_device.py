import argparse
import sys

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

    #model
    parser.add_argument('--model_type', type=str, default='palantir_s')
    parser.add_argument('--num_filters', type=int)
    parser.add_argument('--num_blocks', type=int)
    parser.add_argument('--upsample_type', type=str, default='deconv')
    parser.add_argument('--train_type', type=str, default='train_video')
    parser.add_argument('--sampling_method', type=str, help="uniform / anchor", required=True)
    parser.add_argument('--train_method', type=str, default=NORMAL_TRAIN, choices=SUPPORTED_TRAIN_METHODS)
    parser.add_argument('--patch_width', type=int, default=None)
    parser.add_argument('--patch_height', type=int, default=None)
    parser.add_argument('--checkpoint_ready', action='store_true')
    parser.add_argument('--num_chunks', type=int, default=75)

    #anchor point selector
    parser.add_argument('--algorithm', type=str, required=True)
    parser.add_argument('--profile_name', type=str, required=True)

    #device
    parser.add_argument('--device_id', type=str, required=True)

    #codec
    parser.add_argument('--output_width', type=int, default=1920)
    parser.add_argument('--output_height', type=int, default=1080)

    args = parser.parse_args()

    #setup directory
    device_root_dir = os.path.join('/sdcard/PALANTIR', args.content)

    #setup videos
    device_video_dir = os.path.join(device_root_dir, 'video')
    adb_mkdir(device_video_dir, args.device_id)
    video_path = os.path.join(args.data_dir, args.content, 'video', args.video_name)
    adb_push(device_video_dir, video_path, args.device_id)

    #convert and setup dnn(s)
    video_profile = get_video_profile(video_path)
    input_shape = [1, args.patch_height, args.patch_width, 3]
    scale = args.output_height // video_profile['height']
    if args.algorithm.startswith("palantir_block") or args.algorithm in ["palantir", "NeuroScaler_patch"]:
        assert args.patch_width > 0 and args.patch_height > 0
        dlc_postfix = f"_block_{args.patch_width}_{args.patch_height}"
    else:
        dlc_postfix = "_frame"

    model = build_model(args.model_type, args.num_blocks, args.num_filters, scale, args.upsample_type, apply_clip=True)
    if args.train_type == 'train_video':
        checkpoint_dir = os.path.join(args.data_dir, args.content, 'checkpoint', args.video_name, model.name)
    elif args.train_type == 'finetune_video':
        checkpoint_dir = os.path.join(args.data_dir, args.content, 'checkpoint', args.video_name, '{}_finetune'.format(model.name))
    else:
        raise ValueError('Unsupported training types')
    checkpoint_dir = os.path.join(checkpoint_dir, args.sampling_method)


    device_checkpoint_dir = os.path.join(device_root_dir, 'checkpoint', args.video_name)
    adb_mkdir(device_checkpoint_dir, args.device_id)
    if args.sampling_method == "anchor":
        for chunk_idx in range(args.num_chunks):
            chunk_path = 'chunk{:04d}'.format(chunk_idx)
            chunk_checkpoint_dir = os.path.join(checkpoint_dir, chunk_path)
            snpe_convert_model(model, input_shape, chunk_checkpoint_dir, dlc_postfix)
            chunk_dlc_path = os.path.join(chunk_checkpoint_dir, '{}{}.dlc'.format(model.name, dlc_postfix))
            chunk_device_checkpoint_dir = os.path.join(device_checkpoint_dir, chunk_path)
            adb_mkdir(chunk_device_checkpoint_dir, args.device_id)
            adb_push(chunk_device_checkpoint_dir, chunk_dlc_path, args.device_id)
    elif args.sampling_method == "uniform":
        print(checkpoint_dir)
        snpe_convert_model(model, input_shape, checkpoint_dir, dlc_postfix)
        dlc_path = os.path.join(checkpoint_dir, '{}{}.dlc'.format(model.name, dlc_postfix))
        adb_push(device_checkpoint_dir, dlc_path, args.device_id)
    else:
        raise NotImplementedError

    #setup a cache profile
    device_cache_profile_dir = os.path.join(device_root_dir, 'profile', args.video_name, model.name)
    adb_mkdir(device_cache_profile_dir, args.device_id)
    cache_profile_path = os.path.join(args.data_dir, args.content, 'profile', args.video_name, model.name, '{}.profile'.format(args.profile_name))
    adb_push(device_cache_profile_dir, cache_profile_path, args.device_id)
