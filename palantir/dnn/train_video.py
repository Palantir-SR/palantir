import argparse
import os
import shutil
import time
import tensorflow as tf
import numpy as np

from palantir.dnn.dataset import train_video_dataset, test_video_dataset, sample_and_save_images
from palantir.dnn.trainer import PalantirTrainer
from palantir.dnn.utility import build_model
from palantir.tool.video import get_video_profile

if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--hr_video_name', type=str, required=True)
    parser.add_argument('--sample_fps', type=float, default=1.0)
    parser.add_argument('--output_height', type=int, default=2160)

    # training & testing
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--num_steps_per_epoch', type=int, default=10)
    parser.add_argument('--load_on_memory', action='store_true')
    parser.add_argument('--num_samples', type=int, default=10)

    # dnn
    parser.add_argument('--model_type', type=str, default='palantir_s')
    parser.add_argument('--num_filters', type=int, required=True)
    parser.add_argument('--num_blocks', type=int, required=True)
    parser.add_argument('--upsample_type', type=str, default='deconv')

    # tool
    parser.add_argument('--ffmpeg_path', type=str, default='/usr/bin/ffmpeg')

    args = parser.parse_args()

    lr_video_path = os.path.join(args.data_dir, args.content, 'video', args.lr_video_name)
    hr_video_path = os.path.join(args.data_dir, args.content, 'video', args.hr_video_name)
    lr_video_profile = get_video_profile(lr_video_path)
    hr_video_profile = get_video_profile(hr_video_path)
    scale = args.output_height// lr_video_profile['height'] #Palantir upscales a LR image to a 1080p version

    # prepare datasets
    lr_image_shape = [lr_video_profile['height'], lr_video_profile['width'], 3]
    hr_image_shape = [lr_video_profile['height'] * scale, lr_video_profile['width'] * scale, 3]
    lr_image_dir = os.path.join(args.data_dir, args.content, 'image', args.lr_video_name, '{}fps'.format(args.sample_fps))
    hr_image_dir = os.path.join(args.data_dir, args.content, 'image', args.hr_video_name, '{}fps'.format(args.sample_fps))
    sample_and_save_images(lr_video_path, lr_image_dir, args.sample_fps, args.ffmpeg_path)
    sample_and_save_images(hr_video_path, hr_image_dir, args.sample_fps, args.ffmpeg_path)
    train_ds = train_video_dataset(lr_image_dir, hr_image_dir, lr_image_shape, hr_image_shape, args.batch_size, args.patch_size, args.load_on_memory)
    test_ds = test_video_dataset(lr_image_dir, hr_image_dir, lr_image_shape, hr_image_shape, args.num_samples, args.load_on_memory)

    model = build_model(args.model_type, args.num_blocks, args.num_filters, scale, args.upsample_type)
    checkpoint_dir = os.path.join(args.data_dir, args.content, 'checkpoint', args.lr_video_name, model.name)
    log_dir = os.path.join(args.data_dir, args.content, 'log', args.lr_video_name,  model.name)

    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "training_time.log"), "w") as f:
        f.write(f"[Config]: {str(args)}\n")
        f.flush()
        start_time = time.time()
        trainer = PalantirTrainer(model, checkpoint_dir, log_dir)
        trainer.train(train_ds, test_ds, args.num_epochs, args.num_steps_per_epoch, f, select_best=True, saturation_detection=False)
        end_time = time.time()
        f.write(f"[Total time]: {end_time-start_time} seconds\n")
        f.flush()
        f.close()

    for epoch_dir in os.listdir(checkpoint_dir):
        if epoch_dir.startswith("epoch"):
            shutil.rmtree(os.path.join(checkpoint_dir, epoch_dir))

    # remove datasets
    shutil.rmtree(lr_image_dir)
    shutil.rmtree(hr_image_dir)