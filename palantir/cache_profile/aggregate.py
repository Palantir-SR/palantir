import os
import sys
import shutil
import argparse
import tensorflow as tf
import numpy as np
from palantir.tool.video import get_video_profile
from palantir.dnn.utility import build_model
from anchor_point_selector import AnchorPointSelector

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--content', type=str, required=True)
    parser.add_argument('--lr_video_name', type=str, required=True)
    parser.add_argument('--num_blocks', type=int)
    parser.add_argument('--num_filters', type=int)
    parser.add_argument('--algorithm', choices=['palantir', 'neuroscaler', 'key_uniform'])
    parser.add_argument('--lr_height', type=int, default=480)
    parser.add_argument('--output_height', type=int, default=2160)
    parser.add_argument('--model_type', type=str, default='palantir_s')
    parser.add_argument('--upsample_type', type=str, default='deconv')
    parser.add_argument('--patch_width', type=int, default=170)
    parser.add_argument('--patch_height', type=int, default=160)
    parser.add_argument('--maximum_anchor', type=int, default=None)
    parser.add_argument('--num_chunks', type=int, default=75)
    parser.add_argument('--interval', type=int, default=1)

    args = parser.parse_args()

    cache_profile_dir = os.path.join(args.data_dir, args.content, "profile", args.lr_video_name, f'{args.model_type.upper()}_B{args.num_blocks}_F{args.num_filters}_S{args.output_height//args.lr_height}_{args.upsample_type}')
    algorithm_name = f"{args.algorithm}_{args.maximum_anchor}_w{args.patch_width}_h{args.patch_height}"
    for num_anchor in range(0, args.maximum_anchor+1, args.interval):
        cache_profile_prefix = f"{algorithm_name}_nanchors_{num_anchor}"
        aggregated_cache_profile_dir = os.path.join(cache_profile_dir, "aggregated")
        os.makedirs(aggregated_cache_profile_dir, exist_ok=True)
        aggregated_cache_profile_name = os.path.join(aggregated_cache_profile_dir, f"nchunks_{args.num_chunks}_{cache_profile_prefix}.profile")
        qualities = []
        with open(aggregated_cache_profile_name, "wb") as f1:
            for chunk_idx in range(args.num_chunks):
                chunk_cache_profile_dir = os.path.join(cache_profile_dir, "chunk{:04d}".format(chunk_idx))
                chunk_cache_profile_names = sorted([os.path.basename(name) for name in os.listdir(chunk_cache_profile_dir) if name.startswith(f"{cache_profile_prefix}.") or name.startswith(f"{cache_profile_prefix}_")], key = lambda i:len(i),reverse=False)
                assert len(chunk_cache_profile_names) == 2
                qualities.append(float(chunk_cache_profile_names[1][len(cache_profile_prefix)+1:-len(".profile")]))
                chunk_cache_profile_name = os.path.join(chunk_cache_profile_dir, chunk_cache_profile_names[0])
                with open(chunk_cache_profile_name, "rb") as f2:
                    f1.write(f2.read())
                    f2.close()
            f1.close()
        
        avg_quality = np.average(np.array(qualities))
        shutil.copy(aggregated_cache_profile_name, f"{aggregated_cache_profile_name[:-len('.profile')]}_{avg_quality}.profile")
        print(f"Aggregated in {aggregated_cache_profile_name[:-len('.profile')]}_{avg_quality}.profile.")