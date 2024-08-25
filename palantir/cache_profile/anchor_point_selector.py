import os
import sys
import argparse
import shlex
import math
import multiprocessing as mp
import shutil
import random
import itertools
import copy
import glob
from math import log10
from joypy import joyplot
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from palantir.tool.video import get_video_profile
import palantir.tool.libvpx as libvpx

class AnchorPointSelector():
    def __init__(self, model, vpxdec_path, dataset_dir, lr_video_name, hr_video_name, gop, output_width, output_height, num_decoders, scale, patch_width, patch_height, maximum_anchor):
        self.model = model
        self.vpxdec_path = vpxdec_path
        self.dataset_dir = dataset_dir
        self.lr_video_name = lr_video_name
        self.hr_video_name = hr_video_name
        self.gop = gop
        self.output_width = output_width
        self.output_height = output_height
        self.num_decoders = num_decoders
        self.scale = scale
        self.patch_width = patch_width
        self.patch_height = patch_height

        lr_video_path = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
        lr_video_profile = get_video_profile(lr_video_path)
        self.lr_width = int(lr_video_profile['width'])
        self.lr_height = int(lr_video_profile['height'])

        self.num_blocks_per_row = lr_video_profile['width'] // self.patch_width
        self.num_blocks_per_column = lr_video_profile['height'] // self.patch_height

        self.cache_profile_dir = None
        self.num_skipped_frames=0
        self.num_decoded_frames=0
        self.postfix=None
        self.maximum_anchor = maximum_anchor
        
    def _select_anchor_point_set_palantir(self, chunk_idx, processing_mode="generate_profile", disable_weights=False, disable_texture_complexities=False):
        assert processing_mode in ["generate_profile", "measure_profile"]

        postfix = 'chunk{:04d}'.format(chunk_idx)
        cache_profile_dir = os.path.join(self.dataset_dir, 'profile', self.lr_video_name, self.model.name, postfix)
        log_dir = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, postfix)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(cache_profile_dir, exist_ok=True)
        algorithm_type = 'palantir{}{}_{}_w{}_h{}'.format("_disable_weights" if disable_weights else "", "_disable_texture_complexities" if disable_texture_complexities else "", self.maximum_anchor, self.patch_width, self.patch_height)
        self.algorithm_type = algorithm_type

        ###########step 1: save hr images ##########
        lr_video_path = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
        lr_video_profile = get_video_profile(lr_video_path)
        num_total_frames = int(round(lr_video_profile['frame_rate'], 3) * round(lr_video_profile['duration']))
        num_left_frames = num_total_frames - chunk_idx * self.gop
        num_skipped_frames = chunk_idx * self.gop
        num_decoded_frames = self.gop if num_left_frames >= self.gop else num_left_frames

        ###########step 2: construct the dependency graph ##########

        if processing_mode == "generate_profile":
            graph = libvpx.fastGraph(self.vpxdec_path, self.num_blocks_per_row, self.num_blocks_per_column, self.patch_width, self.patch_height, self.dataset_dir, postfix, self.lr_video_name, self.hr_video_name, self.lr_height, self.lr_width, self.output_height, self.output_width, os.path.join(self.dataset_dir, "palantir_anchor", self.lr_video_name, self.hr_video_name, algorithm_type), self.gop, num_skipped_frames, num_decoded_frames, disable_weights=disable_weights, disable_texture_complexities=disable_texture_complexities, intra_parallelism=True, inter_parallelism=True, schedulingInterval=self.gop)
            graph.construct_graph_through_metadata(self.model)
            anchor_blocks = []
            non_root_anchor_blocks = []

        ###########step 3: search anchors ##########
        if processing_mode == "generate_profile":

            anchor_point_set = graph.get_anchor_point_set(cache_profile_dir, f"{algorithm_type}_nanchors_{graph.get_num_anchor_points()}")
            anchor_point_set.save_cache_profile()

            for _ in range(self.maximum_anchor):
                non_root_anchor_blocks.append(graph.search_new_anchor_blocks())
                anchor_point_set = graph.get_anchor_point_set(cache_profile_dir, f"{algorithm_type}_nanchors_{graph.get_num_anchor_points()}")
                anchor_point_set.save_cache_profile()
            print(f"Profile generated for {postfix}.")
        
        ###########step 4: measure qualities ##########
        elif processing_mode == "measure_profile":
            libvpx.save_rgb_frame(self.vpxdec_path, self.dataset_dir, self.lr_video_name, skip=num_skipped_frames, limit=num_decoded_frames, postfix=postfix)
            frames = libvpx.load_frame_index(self.dataset_dir, self.lr_video_name, postfix)
            #load the cache profiles
            anchor_point_sets = [libvpx.AnchorPointSet.create(frames, cache_profile_dir, f"{algorithm_type}_nanchors_{idx}", self.num_blocks_per_row, self.num_blocks_per_column) for idx in range(self.maximum_anchor+1)]
            for idx in range(self.maximum_anchor+1):
                anchor_point_sets[idx].load_from_file(os.path.join(cache_profile_dir, f"{algorithm_type}_nanchors_{idx}.profile"))
            libvpx.save_yuv_frame(self.vpxdec_path, self.dataset_dir, self.hr_video_name, self.output_width, self.output_height, num_skipped_frames, num_decoded_frames, postfix)
            #libvpx.setup_sr_frame(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.model, postfix)

            self.measure_quality(anchor_point_sets, True, num_skipped_frames, num_decoded_frames, postfix)

            ###########step 5: remove images ##########
            lr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, postfix)
            hr_image_dir = os.path.join(self.dataset_dir, 'image', self.hr_video_name, postfix)
            sr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, self.model.name, postfix)
            shutil.rmtree(lr_image_dir, ignore_errors=True)
            shutil.rmtree(hr_image_dir, ignore_errors=True)
            #shutil.rmtree(sr_image_dir, ignore_errors=True)

    def _preliminary_exp(self, chunk_idx):
        postfix = 'chunk{:04d}'.format(chunk_idx)
        num_total_frames = int(round(lr_video_profile['frame_rate'], 3) * round(lr_video_profile['duration']))
        num_left_frames = num_total_frames - chunk_idx * self.gop
        num_skipped_frames = chunk_idx * self.gop
        num_decoded_frames = self.gop if num_left_frames >= self.gop else num_left_frames
        sorted_intra, sorted_inter, sorted_key_and_intra, sorted_normal = graph.offline_fast_graph_finegrained_metadata(self.patch_width, self.patch_height, self.num_blocks_per_row, self.num_blocks_per_column, self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.gop, skip=num_skipped_frames, limit=num_decoded_frames, postfix=postfix)
        return sorted_intra, sorted_inter, sorted_key_and_intra, sorted_normal



    def _select_anchor_point_set_neuroscaler(self, chunk_idx, processing_mode):
        assert self.maximum_anchor >= 1
        postfix = 'chunk{:04d}'.format(chunk_idx)
        cache_profile_dir = os.path.join(self.dataset_dir, 'profile', self.lr_video_name, self.model.name, postfix)
        log_dir = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, postfix)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(cache_profile_dir, exist_ok=True)
        algorithm_type = 'neuroscaler_{}_w{}_h{}'.format(self.maximum_anchor, self.patch_width, self.patch_height)
        self.algorithm_type = algorithm_type

        ###########step 1: generate metadata and images (decorded, super-resoluted and high-resolution)##########
        lr_video_path = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
        lr_video_profile = get_video_profile(lr_video_path)
        num_total_frames = int(round(lr_video_profile['frame_rate'], 3) * round(lr_video_profile['duration']))
        num_left_frames = num_total_frames - chunk_idx * self.gop
        num_skipped_frames = chunk_idx * self.gop
        num_decoded_frames = self.gop if num_left_frames >= self.gop else num_left_frames

        #save low-resolution images
        libvpx.save_residual(self.vpxdec_path, self.dataset_dir, self.lr_video_name, skip=num_skipped_frames, limit=num_decoded_frames, postfix=postfix)

        ###########step 2: calculate the residual##########

        frames = libvpx.load_frame_index(self.dataset_dir, self.lr_video_name, postfix, load_frame_size=True)
        lr_image_dir = os.path.join(self.dataset_dir, "image", self.lr_video_name, postfix)

        if processing_mode=="generate_profile":

            arf_indices = [] # alternative reference frames
            nf_indices = [] # normal frames
            kf_indices = [] # key frames
            is_anchor = []
            residuals = []
            accumulated_residuals = []

            for idx, frame in enumerate(frames):
                if frame.frame_type == "key_frame":
                    residuals.append(0)
                    accumulated_residuals.append(0)
                    kf_indices.append(idx)
                    is_anchor.append(True)
                else:
                    residuals.append(frame.frame_size)
                    accumulated_residuals.append(accumulated_residuals[-1]+residuals[-1])
                    is_anchor.append(False)
                    if frame.frame_type == "alternative_reference_frame":
                        arf_indices.append(idx)
                    else:
                        nf_indices.append(idx)

        ###########step 3: sort the candidate frames##########

        if processing_mode=="generate_profile":

            def sort_candidates(candidates, residuals, accumulated_residuals, is_done):
                assert len(residuals) == len(accumulated_residuals)
                assert len(accumulated_residuals) == len(is_done)
                N = len(candidates)
                N2 = len(accumulated_residuals)
                sorted_candidates = []
                estimated_benifits = []
                for i in range(N):
                    max_gain = -1
                    for j in candidates:
                        if not is_done[j]:
                            for k in range(j+1, N2+1):
                                if k == N2:
                                    break
                                if is_done[k]:
                                    break
                            gain = (k-j)*accumulated_residuals[j]
                            if gain > max_gain:
                                max_gain = gain
                                best_candidate = j
                                reduced_residual = accumulated_residuals[best_candidate]
                    candidates.remove(best_candidate)
                    sorted_candidates.append(best_candidate)
                    estimated_benifits.append(max_gain)
                    for k in range(best_candidate, N2):
                        if is_done[k]:
                            break
                        accumulated_residuals[k] -= reduced_residual
                    is_done[best_candidate] = True
                return sorted_candidates, estimated_benifits

            sorted_candidate_arfs, arf_weights = sort_candidates(copy.deepcopy(arf_indices), residuals, copy.deepcopy(accumulated_residuals), copy.deepcopy(is_anchor))
            sorted_candidate_nfs, nf_weights = sort_candidates(copy.deepcopy(nf_indices), residuals, copy.deepcopy(accumulated_residuals), copy.deepcopy(is_anchor))

        ###########step 4: save the profile##########

        if processing_mode=="generate_profile":
            anchor_point_set = libvpx.AnchorPointSet.create(frames, cache_profile_dir, '{}_nanchors_0'.format(algorithm_type), self.num_blocks_per_row, self.num_blocks_per_column)

            anchor_point_set.save_cache_profile()

            anchor_frame_indices = []
            for i in range(self.maximum_anchor):
                anchor_point_set = libvpx.AnchorPointSet.load(anchor_point_set, cache_profile_dir, '{}_nanchors_{}'.format(algorithm_type, anchor_point_set.get_num_anchor_points() + 1), self.num_blocks_per_row, self.num_blocks_per_column)
                if i < len(kf_indices):
                    idx = kf_indices[i]
                    anchor_point_set.add_anchor_point(frames[idx], 0, 0)
                    #print(frames[idx].name)
                    anchor_frame_indices.append(idx)
                elif i < len(kf_indices) + len(sorted_candidate_arfs):
                    idx = sorted_candidate_arfs[i-len(kf_indices)]
                    anchor_point_set.add_anchor_point(frames[idx], 0, 0)
                    #print(frames[idx].name)
                    anchor_frame_indices.append(idx)
                else:
                    idx = sorted_candidate_nfs[i-len(kf_indices)-len(sorted_candidate_arfs)]
                    anchor_point_set.add_anchor_point(frames[idx], 0, 0)
                    #print(frames[idx].name)
                    anchor_frame_indices.append(idx)
                anchor_point_set.save_cache_profile()
            print(f"Profile generated for {postfix}.")

        elif processing_mode=="measure_profile":
            
            anchor_point_sets = [libvpx.AnchorPointSet.create(frames, cache_profile_dir, f"{algorithm_type}_nanchors_{idx}", self.num_blocks_per_row, self.num_blocks_per_column) for idx in range(self.maximum_anchor+1)]
            for idx in range(self.maximum_anchor+1):
                anchor_point_sets[idx].load_from_file(os.path.join(cache_profile_dir, f"{algorithm_type}_nanchors_{idx}.profile"))
            #save low-resolution, super-resoluted, high-resolution frames to local storage
            libvpx.save_rgb_frame(self.vpxdec_path, self.dataset_dir, self.lr_video_name, skip=num_skipped_frames, limit=num_decoded_frames, postfix=postfix)
            libvpx.save_yuv_frame(self.vpxdec_path, self.dataset_dir, self.hr_video_name, self.output_width, self.output_height, num_skipped_frames, num_decoded_frames, postfix)
            #libvpx.setup_sr_frame(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.model, postfix)

            self.measure_quality(anchor_point_sets, False, num_skipped_frames, num_decoded_frames, postfix)

            #remove images
            lr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, postfix)
            hr_image_dir = os.path.join(self.dataset_dir, 'image', self.hr_video_name, postfix)
            sr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, self.model.name, postfix)
            shutil.rmtree(lr_image_dir, ignore_errors=True)
            shutil.rmtree(hr_image_dir, ignore_errors=True)
            #shutil.rmtree(sr_image_dir, ignore_errors=True)

    def _select_anchor_point_set_key_uniform(self, chunk_idx, processing_mode):
        assert self.maximum_anchor >= 1
        postfix = 'chunk{:04d}'.format(chunk_idx)
        cache_profile_dir = os.path.join(self.dataset_dir, 'profile', self.lr_video_name, self.model.name, postfix)
        log_dir = os.path.join(self.dataset_dir, 'log', self.lr_video_name, self.model.name, postfix)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(cache_profile_dir, exist_ok=True)
        algorithm_type = 'key_uniform_{}_w{}_h{}'.format(self.maximum_anchor, self.patch_width, self.patch_height)
        self.algorithm_type = algorithm_type

        ###########step 1: generate metadata and images (decoded, super-resoluted and high-resolution)##########
        lr_video_path = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
        lr_video_profile = get_video_profile(lr_video_path)
        num_total_frames = int(round(lr_video_profile['frame_rate'], 3) * round(lr_video_profile['duration']))
        num_left_frames = num_total_frames - chunk_idx * self.gop
        num_skipped_frames = chunk_idx * self.gop
        num_decoded_frames = self.gop if num_left_frames >= self.gop else num_left_frames

        #save low-resolution images
        libvpx.save_metadata(self.vpxdec_path, self.dataset_dir, self.lr_video_name, skip=num_skipped_frames, limit=num_decoded_frames, postfix=postfix, save_frame_size=True)

        ###########step 2: calculate the residual##########
        frames = libvpx.load_frame_index(self.dataset_dir, self.lr_video_name, postfix, load_frame_size=False)
        lr_image_dir = os.path.join(self.dataset_dir, "image", self.lr_video_name, postfix)

        kf_indices = [] # key frames
        remaining_indices = [] # remaining frames

        for idx, frame in enumerate(frames):
            if frame.frame_type == "key_frame":
                kf_indices.append(idx)
            else:
                remaining_indices.append(idx)

        ###########step 3: save the profile##########

        if processing_mode=="generate_profile":
            anchor_point_set = libvpx.AnchorPointSet.create(frames, cache_profile_dir, '{}_nanchors_0'.format(algorithm_type), self.num_blocks_per_row, self.num_blocks_per_column)

            anchor_point_set.save_cache_profile()

            anchor_frame_indices = []
            if self.maximum_anchor >= len(kf_indices)*self.num_blocks_per_row*self.num_blocks_per_column:
                anchor_point_set = libvpx.AnchorPointSet.load(anchor_point_set, cache_profile_dir, '{}_nanchors_{}'.format(algorithm_type, len(kf_indices)*self.num_blocks_per_row*self.num_blocks_per_column), self.num_blocks_per_row, self.num_blocks_per_column)
                for idx in kf_indices:
                    for row_idx in range(self.num_blocks_per_column):
                        for col_idx in range(self.num_blocks_per_row):
                            anchor_point_set.add_anchor_point(frames[idx], row_idx, col_idx)
                            libvpx.AnchorPointSet.load(anchor_point_set, cache_profile_dir, '{}_nanchors_{}'.format(algorithm_type, anchor_point_set.get_num_anchor_points()), self.num_blocks_per_row, self.num_blocks_per_column).save_cache_profile()
                base_anchor_point_set = anchor_point_set
                idx = 0
                for num_remaining_anchors in range(1, self.maximum_anchor-len(kf_indices)*self.num_blocks_per_row*self.num_blocks_per_column+1):
                    patch_interval = len(remaining_indices)*self.num_blocks_per_row*self.num_blocks_per_column // num_remaining_anchors
                    anchor_point_set = libvpx.AnchorPointSet.load(base_anchor_point_set, cache_profile_dir, '{}_nanchors_{}'.format(algorithm_type, anchor_point_set.get_num_anchor_points() + 1), self.num_blocks_per_row, self.num_blocks_per_column)
                    idx = 0
                    anchor_blocks = []
                    for _ in range(num_remaining_anchors):
                        anchor_point_set.add_anchor_point(frames[remaining_indices[idx//(self.num_blocks_per_row*self.num_blocks_per_column)]], (idx % (self.num_blocks_per_row*self.num_blocks_per_column)) // self.num_blocks_per_row, (idx % (self.num_blocks_per_row*self.num_blocks_per_column)) % self.num_blocks_per_row)
                        anchor_blocks.append((-1,remaining_indices[idx//(self.num_blocks_per_row*self.num_blocks_per_column)], (idx % (self.num_blocks_per_row*self.num_blocks_per_column)) // self.num_blocks_per_row, (idx % (self.num_blocks_per_row*self.num_blocks_per_column)) % self.num_blocks_per_row))
                        idx += patch_interval
                    anchor_point_set.save_cache_profile()
            print(f"Profile generated for {postfix}.")

        elif processing_mode=="measure_profile":
        
            anchor_point_sets = [libvpx.AnchorPointSet.create(frames, cache_profile_dir, f"{algorithm_type}_nanchors_{idx}", self.num_blocks_per_row, self.num_blocks_per_column) for idx in range(self.maximum_anchor+1)]
            for anchor_point_set in anchor_point_sets:
                anchor_point_set.load_from_file(os.path.join(cache_profile_dir, anchor_point_set.name+".profile"))

            #save low-resolution, super-resoluted, high-resolution frames to local storage
            libvpx.save_rgb_frame(self.vpxdec_path, self.dataset_dir, self.lr_video_name, skip=num_skipped_frames, limit=num_decoded_frames, postfix=postfix)
            libvpx.save_yuv_frame(self.vpxdec_path, self.dataset_dir, self.hr_video_name, self.output_width, self.output_height, num_skipped_frames, num_decoded_frames, postfix)
            #libvpx.setup_sr_frame(self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.model, postfix)
            
            self.measure_quality(anchor_point_sets, True, num_skipped_frames, num_decoded_frames, postfix)
            #remove images
            lr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, postfix)
            hr_image_dir = os.path.join(self.dataset_dir, 'image', self.hr_video_name, postfix)
            sr_image_dir = os.path.join(self.dataset_dir, 'image', self.lr_video_name, self.model.name, postfix)
            shutil.rmtree(lr_image_dir, ignore_errors=True)
            shutil.rmtree(hr_image_dir, ignore_errors=True)
            #shutil.rmtree(sr_image_dir, ignore_errors=True)

    def select_anchor_point_set(self, algorithm_type, chunk_idx, processing_mode, max_nemo_num_anchor_points=None, ssim_threshold=0.5):
        if algorithm_type in ['neuroscaler']:
            assert self.num_blocks_per_row == 1
            assert self.num_blocks_per_column == 1
        if chunk_idx is not None:
            if algorithm_type == 'preliminary_exp':
                all_sorted_intra, all_sorted_inter, all_sorted_key_and_altref, all_sorted_normal = self._select_anchor_point_set_palantir(chunk_idx)
            elif algorithm_type == 'palantir':
                self._select_anchor_point_set_palantir(chunk_idx, processing_mode=processing_mode, disable_weights=False, disable_texture_complexities=False)
            elif algorithm_type == 'palantir_wo_weight':
                self._select_anchor_point_set_palantir(chunk_idx, processing_mode=processing_mode, disable_weights=True, disable_texture_complexities=False)
            elif algorithm_type == 'palantir_wo_tc':
                self._select_anchor_point_set_palantir(chunk_idx, processing_mode=processing_mode, disable_weights=False, disable_texture_complexities=True)
            elif algorithm_type == 'neuroscaler':
                self._select_anchor_point_set_neuroscaler(chunk_idx, processing_mode)
            elif algorithm_type == 'key_uniform':
                self._select_anchor_point_set_key_uniform(chunk_idx, processing_mode)
        else:
            lr_video_path = os.path.join(self.dataset_dir, 'video', self.lr_video_name)
            lr_video_profile = get_video_profile(lr_video_path)

            num_chunks = int(math.ceil(lr_video_profile['duration'] / (self.gop / lr_video_profile['frame_rate'])))
            if num_chunks != 150:
                print(f"Warning: num_chunks={num_chunks} but we reset it to 150")
                num_chunks = 150

            all_sorted_intra =[]
            all_sorted_inter =[]
            all_sorted_key_and_altref =[]
            all_sorted_normal =[]

            for i in range(num_chunks):
                if algorithm_type == 'preliminary_exp':
                    sorted_intra, sorted_inter, sorted_key_and_altref, sorted_normal = self._select_anchor_point_set_palantir(i)
                    all_sorted_intra += sorted_intra
                    all_sorted_inter += sorted_inter
                    all_sorted_key_and_altref += sorted_key_and_altref
                    all_sorted_normal += sorted_normal
                elif algorithm_type == 'palantir':
                    self._select_anchor_point_set_palantir(i, processing_mode=processing_mode, disable_weights=False, disable_texture_complexities=False)
                elif algorithm_type == 'palantir_wo_weight':
                    self._select_anchor_point_set_palantir(i, processing_mode=processing_mode, disable_weights=True, disable_texture_complexities=False)
                elif algorithm_type == 'palantir_wo_tc':
                    self._select_anchor_point_set_palantir(i, processing_mode=processing_mode, disable_weights=False, disable_texture_complexities=True)
                elif algorithm_type == 'neuroscaler':
                    self._select_anchor_point_set_neuroscaler(i, processing_mode)
                elif algorithm_type == 'key_uniform':
                    self._select_anchor_point_set_key_uniform(i, processing_mode)
        if algorithm_type == 'preliminary_exp':
            fig, ax = joyplot([all_sorted_intra, all_sorted_inter, all_sorted_key_and_altref, all_sorted_normal], labels=["Intra-coded MBs", "Inter-coded MBs", "Keyframes & Altrefs", "Normal Frames"], colormap=sns.color_palette("crest", as_cmap=True))
            plt.xlabel('Degree of reference')
            plt.title("Test")
            plt.savefig("test.png")

            np.save("all_sorted_intra_0_29.npy", np.array(all_sorted_intra))
            np.save("all_sorted_inter_0_29.npy", np.array(all_sorted_inter))
            np.save("all_sorted_key_and_altref_0_29.npy", np.array(all_sorted_key_and_altref))
            np.save("all_sorted_normal_0_29.npy", np.array(all_sorted_normal))

    def measure_quality(self, anchor_point_sets, decode_block_cache, num_skipped_frames, num_decoded_frames, postfix):
        q0 = mp.Queue()
        q1 = mp.Queue()
        decoders = [mp.Process(target=libvpx.offline_cache_quality_mt, args=(q0, q1, self.vpxdec_path, self.dataset_dir, self.lr_video_name, self.hr_video_name, self.model.name, self.output_width, self.output_height)) for i in range(self.num_decoders)]
        for decoder in decoders:
            decoder.start()
        num_decoder_executions = 0
        for anchor_point_set in anchor_point_sets:
            q0.put((anchor_point_set.get_cache_profile_name(), num_skipped_frames, num_decoded_frames, postfix, num_decoder_executions, self.num_blocks_per_row, self.num_blocks_per_column, self.patch_width, self.patch_height, decode_block_cache))            
            num_decoder_executions += 1
        for _ in range(num_decoder_executions):
            item = q1.get()
            idx = item[0]
            quality = item[1]
            anchor_point_sets[idx].set_measured_quality(quality)
            anchor_point_sets[idx].set_cache_profile_name(anchor_point_sets[idx].get_cache_profile_name()+f"_{np.average(quality)}")
            anchor_point_sets[idx].save_cache_profile()
        for decoder in decoders:
            q0.put('end')
        for decoder in decoders:
            decoder.join()