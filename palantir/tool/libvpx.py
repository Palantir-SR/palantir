import math
import os
import glob
import struct
import copy
import subprocess
import shlex
import time
import filecmp
import gc
import cv2
import numpy as np
import tensorflow as tf
import queue
from abc import abstractmethod, ABCMeta
from enum import Enum
import random
import shutil
import multiprocessing
import sys
sys.setrecursionlimit(30000)
import torch
import torch.nn as nn
import threading
import matplotlib.pyplot as plt
import networkx as nx

from palantir.tool.video import get_video_profile
from palantir.tool.motion import blockMotion
from palantir.dnn.dataset import single_raw_dataset, single_raw_dataset_with_name

class Frame():
    def __init__(self, video_index, super_index, frame_type, frame_size=None):
        self.video_index = video_index
        self.super_index= super_index
        self.setImageFn(frame_type)
        self.frame_size = frame_size

    @property
    def name(self):
        return '{}.{}'.format(self.video_index, self.super_index)

    def setImageFn(self, frame_type):
        self.frame_type = frame_type
        if frame_type == "key_frame" or frame_type == "normal_frame":
            self.imageFn = f"{str(self.video_index).zfill(4)}.raw"
        elif frame_type == "alternative_reference_frame":
            self.imageFn = f"{str(self.video_index).zfill(4)}_{self.super_index}.raw"
        else:
            self.imageFn = None

    def __lt__(self, other):
        if self.video_index == other.video_index:
            return self.super_index < other.super_index
        else:
            return self.video_index < other.video_index

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.video_index == other.video_index and self.super_index == other.super_index:
                return True
            else:
                return False
        else:
            return False

class AnchorPointSet():
    def __init__(self, frames, anchor_point_set, save_dir, name, num_blocks_per_row, num_blocks_per_column):
        self.num_blocks_per_row = num_blocks_per_row
        self.num_blocks_per_column = num_blocks_per_column

        assert (frames is None or anchor_point_set is None)
        if frames is not None:
            self.frames = frames
            self.anchor_points = {}
            self.anchor_points_list = []
            for frame in self.frames:
                self.anchor_points[frame.name] = []
            self.estimated_quality = None
            self.measured_quality = None

        if anchor_point_set is not None:
            self.frames = copy.deepcopy(anchor_point_set.frames)
            self.anchor_points = copy.deepcopy(anchor_point_set.anchor_points)
            self.anchor_points_list = copy.deepcopy(anchor_point_set.anchor_points_list)
            self.estimated_quality = copy.deepcopy(anchor_point_set.estimated_quality)
            self.measured_quality = copy.deepcopy(anchor_point_set.measured_quality)

        self.save_dir = save_dir
        self.name = name

    @classmethod
    def create(cls, frames, save_dir, name, num_blocks_per_row, num_blocks_per_column):
        return cls(frames, None, save_dir, name, num_blocks_per_row, num_blocks_per_column)

    @classmethod
    def load(cls, anchor_point_set, save_dir, name, num_blocks_per_row, num_blocks_per_column):
        return cls(None, anchor_point_set, save_dir, name, num_blocks_per_row, num_blocks_per_column)

    @property
    def path(self):
        return os.path.join(self.save_dir, self.name)

    def add_anchor_point(self, frame, row_idx, col_idx):
        self.anchor_points[frame.name].append((row_idx, col_idx))
        self.anchor_points_list.append((frame, row_idx, col_idx))

    def merge_another_anchor_point_set(self, another):
        for frame in self.frames:
            for row_idx, col_idx in another.anchor_points[frame.name]:
                if (row_idx, col_idx) not in self.anchor_points[frame.name]:
                    self.anchor_points[frame.name].append((row_idx, col_idx))
                    self.anchor_points_list.append((frame, row_idx, col_idx))

    def calc_recall(self, anotherAnchorPointSet):
        num_common_anchors = 0
        num_total_anchors = 0
        if anotherAnchorPointSet.num_blocks_per_column == self.num_blocks_per_column and anotherAnchorPointSet.num_blocks_per_row == self.num_blocks_per_row:
            for frame in self.frames:
                num_total_anchors += len(self.anchor_points[frame.name])
                for row_idx, col_idx in self.anchor_points[frame.name]:
                    if (row_idx, col_idx) in anotherAnchorPointSet.anchor_points[frame.name]:
                        num_common_anchors += 1
        elif anotherAnchorPointSet.num_blocks_per_column == 1 and anotherAnchorPointSet.num_blocks_per_row == 1:
            for frame in self.frames:
                num_total_anchors += len(self.anchor_points[frame.name])
                if (0, 0) in anotherAnchorPointSet.anchor_points[frame.name]:
                    num_common_anchors += len(self.anchor_points[frame.name])
        else:
            raise NotImplementedError
        return num_common_anchors/num_total_anchors

    def is_anchor_point(self, frame, row_idx, col_idx):
        return (row_idx, col_idx) in self.anchor_points[frame.name]

    def get_num_anchor_points(self):
        num_anchor_points = 0
        for frame in self.frames:
            num_anchor_points += len(self.anchor_points[frame.name])
        return num_anchor_points

    def get_cache_profile_name(self):
        return self.name

    def set_cache_profile_name(self, name):
        self.name = name

    def get_estimated_quality(self):
        return self.estimated_quality

    def get_measured_quality(self):
        return self.measured_quality

    def set_estimated_quality(self, quality):
        self.estimated_quality = quality

    def set_measured_quality(self, quality):
        self.measured_quality = quality

    def save_cache_profile(self):
        path = os.path.join(self.save_dir, '{}.profile'.format(self.name))

        num_remained_bits = 8 - (len(self.frames)*self.num_blocks_per_row*self.num_blocks_per_column % 8)
        num_remained_bits = num_remained_bits % 8

        with open(path, "wb") as f:
            f.write(struct.pack("=I", num_remained_bits))

            byte_value = 0
            for i, frame in enumerate(self.frames):
                frame_anchor_blocks = self.anchor_points[frame.name]
                for col_idx in range(self.num_blocks_per_row):
                    for row_idx in range(self.num_blocks_per_column):
                        idx = i*self.num_blocks_per_row*self.num_blocks_per_column + col_idx*self.num_blocks_per_column + row_idx
                        if (row_idx, col_idx) in frame_anchor_blocks:
                            byte_value += 1 << (idx % 8)
                        if idx % 8 == 7:
                            f.write(struct.pack("=B", byte_value))
                            byte_value = 0

            if num_remained_bits != 0:
                f.write(struct.pack("=B", byte_value))

    def load_from_file(self, profile_path):
        with open(profile_path, "rb") as f:
            num_remained_bits, = struct.unpack("=I", f.read(4))
            byte_value = 0
            idx = 0
            for i, frame in enumerate(self.frames):
                frame_anchor_blocks = self.anchor_points[frame.name]
                for col_idx in range(self.num_blocks_per_row):
                    for row_idx in range(self.num_blocks_per_column):
                        if idx % 8 == 0:
                            byte_value, = struct.unpack("=B", f.read(1))

                        is_anchor_point = (byte_value >> (idx % 8)) & 1
                        if is_anchor_point:
                            frame_anchor_blocks.append((row_idx, col_idx))
                            self.anchor_points_list.append((frame, row_idx, col_idx))
                        idx += 1

                self.anchor_points[frame.name] = frame_anchor_blocks

    def remove_cache_profile(self):
        cache_profile_path = os.path.join(self.save_dir, '{}.profile'.format(self.name))
        if os.path.exists(cache_profile_path):
            os.remove(cache_profile_path)

    def __lt__(self, other):
        return self.count_anchor_points() < other.count_anchor_points()

class EstimatorNetwork(nn.Module):
    def __init__(self, num_blocks_per_row, num_blocks_per_column, weights, biases, root_frames_names, non_root_frames_references, frames, intra_parallelism, use_sparse_matrix=True):
        super(EstimatorNetwork, self).__init__()
        self.num_blocks_per_row = num_blocks_per_row
        self.num_blocks_per_column = num_blocks_per_column
        self.num_nodes_per_layer = self.num_blocks_per_row*self.num_blocks_per_column
        self.weights = weights
        self.biases = biases
        self.root_frames_names = root_frames_names
        self.non_root_frames_references = non_root_frames_references
        self.frames = frames
        layers = {}
        self.intra_parallelism = intra_parallelism
        self.use_sparse_matrix = use_sparse_matrix
        if self.intra_parallelism:
            if self.use_sparse_matrix:
                for key, value in self.weights.items():
                    self.weights[key] = value.to_sparse()
            else:
                for frame_name, reference_frames in self.non_root_frames_references.items():
                    for reference_frame in reference_frames:
                        layer = nn.Linear(self.num_nodes_per_layer, self.num_nodes_per_layer, bias=False)
                        layer.weight = nn.parameter.Parameter(self.weights[f'{reference_frame.name}+{frame_name}'])
                        layers[f'{reference_frame.name}+{frame_name}'.replace('.', '-')] = layer
                self.layers = nn.ModuleDict(layers)
                self.num_layers = len(self.layers)
        else:
            self.DAG = nx.DiGraph()
            for frame_idx, frame in enumerate(self.frames):
                tc = self.biases[frame.name]
                self.DAG.add_nodes_from([(f"{frame.name}+{idx}", {"tc":tc[idx], "frame_idx": frame_idx}) for idx in range(self.num_blocks_per_row*self.num_blocks_per_column)])
            
            for frame_idx, frame in enumerate(self.frames):
                if frame.frame_type != 'key_frame':
                    for reference_frame in self.non_root_frames_references[frame.name]:
                        weight = self.weights[f'{reference_frame.name}+{frame.name}']
                        for dst_patch in range(self.num_blocks_per_row*self.num_blocks_per_column):
                            for src_patch in range(self.num_blocks_per_row*self.num_blocks_per_column):
                                if weight[dst_patch][src_patch] != 0:
                                    self.DAG.add_weighted_edges_from([(f"{reference_frame.name}+{src_patch}", f"{frame.name}+{dst_patch}", weight[dst_patch][src_patch])])

    def forward(self, selected_anchor_points, candidate_anchor_points):
        '''
        @ param anchor_points: [(frame_idx, row_idx, col_idx), ...]
        @ param candidate_anchor_points: [(frame_idx, row_idx, col_idx), ...]
        '''
        def get_anchor_blocks_in_current_frame(anchor_points, frame_idx):
            result = []
            for idx, value in enumerate(anchor_points):
                if value[0] == frame_idx:
                    result.append((value[1], value[2], idx))
            return result
        with torch.no_grad():
            self.eval()
            batch_size = len(candidate_anchor_points)
            if self.intra_parallelism:
                cumulated_errors = {}
                for frame_idx, frame in enumerate(self.frames):
                    # weights & biases & anchors
                    current_error = torch.squeeze(self.biases[frame.name], axis=0).repeat(batch_size, 1)
                    if not frame.name in self.root_frames_names:
                        for reference_frame in self.non_root_frames_references[frame.name]:      
                            if self.use_sparse_matrix:
                                current_error = current_error + torch.sparse.mm(self.weights[f'{reference_frame.name}+{frame.name}'], cumulated_errors[reference_frame.name].t()).t()
                            else:
                                current_error = current_error + self.layers[f'{reference_frame.name}+{frame.name}'.replace('.', '-')](cumulated_errors[reference_frame.name])
                    selected_anchor_points_in_current_frame = get_anchor_blocks_in_current_frame(selected_anchor_points, frame_idx)
                    for row_idx, col_idx, _ in selected_anchor_points_in_current_frame:
                        current_error[:, row_idx*self.num_blocks_per_row+col_idx] = 0
                    selected_candidates_in_current_frame = get_anchor_blocks_in_current_frame(candidate_anchor_points, frame_idx)
                    for row_idx, col_idx, batch_idx in selected_candidates_in_current_frame:
                        current_error[batch_idx, row_idx*self.num_blocks_per_row+col_idx] = 0
                    cumulated_errors[frame.name] = current_error
                sum_values = 0
                for frame_idx, frame in enumerate(self.frames):
                    if frame.frame_type == "alternative_reference_frame":
                        continue
                    else:
                        sum_values = sum_values + torch.sum(cumulated_errors[frame.name], axis=1)
                return sum_values
            else:
                assert batch_size == 1, "INTRA_PARALLELISM is the basis of INTER_PARALLELISM"
                sum_values = 0
                for frame_idx, frame in enumerate(self.frames):
                    for row_idx in range(self.num_blocks_per_column):
                        for col_idx in range(self.num_blocks_per_row):
                            idx = col_idx*self.num_blocks_per_column+row_idx
                            node = f"{frame.name}+{idx}"
                            _node = self.DAG.nodes[node]
                            error = 0
                            if (frame_idx, row_idx, col_idx) in selected_anchor_points or (frame_idx, row_idx, col_idx) in candidate_anchor_points:
                                pass
                            else:
                                parents = list(self.DAG.predecessors(node))
                                error = _node["tc"]
                                for parent in parents:
                                    error += self.DAG.nodes[parent]["error"] * self.DAG.edges[parent, node]['weight']
                            _node["error"] = error
                            sum_values += error
                return torch.Tensor([sum_values])

class AnchorBlockIndicator():

    def __init__(self, num_blocks_per_row, num_blocks_per_column, num_frames, frame_range=None):
        self.num_blocks_per_row = num_blocks_per_row
        self.num_blocks_per_column = num_blocks_per_column
        self.num_frames = num_frames
        self.is_anchor_matrices = np.zeros((self.num_frames, self.num_blocks_per_row*self.num_blocks_per_column))
        self.frame_range = frame_range

    def get_anchor_blocks(self):
        result = []
        for frame_idx in range(self.num_frames):
            for row_idx in range(self.num_blocks_per_column):
                for col_idx in range(self.num_blocks_per_row):
                    if self.is_anchor_matrices[frame_idx][row_idx*self.num_blocks_per_row+col_idx] == 1:
                        result.append((frame_idx, row_idx, col_idx))
        return result

    def add_anchor_block(self, frame_idx, row_idx, col_idx):
        if self.frame_range is not None:
            assert frame_idx in self.frame_range
        self.is_anchor_matrices[frame_idx][row_idx*self.num_blocks_per_row+col_idx] = 1

    def is_anchor_block(self, frame_idx, row_idx, col_idx):
        return self.is_anchor_matrices[frame_idx][row_idx*self.num_blocks_per_row+col_idx] == 1

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        while True:
            #print(f"Maximum={self.num_frames*self.num_blocks_per_row*self.num_blocks_per_column}")
            if self.idx == self.num_frames*self.num_blocks_per_row*self.num_blocks_per_column:
                raise StopIteration
            idx = self.idx
            frame_idx = idx // (self.num_blocks_per_row*self.num_blocks_per_column)
            if idx % (self.num_blocks_per_row*self.num_blocks_per_column) == 0:
                if self.frame_range is not None:
                    if frame_idx not in self.frame_range:
                        self.idx += (self.num_blocks_per_row*self.num_blocks_per_column)
                        continue
            idx = idx - frame_idx*self.num_blocks_per_row*self.num_blocks_per_column
            self.idx = self.idx + 1
            if self.is_anchor_matrices[frame_idx][idx] == 0:
                row_idx = idx // self.num_blocks_per_row
                col_idx = idx - row_idx*self.num_blocks_per_row
                return frame_idx, row_idx, col_idx

class block():
    def __init__(self, frame, x, y, w, h, type):
        self.frame = frame
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.type = type
        self.name = self.frame.name+f"-x-{x}-y-{y}"

class fastGraph():
    def __init__(self, vpxdec_path, num_blocks_per_row, num_blocks_per_column, patch_width, patch_height, dataset_dir, postfix, lr_video_name, hr_video_name, lr_height, lr_width, output_height, output_width, export_path, gop, skip, limit, disable_weights=False, disable_texture_complexities=False, intra_parallelism=True, inter_parallelism=True, schedulingInterval=None):
        self.vpxdec_path = vpxdec_path
        self.num_blocks_per_row = num_blocks_per_row
        self.num_blocks_per_column = num_blocks_per_column
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.dataset_dir = dataset_dir
        self.lr_video_name = lr_video_name
        self.hr_video_name = hr_video_name
        self.postfix = postfix
        self.lr_image_dir = os.path.join(self.dataset_dir, "image", self.lr_video_name, self.postfix) # the directory containing low-resolution images
        self.lr_height = lr_height
        self.lr_width = lr_width
        self.hr_image_dir = os.path.join(self.dataset_dir, "image", self.hr_video_name, self.postfix)
        self.output_height = output_height
        self.output_width = output_width
        self.scale = self.output_height // self.lr_height
        self.nodes = {}
        self.edges = {}
        self.key_frame_blocks = []
        self.export_path = export_path
        self.matrix_multiplication_time = 0
        self.gop = gop
        self.skip = skip
        self.limit = limit
        self.disable_weights = disable_weights
        self.disable_texture_complexities = disable_texture_complexities
        self.intra_parallelism = intra_parallelism
        self.inter_parallelism = inter_parallelism
        self.bad_region = 0
        self.schedulingInterval = schedulingInterval
        if self.schedulingInterval == 60:
            self.sizeof_frame_ranges = 8
        else:
            raise NotImplementedError
        assert not (intra_parallelism==False and inter_parallelism==True), "INTRA_PARALLELISM is the prerequisite of INTER_PARALLELISM"

    def construct_graph_through_metadata(self, model):

        self.weights = {} # key: reference_frame.name + current_frame.name
        self.biases = {} # key: current_frame.name
        self.dnndistortions = {}

        '''
        self.root_blocks = []

        for frame_idx, frame in enumerate(self.frames):
            if frame.frame_type == 'keyframe':
                for idx1 in range(self.num_blocks_per_column):
                    for idx2 in range(self.num_blocks_per_row):
                        self.key_frame_blocks.append((frame_idx, idx1, idx2))

            if frame.name in root_frames_names: # keyframe or intra-only frame
                for idx1 in range(self.num_blocks_per_column):
                    for idx2 in range(self.num_blocks_per_row):
                        self.indicator.add_anchor_block(frame_idx, idx1, idx2)
                        self.root_blocks.append((frame_idx, idx1, idx2))
        '''

        weights, biases = self.offline_finegrained_metadata(self.patch_width, self.patch_height, self.num_blocks_per_row, self.num_blocks_per_column, self.vpxdec_path, self.dataset_dir, self.lr_video_name, skip=self.skip, limit=self.limit, postfix=self.postfix)
            
        self.weights.update(weights)
        self.biases.update(biases)

        self.frame_weights = self.merge_by_frame(self.weights, [1,1], self.num_blocks_per_row*self.num_blocks_per_column)
        self.frame_biases = self.merge_by_frame(self.biases, [1], 1)
        #self.frame_indicator = AnchorBlockIndicator(1, 1, len(self.frames))
        self.frame_indicator = AnchorBlockIndicator(1, 1, self.schedulingInterval)
        self.frameEstimatorNetwork = EstimatorNetwork(1, 1, self.frame_weights, self.frame_biases, self.root_frames_names, self.non_root_frames_references, self.frames, intra_parallelism = self.intra_parallelism)
        self.set_frame_range()
        
        #self.indicator = AnchorBlockIndicator(self.num_blocks_per_row, self.num_blocks_per_column, len(self.frames), self.frame_range)
        self.indicator = AnchorBlockIndicator(self.num_blocks_per_row, self.num_blocks_per_column, self.schedulingInterval, self.frame_range)
        self.estimatorNetwork = EstimatorNetwork(self.num_blocks_per_row, self.num_blocks_per_column, self.weights, self.biases, self.root_frames_names, self.non_root_frames_references, self.frames, intra_parallelism = self.intra_parallelism)

    def set_frame_range(self):
        start_time = time.time()
        self.frameEstimatorNetwork.eval()
        self.frame_range = []
        with torch.no_grad():
            for _ in range(self.sizeof_frame_ranges):
                batch_size = int(self.sizeof_frame_ranges*2) if self.inter_parallelism else 1
                candidate_anchor_frames = []
                batch_candidate_anchor_frames = []
                values = []
                for idx, (frame_idx, row_idx, col_idx) in enumerate(iter(self.frame_indicator)):
                    candidate_anchor_frames.append((frame_idx, row_idx, col_idx))
                    batch_candidate_anchor_frames.append((frame_idx, row_idx, col_idx))
                    if (idx+1) % batch_size == 0:
                        time1 = time.time()
                        values.append(self.frameEstimatorNetwork.forward(self.frame_indicator.get_anchor_blocks(), batch_candidate_anchor_frames))
                        time2 = time.time()
                        batch_candidate_anchor_frames = []
                if len(batch_candidate_anchor_frames) > 0:
                    values.append(self.frameEstimatorNetwork.forward(self.frame_indicator.get_anchor_blocks(), batch_candidate_anchor_frames))
                values = torch.cat(tuple(values))
                frame_idx, row_idx, col_idx = candidate_anchor_frames[int(torch.min(values, 0)[1])]
                self.frame_indicator.add_anchor_block(frame_idx, row_idx, col_idx)
                self.frame_range.append(frame_idx)
        
        end_time = time.time()

    def merge_by_frame(self, attrs, shape, factor):
        merged_attrs = {}
        for key, value in attrs.items():
            merged_attrs[key] = torch.sum(value).reshape(shape)/factor
        return merged_attrs

    def offline_finegrained_metadata(self, patch_width, patch_height, num_blocks_per_row, num_blocks_per_column, vpxdec_path, dataset_dir, input_video_name, skip=None, limit=None, postfix=None):

        cache_profile_dir = os.path.join(dataset_dir, 'profile', input_video_name, "no_model")
        if postfix is not None:
            cache_profile_dir = os.path.join(cache_profile_dir, postfix)
        os.makedirs(cache_profile_dir, exist_ok=True)

        anchor_point_set = AnchorPointSet.create([Frame(i, 0, 'normal_frame') for i in range(self.gop*2)], cache_profile_dir, 'no_anchor', num_blocks_per_row, num_blocks_per_column)
        anchor_point_set.save_cache_profile()
        cache_profile_name = anchor_point_set.get_cache_profile_name()

        #log file
        model_name = "no_model"
        if postfix is not None:
            log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, postfix, os.path.basename(cache_profile_name), 'metadata.txt')
            finegrained_log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, postfix, os.path.basename(cache_profile_name), 'finegrained_metadata.txt')
        else:
            log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, os.path.basename(cache_profile_name), 'metadata.txt')
            finegrained_log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, os.path.basename(cache_profile_name), 'finegrained_metadata.txt')
        #run sr-integrated decoder
        input_video_path = os.path.join(dataset_dir, 'video', input_video_name)
        input_video_profile = get_video_profile(input_video_path)
        input_video_width = input_video_profile['width']
        input_video_height = input_video_profile['height']

        command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={} \
        --input-video-name={} --decode-mode=decode_block_cache --dnn-mode=offline_dnn --cache-mode=profile_cache \
        --output-width={} --output-height={} --save-metadata --dnn-name={} --dnn-scale=1 --cache-profile-name={} --threads=1 --patch-width={} --patch-height={} --npatches-per-row={} --npatches-per-column={} --gop={} --save-finegrained-metadata --save-metadata --save-metadata-framesize'.format(vpxdec_path, \
                            dataset_dir, input_video_name, input_video_width, input_video_height, "no_model", cache_profile_name, patch_width, patch_height, num_blocks_per_row, num_blocks_per_column, self.gop)
        if skip is not None:
            command += ' --skip={}'.format(skip)
        if limit is not None:
            command += ' --limit={}'.format(limit)
        if postfix is not None:
            command += ' --postfix={}'.format(postfix)
        subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

        self.frames = load_frame_index(dataset_dir, input_video_name, postfix, log_path=log_path, load_frame_size=False)

        weights = {}
        biases = {}
            
        self.root_frames_names, self.non_root_frames_references, self.frame_dependencies, self.non_root_frames_references_all = get_frame_dependency(dataset_dir, input_video_name, self.frames, postfix, all=True, log_path=log_path)
        with open(finegrained_log_path, 'r') as f:
            lines = f.readlines()
            for idx, frame in enumerate(self.frames):
                bias = [float(item) for item in lines[4*idx].strip().split("\t")]
                if np.sum(np.array(bias)) != 0:
                    biases[frame.name] = torch.Tensor(bias) / 100000
                else:
                    biases[frame.name] = torch.zeros(num_blocks_per_row*num_blocks_per_column)
                if frame.frame_type == "key_frame":
                    continue
                else:
                    for reference_idx, reference_frame in enumerate(self.non_root_frames_references_all[frame.name]):
                        data = torch.Tensor([float(item) for item in lines[4*idx+1+reference_idx].strip().split("\t")]).view(num_blocks_per_row*num_blocks_per_column, num_blocks_per_row*num_blocks_per_column)
                        key = f'{reference_frame.name}+{frame.name}'
                        if not self.disable_weights:
                            if key in weights.keys():
                                weights[key] += data
                            else:
                                weights[key] = data
        if not self.disable_weights:
            for key, value in weights.items():
                weights[key] = value / (patch_width*patch_height)

        if self.disable_weights:
            # update self.non_root_frames_references, self.frame_dependencies, and self.non_root_frames_references_all
            self.set_sequential_dependency()
            for idx in range(len(self.frames)-1):
                key = f'{self.frames[idx].name}+{self.frames[idx+1].name}'
                data = torch.eye(num_blocks_per_row*num_blocks_per_column)
                weights[key] = data
        if self.disable_texture_complexities:
            for idx, frame in enumerate(self.frames):
                biases[frame.name] = torch.ones(num_blocks_per_row*num_blocks_per_column)

        anchor_point_set.remove_cache_profile()
        return weights, biases

    def set_sequential_dependency(self):
        non_root_frames_references = {}
        non_root_frames_references_all = {}
        frame_dependencies = {} # key: frame_name; value: list of frames depending on it
        frame_map = {}
        num_frames = len(self.frames)
        for idx in range(num_frames):
            if idx != num_frames-1:
                non_root_frames_references[self.frames[idx+1].name] = [self.frames[idx]]
                non_root_frames_references_all[self.frames[idx+1].name] = [self.frames[idx]]
                frame_dependencies[self.frames[idx].name] = [self.frames[idx+1]]
            else:
                frame_dependencies[self.frames[idx].name] = []
        self.frame_dependencies = frame_dependencies
        self.non_root_frames_references = non_root_frames_references
        self.non_root_frames_references_all = non_root_frames_references_all

    def search_new_anchor_blocks(self, block_type="all"):
        assert block_type in ["all", "alternative_reference_frame_only", "normal_frame_only"]
        if hasattr(self, "searching_block_type"):
            assert self.searching_block_type == block_type
        else:
            self.searching_block_type = block_type
        self.estimatorNetwork.eval()
        with torch.no_grad():
            batch_size = int(self.num_blocks_per_row*self.num_blocks_per_column*15) if self.inter_parallelism else 1
            candidate_anchor_points = []
            batch_candidate_anchor_points = []
            values = []
            no_choices = True
            for idx, (frame_idx, row_idx, col_idx) in enumerate(iter(self.indicator)):
                if self.searching_block_type == "all" or (self.frames[frame_idx].frame_type == "alternative_reference_frame" and self.searching_block_type == "alternative_reference_frame_only") or (self.frames[frame_idx].frame_type == "normal_frame" and self.searching_block_type == "normal_frame_only"):
                    no_choices = False
                else:
                    continue
                candidate_anchor_points.append((frame_idx, row_idx, col_idx))
                batch_candidate_anchor_points.append((frame_idx, row_idx, col_idx))
                if (idx+1) % batch_size == 0:
                    values.append(self.estimatorNetwork.forward(self.indicator.get_anchor_blocks(), batch_candidate_anchor_points))
                    batch_candidate_anchor_points = []
            if len(batch_candidate_anchor_points) > 0:
                values.append(self.estimatorNetwork.forward(self.indicator.get_anchor_blocks(), batch_candidate_anchor_points))
            if no_choices:
                return None, None, None, None
            values = torch.cat(tuple(values))
            frame_idx, row_idx, col_idx = candidate_anchor_points[int(torch.min(values, 0)[1])]
            self.indicator.add_anchor_block(frame_idx, row_idx, col_idx)

            return -1, frame_idx, row_idx, col_idx

    def get_anchor_point_set(self, cache_profile_dir, profile_name):
        anchor_point_set = AnchorPointSet.create(self.frames, cache_profile_dir, profile_name, self.num_blocks_per_row, self.num_blocks_per_column)
        for frame_idx, row_idx, col_idx in self.indicator.get_anchor_blocks():
            anchor_point_set.add_anchor_point(self.frames[frame_idx], row_idx=row_idx, col_idx=col_idx)
        return anchor_point_set

    def get_num_anchor_points(self):
        return len(self.indicator.get_anchor_blocks())

def load_frame_index(dataset_dir, video_name, postfix=None, load_frame_size=False, log_path=None):
    frames = []
    if log_path is None:
        if postfix is None:
            log_path = os.path.join(dataset_dir, 'log', video_name, 'metadata.txt')
        else:
            log_path = os.path.join(dataset_dir, 'log', video_name, postfix, 'metadata.txt')
    
    f = open(log_path, 'r')
    lines = f.readlines()
    for idx, line in enumerate(lines):
        line = line.strip()
        current_video_frame = int(line.split('\t')[0])
        current_super_frame = int(line.split('\t')[1])
        frame_type = line.split('\t')[-1]
        if load_frame_size:
            frame_size = float(line.split('\t')[-2])
            frames.append(Frame(current_video_frame, current_super_frame, frame_type, frame_size))
        else:
            frames.append(Frame(current_video_frame, current_super_frame, frame_type))
    return frames

def get_frame_dependency(dataset_dir, video_name, frames, postfix=None, all=False, log_path=None):
    root_frames_names = []
    non_root_frames_references = {} # key: frame_name; value: list of frames on which it depends
    if all:
        non_root_frames_references_all = {}
    frame_dependencies = {} # key: frame_name; value: list of frames depending on it
    frame_map = {}
    for frame in frames:
        frame_map[frame.name] = frame

    if log_path is None:
        if postfix is None:
            log_path = os.path.join(dataset_dir, 'log', video_name, 'metadata.txt')
        else:
            log_path = os.path.join(dataset_dir, 'log', video_name, postfix, 'metadata.txt')
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            result = line.strip().split('\t')
            current_video_frame = int(result[0])
            current_super_frame = int(result[1])
            frame_name = '{}.{}'.format(current_video_frame, current_super_frame)
            frame_dependencies[frame_name] = []
            if len(result) >= 12:
                ref_frame_names = []
                if all:
                    non_root_frames_references_all[frame_name] = []
                for i in range(3):
                    ref_video_frame = int(result[2*i+5])
                    ref_super_frame = int(result[2*i+6])
                    ref_frame_name = '{}.{}'.format(ref_video_frame, ref_super_frame)
                    if all:
                        non_root_frames_references_all[frame_name].append(frame_map[ref_frame_name])
                    if not ref_frame_name in ref_frame_names:
                        ref_frame_names.append(ref_frame_name)
                non_root_frames_references[frame_name] = []
                for ref_frame_name in ref_frame_names:
                    non_root_frames_references[frame_name].append(frame_map[ref_frame_name])
            else:
                root_frames_names.append(frame_name)
    for src, dsts in non_root_frames_references.items():
        for dst in dsts:
            if not frame_map[src] in frame_dependencies[dst.name]:
                frame_dependencies[dst.name].append(frame_map[src])

    if all:
        return root_frames_names, non_root_frames_references, frame_dependencies, non_root_frames_references_all
    else:
        return root_frames_names, non_root_frames_references, frame_dependencies

def save_rgb_frame(vpxdec_path, dataset_dir, video_name, output_width=None, output_height=None, skip=None, limit=None, postfix=None, save_frame_size=False):
    video_path = os.path.join(dataset_dir, 'video', video_name)
    video_profile = get_video_profile(video_path)

    command = '{} --codec=vp9 --noblit --frame-buffers=50 --npatches-per-row=1 --npatches-per-column=1 --patch-width=0 --patch-height=0 --dataset-dir={}  \
        --input-video-name={} --threads={} --save-rgbframe --save-metadata'.format(vpxdec_path, dataset_dir, video_name, get_num_threads(video_profile['height']))

    if skip is not None:
        command += ' --skip={}'.format(skip)
    if limit is not None:
        command += ' --limit={}'.format(limit)
    if postfix is not None:
        command += ' --postfix={}'.format(postfix)
    if output_width is not None:
        command += ' --output-width={}'.format(output_width)
    if output_height is not None:
        command += ' --output-height={}'.format(output_height)
    if save_frame_size:
        command += ' --save-metadata-framesize'
    print(command)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def save_residual(neuroscaler_vpxdec_path, dataset_dir, video_name, skip=None, limit=None, postfix=None):
    video_path = os.path.join(dataset_dir, 'video', video_name)
    #log file
    if postfix is not None:
        log_path = os.path.join(dataset_dir, 'log', video_name, postfix, 'residual.txt')
    else:
        log_path = os.path.join(dataset_dir, 'log', video_name, 'residual.txt')

    command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={}  \
        --input-video-name={} --threads=1 --save-metadata --save-metadata-framesize'.format(neuroscaler_vpxdec_path, dataset_dir, video_name)
    if skip is not None:
        command += ' --skip={}'.format(skip)
    if limit is not None:
        command += ' --limit={}'.format(limit)
    if postfix is not None:
        command += ' --postfix={}'.format(postfix)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def save_yuv_frame(vpxdec_path, dataset_dir, video_name, output_width=None, output_height=None, skip=None, limit=None, postfix=None):
    video_path = os.path.join(dataset_dir, 'video', video_name)
    video_profile = get_video_profile(video_path)

    command = '{} --codec=vp9 --npatches-per-row=1 --npatches-per-column=1 --patch-width=0 --patch-height=0 --noblit --frame-buffers=50 --dataset-dir={}  \
        --input-video-name={} --threads={} --save-yuvframe --save-metadata'.format(vpxdec_path, dataset_dir, video_name, get_num_threads(video_profile['height']))
    if skip is not None:
        command += ' --skip={}'.format(skip)
    if limit is not None:
        command += ' --limit={}'.format(limit)
    if postfix is not None:
        command += ' --postfix={}'.format(postfix)
    if output_width is not None:
        command += ' --output-width={}'.format(output_width)
    if output_height is not None:
        command += ' --output-height={}'.format(output_height)
    print(command)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def save_metadata(vpxdec_path, dataset_dir, video_name, skip=None, limit=None, postfix=None, save_frame_size=False):
    video_path = os.path.join(dataset_dir, 'video', video_name)
    video_profile = get_video_profile(video_path)

    command = '{} --codec=vp9 --noblit --frame-buffers=50 --npatches-per-row=1 --npatches-per-column=1 --patch-width=0 --patch-height=0 --dataset-dir={}  \
        --input-video-name={} --threads={} --save-metadata'.format(vpxdec_path, dataset_dir, video_name, get_num_threads(video_profile['height']))

    if skip is not None:
        command += ' --skip={}'.format(skip)
    if limit is not None:
        command += ' --limit={}'.format(limit)
    if postfix is not None:
        command += ' --postfix={}'.format(postfix)
    if save_frame_size:
        command += ' --save-metadata-framesize'
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

def setup_sr_frame(vpxdec_path, dataset_dir, video_name, model, postfix=None):
    if postfix is None:
        lr_image_dir = os.path.join(dataset_dir, 'image', video_name)
        sr_image_dir = os.path.join(dataset_dir, 'image', video_name, model.name)
    else:
        lr_image_dir = os.path.join(dataset_dir, 'image', video_name, postfix)
        sr_image_dir = os.path.join(dataset_dir, 'image', video_name, model.name, postfix)
    os.makedirs(sr_image_dir, exist_ok=True)

    video_path = os.path.join(dataset_dir, 'video', video_name)
    video_profile = get_video_profile(video_path)

    single_raw_ds = single_raw_dataset_with_name(lr_image_dir, video_profile['width'], video_profile['height'], 3, exp='.raw')
    for idx, img in enumerate(single_raw_ds):
        lr = img[0]
        lr = tf.cast(lr, tf.float32)
        '''
        if os.path.exists(os.path.join(sr_image_dir, os.path.basename(img[1].numpy()[0].decode()))):
            continue
        '''

        print(img[1].numpy()[0].decode())
        sr = model(lr)

        sr = tf.clip_by_value(sr, 0, 255)
        sr = tf.round(sr)
        sr = tf.cast(sr, tf.uint8)

        sr_image = tf.squeeze(sr).numpy()
        name = os.path.basename(img[1].numpy()[0].decode())
        sr_image.tofile(os.path.join(sr_image_dir, name))

        #validate
        #sr_image = tf.image.encode_png(tf.squeeze(sr))
        #tf.io.write_file(os.path.join(sr_image_dir, '{0:04d}.png'.format(idx+1)), sr_image)

def bilinear_quality_vmaf(vpxdec_path, dataset_dir, input_video_name, reference_video_name,
                               output_width, output_height, skip=None, limit=None, postfix=None, save_quality=True, vmaf=False):
    #log file
    if postfix is not None:
        log_path = os.path.join(dataset_dir, 'log', input_video_name, postfix, 'quality.txt')
    else:
        log_path = os.path.join(dataset_dir, 'log', input_video_name, 'quality.txt')

    #run sr-integrated decoder
    input_video_path = os.path.join(dataset_dir, 'video', input_video_name)
    input_video_profile = get_video_profile(input_video_path)

    command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={} --input-video-name={} --reference-video-name={} \
        --output-width={} --output-height={} --save-metadata --threads={}'.format(vpxdec_path, dataset_dir, input_video_name, reference_video_name, output_width, output_height, get_num_threads(input_video_profile['height']))

    if skip is not None:
        command += ' --skip={}'.format(skip)
    if limit is not None:
        command += ' --limit={}'.format(limit)
    if postfix is not None:
        command += ' --postfix={}'.format(postfix)
    if save_quality:
        command += ' --save-quality'
    if vmaf:
        command += '--save-rgbframe'
    print(command)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    # load quality from a log file
    quality = []
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            quality.append(float(line.split('\t')[1]))
    print(f"Quality from the libvpx decoder: {np.average(np.array(quality))}")

    return quality

def offline_dnn_quality_vmaf(vpxdec_path, dataset_dir, input_video_name, reference_video_name,  \
                                model_name, output_width, output_height, skip=None, limit=None, postfix=None, save_quality=True, vmaf=False, psnr_block_format=False, psnr_block_width=None, psnr_block_height=None):
    #log file
    if postfix is not None:
        log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, postfix, 'quality.txt')
    else:
        log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, 'quality.txt')

    #run sr-integrated decoder
    input_video_path = os.path.join(dataset_dir, 'video', input_video_name)
    input_resolution = get_video_profile(input_video_path)['height']
    scale = output_height // input_resolution


    command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={} --input-video-name={} --reference-video-name={} \
        --dnn-scale={} --dnn-name={} --output-width={} --output-height={} --decode-mode=decode_sr --dnn-mode=offline_dnn --save-metadata \
            --threads={}'.format(vpxdec_path, dataset_dir, input_video_name, reference_video_name, scale, model_name, output_width, output_height, get_num_threads(input_resolution))
    if skip is not None:
        command += ' --skip={}'.format(skip)
    if limit is not None:
        command += ' --limit={}'.format(limit)
    if postfix is not None:
        command += ' --postfix={}'.format(postfix)
    if save_quality:
        command += ' --save-quality'
    if vmaf:
        command += ' --save-rgbframe'
    if psnr_block_format:
        command += f' --save-quality-block-format --psnr-nblocks-per-column={psnr_nblocks_per_column} --psnr-nblocks-per-row={psnr_nblocks_per_row} --psnr-block-width={psnr_block_width} --psnr-block-height={psnr_block_height}'
    print(command)
    subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    #load quality from a log file
    if save_quality:
        #First, load quality from a log file
        quality = []
        with open(log_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                quality.append(float(line.split('\t')[1]))
        return quality

def offline_cache_quality_mt(q0, q1, vpxdec_path, dataset_dir, input_video_name, reference_video_name, model_name, output_width, output_height, gop):
    input_video_path = os.path.join(dataset_dir, 'video', input_video_name)
    input_video_profile = get_video_profile(input_video_path)
    input_video_height = input_video_profile['height']
    input_video_width = input_video_profile['width']
    scale = output_height // input_video_height

    while True:
        item = q0.get()
        if item == 'end':
            return
        else:
            start_time = time.time()
            cache_profile_name = item[0]
            skip = item[1]
            limit = item[2]
            postfix = item[3]
            idx = item[4]
            nblocks_per_row = item[5]
            nblocks_per_column = item[6]
            patch_width = item[7]
            patch_height = item[8]
            decode_block_cache = item[9]

            #log file
            if postfix is not None:
                log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, postfix, os.path.basename(cache_profile_name), 'quality.txt')
            else:
                log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, os.path.basename(cache_profile_name), 'quality.txt')

            if decode_block_cache:
                decode_mode = "decode_block_cache"
            else:
                decode_mode = "decode_cache"

            #run sr-integrated decoder   
            command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={} --input-video-name={} --reference-video-name={} --decode-mode={} \
            --dnn-mode=offline_dnn --cache-mode=profile_cache --output-width={} --output-height={} --save-metadata --dnn-name={} --dnn-scale={} \
            --cache-profile-name={} --threads={} --npatches-per-row={} --npatches-per-column={} --patch-width={} --patch-height={} --gop={} --save-quality'.format(vpxdec_path, dataset_dir, input_video_name, reference_video_name, decode_mode, output_width, output_height, model_name, scale, cache_profile_name, get_num_threads(input_video_height), nblocks_per_row, nblocks_per_column, patch_width, patch_height, gop)

            if skip is not None:
                command += ' --skip={}'.format(skip)
            if limit is not None:
                command += ' --limit={}'.format(limit)
            if postfix is not None:
                command += ' --postfix={}'.format(postfix)
            
            print(command)
            subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

            quality = []
            with open(log_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    quality.append(float(line.split('\t')[-1]))

            q1.put((idx, quality))

#ref: https://developers.google.com/media/vp9/settings/vod
def get_num_threads(resolution):
    tile_size = 256
    if resolution >= tile_size:
        num_tiles = resolution // tile_size
        log_num_tiles = math.floor(math.log(num_tiles, 2))
        num_threads = (2**log_num_tiles) * 2
    else:
        num_threads = 2
    return num_threads

def offline_fast_graph_finegrained_metadata(patch_width, patch_height, num_blocks_per_row, num_blocks_per_column, vpxdec_path, dataset_dir, input_video_name, gop, skip=None, limit=None, postfix=None):

        cache_profile_dir = os.path.join(dataset_dir, 'profile', input_video_name, "no_model")
        if postfix is not None:
            cache_profile_dir = os.path.join(cache_profile_dir, postfix)
        os.makedirs(cache_profile_dir, exist_ok=True)

        anchor_point_set = AnchorPointSet.create([Frame(i, 0, 'normal_frame') for i in range(gop*2)], cache_profile_dir, 'no_anchor', num_blocks_per_row, num_blocks_per_column)
        anchor_point_set.save_cache_profile()
        cache_profile_name = anchor_point_set.get_cache_profile_name()

        #log file
        model_name = "no_model"
        if postfix is not None:
            log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, postfix, os.path.basename(cache_profile_name), 'metadata.txt')
        else:
            log_path = os.path.join(dataset_dir, 'log', input_video_name, model_name, os.path.basename(cache_profile_name), 'metadata.txt')

        #run sr-integrated decoder
        input_video_path = os.path.join(dataset_dir, 'video', input_video_name)
        input_video_profile = get_video_profile(input_video_path)
        input_video_width = input_video_profile['width']
        input_video_height = input_video_profile['height']

        libvpx.save_metadata(vpxdec_path, dataset_dir, input_video_name, skip=num_skipped_frames, limit=num_decoded_frames, postfix=postfix, save_frame_size=True)
        frames = load_frame_index(dataset_dir, input_video_name, postfix, log_path=log_path, load_frame_size=True)
        root_frames_names, non_root_frames_references, frame_dependencies, non_root_frames_references_all = get_frame_dependency(dataset_dir, input_video_name, frames, postfix, all=True, log_path=log_path)

        command = '{} --codec=vp9 --noblit --frame-buffers=50 --dataset-dir={} \
        --input-video-name={} --decode-mode=decode_block_cache --dnn-mode=offline_dnn --cache-mode=profile_cache \
        --output-width={} --output-height={} --save-metadata --save-super-finegrained-metadata --dnn-name={} --dnn-scale=1 --cache-profile-name={} --threads=1 --patch-width={} --patch-height={} --npatches-per-row={} --npatches-per-column={} --gop={}'.format(vpxdec_path, dataset_dir, input_video_name, input_video_width, input_video_height, "no_model", cache_profile_name, patch_width, patch_height, num_blocks_per_row, num_blocks_per_column, gop)
        if skip is not None:
            command += ' --skip={}'.format(skip)
        if limit is not None:
            command += ' --limit={}'.format(limit)
        if postfix is not None:
            command += ' --postfix={}'.format(postfix)
        print(command)
        subprocess.check_call(shlex.split(command),stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        super_finegrained_metadata_log = os.path.join(dataset_dir, 'log', input_video_name, 'no_model', '.' if postfix is None else postfix, 'no_anchor', 'super_finegrained_metadata.txt')
        frame_idx = 0

        total_intra_pixels = 0
        total_inter_pixels = 0
        intra_total_referenced = 0
        inter_total_referenced = 0
        keyframe_total_referenced = 0
        normal_frame_total_referenced = 0
        altref_total_referenced = 0

        type_per_pixel = {}
        total_keyframe_pixels = 0
        total_normal_frame_pixels = 0
        total_altref_pixels = 0
        for frame in frames:
            type_per_pixel[frame.name] = np.zeros((input_video_width, input_video_height)) # 0 -> the pixel belongs to an intra-coded macroblock; 1 -> the pixel belongs to an inter-coded macroblock
            if frame.frame_type == "key_frame":
                total_keyframe_pixels += input_video_width*input_video_height
            elif frame.frame_type == "normal_frame":
                total_normal_frame_pixels += input_video_width*input_video_height
            else:
                total_altref_pixels += input_video_width*input_video_height

        key_frames = []
        normal_frames = []
        altref_frames = []
        frames_referenced = {}

        
        block_per_pixel = {}
        for frame in frames:
            block_per_pixel[frame.name] = {}
            if frame.frame_type == "key_frame":
                key_frames.append(frame)
                frames_referenced[frame.name] = [0, frame]
            elif frame.frame_type == "normal_frame":
                normal_frames.append(frame)
                frames_referenced[frame.name] = [0, frame]
            else:
                altref_frames.append(frame)
                frames_referenced[frame.name] = [0, frame]

        
        intra_blocks = []
        inter_blocks = []
        blocks_referenced = {}

        with open(super_finegrained_metadata_log, "r") as f:
            intra_data = []
            inter_data = []
            bias = np.zeros((num_blocks_per_column, num_blocks_per_row))
            while True:


                line = f.readline().strip()
                data = line[7:].split("\t")
                if line.startswith("[INTRA]"):
                    intra_data.append(data)
                elif line.startswith("[INTER]"):
                    inter_data.append(data)
                elif line.startswith("[FRAME]"):
                    frame = frames[frame_idx]

                    frame_idx += 1
                    assert frame.video_index == int(data[0])
                    assert frame.super_index == int(data[1])

                    idx = 0
                    while idx < len(intra_data):
                        data = intra_data[idx]
                        plane = int(data[0])
                        plane_sf = (2 if plane != 0 else 1)
                        x_offsets = int(data[1])
                        y_offsets = int(data[2])
                        w = int(data[3])
                        h = int(data[4])

                        current_block = block(frame, x_offsets, y_offsets, w, h, "intra")
                        intra_blocks.append(current_block)
                        blocks_referenced[current_block] = 0
                        for ww in range(w):
                            for hh in range(h):
                                if x_offsets+ww<0 or x_offsets+ww>=input_video_width or y_offsets+hh<0 or y_offsets+hh>=input_video_height:
                                    continue
                                block_per_pixel[frame.name][f"{x_offsets+ww}-{y_offsets+hh}"] = current_block


                        total_intra_pixels += min(w, input_video_width-x_offsets)*min(h, input_video_height-y_offsets)
                        start_row_idx = y_offsets // (patch_height // plane_sf)
                        start_col_idx = x_offsets // (patch_width // plane_sf)
                        end_row_idx = min(num_blocks_per_column-1, (y_offsets+h-1) // (patch_height // plane_sf))
                        end_col_idx = min(num_blocks_per_row-1, (x_offsets+w-1) // (patch_width // plane_sf))
                        for row_idx in range(start_row_idx, end_row_idx+1):
                            for col_idx in range(start_col_idx, end_col_idx+1):
                                global_row_start_idx = max(row_idx*patch_height//plane_sf, y_offsets)
                                global_row_end_idx = min((row_idx+1)*patch_height//plane_sf, y_offsets+h)
                                global_col_start_idx = max(col_idx*patch_width//plane_sf, x_offsets)
                                global_col_end_idx = min((col_idx+1)*patch_width//plane_sf, x_offsets+w)
                                bias[row_idx][col_idx] += (global_row_end_idx-global_row_start_idx) * (global_col_end_idx-global_col_start_idx) / (w*h)
                        idx += 3
                    intra_data = []

                    idx = 0
                    while idx < len(inter_data):
                        if x_offsets+w>input_video_width or y_offsets+h>input_video_height:
                            print("warning: x_offsets+w>input_video_width or y_offsets+h>input_video_height")
                        type_per_pixel[frame.name][x_offsets:x_offsets+w, y_offsets:y_offsets+h] = 1
                        total_inter_pixels += min(w, input_video_width-x_offsets)*min(h, input_video_height-y_offsets)
                        data = inter_data[idx]
                        plane = int(data[0])
                        plane_sf = (2 if plane != 0 else 1)
                        x_offsets = int(data[1])
                        y_offsets = int(data[2])
                        w = int(data[3])
                        h = int(data[4])
                        
                        current_block = block(frame, x_offsets, y_offsets, w, h, "inter")
                        inter_blocks.append(current_block)
                        blocks_referenced[current_block] = 0
                        for ww in range(w):
                            for hh in range(h):
                                if x_offsets+ww<0 or x_offsets+ww>=input_video_width or y_offsets+hh<0 or y_offsets+hh>=input_video_height:
                                    continue
                                block_per_pixel[frame.name][f"{x_offsets+ww}-{y_offsets+hh}"] = current_block

                        start_row_idx = y_offsets // (patch_height // plane_sf)
                        start_col_idx = x_offsets // (patch_width // plane_sf)
                        end_row_idx = min(num_blocks_per_column-1, (y_offsets+h-1) // (patch_height // plane_sf))
                        end_col_idx = min(num_blocks_per_row-1, (x_offsets+w-1) // (patch_width // plane_sf))
                        for row_idx in range(start_row_idx, end_row_idx+1):
                            for col_idx in range(start_col_idx, end_col_idx+1):
                                global_row_start_idx = max(row_idx*patch_height//plane_sf, y_offsets)
                                global_row_end_idx = min((row_idx+1)*patch_height//plane_sf, y_offsets+h)
                                global_col_start_idx = max(col_idx*patch_width//plane_sf, x_offsets)
                                global_col_end_idx = min((col_idx+1)*patch_width//plane_sf, x_offsets+w)
                                bias[row_idx][col_idx] += (global_row_end_idx-global_row_start_idx) * (global_col_end_idx-global_col_start_idx) / (w*h)
                        is_compound = int(data[5])
                        if is_compound:
                            idx += 6
                            raise NotImplementedError
                        else:
                            reference_index = int(data[6])
                            if frame.name in non_root_frames_references_all.keys():
                                reference_frame = non_root_frames_references_all[frame.name][reference_index]
                                type_mat = type_per_pixel[reference_frame.name]
                                x_0 = int(data[7])
                                y_0 = int(data[8])
                                for ww in range(w):
                                    for hh in range(h):
                                        if x_0+ww<0 or x_0+ww>=input_video_width or y_0+hh<0 or y_0+hh>=input_video_height:
                                            continue
                                        if type_mat[x_0+ww,y_0+hh] == 0:
                                            intra_total_referenced += 1
                                        else:
                                            inter_total_referenced += 1
                                        
                                        reference_block = block_per_pixel[reference_frame.name][f"{x_0+ww}-{y_0+hh}"]
                                        blocks_referenced[reference_block] += 1

                                        if reference_frame.frame_type == "key_frame":
                                            frames_referenced[reference_frame.name][0] += 1
                                            keyframe_total_referenced += 1
                                        elif reference_frame.frame_type == "normal_frame":
                                            frames_referenced[reference_frame.name][0] += 1
                                            normal_frame_total_referenced += 1
                                        else:
                                            frames_referenced[reference_frame.name][0] += 1
                                            altref_total_referenced += 1
                            idx += 3
                    inter_data = []

                    if frame_idx == len(frames):
                        break

        def group_sorted_results(sorted_results):
            if len(sorted_results)==0:
                return [], []
            if isinstance(sorted_results[0][0], block):
                case = "block"
            elif isinstance(sorted_results[0][1][1], Frame):
                case = "Frame"
            else:
                raise NotImplementedError
            y_group1 = []
            y_group2 = []
            for idx, result in enumerate(sorted_results):
                if case == "block":
                    y = result[1]/(result[0].w*result[0].h)
                    if result[0].type == "intra":
                        y_group1.append(y)
                    else:
                        y_group2.append(y)
                elif case == "Frame":
                    y = result[1][0]/(input_video_width*input_video_height)
                    if result[1][1].frame_type == "normal_frame":
                        y_group2.append(y)
                    else:
                        y_group1.append(y)
            return y_group1, y_group2

        sorted_blocks = sorted(blocks_referenced.items(), key=lambda x:-x[1]/x[0].w/x[0].h)
        intra_rankings = []
        inter_rankings = []
        for idx, _block in enumerate(sorted_blocks):
            if _block[0].type == "intra":
                intra_rankings.append(idx+1)
            else:
                inter_rankings.append(idx+1)
        sorted_intra, sorted_inter = group_sorted_results(sorted_blocks)

        sorted_frames = sorted(frames_referenced.items(), key=lambda x:-x[1][0])
        key_and_altref_rankings = []
        normal_rankings = []
        for idx, _frame in enumerate(sorted_frames):
            if _frame[1][1].frame_type == "normal_frame":
                normal_rankings.append(idx+1)
            else:
                key_and_altref_rankings.append(idx+1)
        
        sorted_key_and_altref, sorted_normal = group_sorted_results(sorted_frames)
        
        return sorted_intra, sorted_inter, sorted_key_and_altref, sorted_normal