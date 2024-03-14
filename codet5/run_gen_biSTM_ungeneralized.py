# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import os
import random

import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing
import time
from important_positions import load_and_cache_gen_data_biSTM
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from models import build_or_load_gen_model, build_or_load_gen_model_small, build_or_load_gen_model_biSTM
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data
from configs import add_args, set_seed, set_dist
from transformers import AdamW, get_linear_schedule_with_warmup
from AST.measure_ast_complete_rate import get_common_segment_only_insert
import sys

sys.path.append("..")

from important_positions import get_valid_segments_in_BiModel, \
    get_valid_segments_in_BiModel_suffix_splicing, get_all_candidate_valid_segments_in_BiModel, \
    handle_BiSTM_type_id_to_input, get_im_in_BiModel_suffix_splicing

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import copy


def get_model():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    set_dist(args)
    set_seed(args)
    args.add_segment_ids = True
    config, model, tokenizer = build_or_load_gen_model_small(args)
    model.to(args.device)
    model.load_state_dict(torch.load(args.trained_model_path))
    return model, tokenizer


def get_First_diff(tgt, preds):  # Returns the current unequal length
    if len(preds) == 0:
        return 0
    else:
        i = 0
        while i < len(tgt) and i < len(preds):
            if tgt[i] != preds[i]: return i
            i += 1
        if i == len(preds): return i
        return -1


def minDistance(num1, num2):
    n1 = len(num1)
    n2 = len(num2)
    dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
    for j in range(1, n2 + 1):
        dp[0][j] = dp[0][j - 1] + 1
    for i in range(1, n1 + 1):
        dp[i][0] = dp[i - 1][0] + 1
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            if num1[i - 1] == num2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]) + 1
    return dp[-1][-1]


def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True)
    # Start evaluating model
    logger.info("  " + "***** Running ppl evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, batch_num = 0, 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval ppl"):
        batch = tuple(t.to(args.device) for t in batch)
        if len(batch) == 3:  # BiSTM model
            source_ids, target_ids, BiSTM_type_ids = batch
        else:
            source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            if args.model_type == 'roberta':
                loss, _, _ = model(
                    source_ids=source_ids, source_mask=source_mask,
                    target_ids=target_ids, target_mask=target_mask
                )
            else:
                if len(batch) == 3:
                    outputs = model(
                        input_ids=source_ids, attention_mask=source_mask,
                        labels=target_ids, decoder_attention_mask=target_mask,
                        BiSTM_type_ids=BiSTM_type_ids
                    )
                else:
                    outputs = model(
                        input_ids=source_ids, attention_mask=source_mask,
                        labels=target_ids, decoder_attention_mask=target_mask,
                    )
                loss = outputs[0]

        eval_loss += loss.item()
        batch_num += 1
    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl


def handle_segment_model_output(im, true_ids, tokenizer, mask_tokens):
    target_ids = true_ids[0][1:]
    cur_mask_index = 0
    empty_token_id = tokenizer.encode('<0_token>')[1]
    sum = 0
    for j in true_ids[1:]:
        cur_mask_token_id = mask_tokens[cur_mask_index]
        next_mask_token_id = mask_tokens[cur_mask_index + 1]
        if cur_mask_token_id not in im or next_mask_token_id not in im:
            return None, 0  # fail
        segment = im[im.index(cur_mask_token_id) + 1:im.index(next_mask_token_id)]
        sum += len(segment)
        tmp_segment = []
        for i in segment:
            if i != empty_token_id: tmp_segment.append(i)
        target_ids += tmp_segment
        target_ids += j
        cur_mask_index += 1
    return target_ids, sum


def handle_segment_model_output_Attention_Cache(im, true_ids, tokenizer, mask_tokens):
    # Recovery algorithm based on Attention Cache
    target_ids = true_ids[0][1:]
    cur_mask_index = 0
    empty_token_id = tokenizer.encode('<0_token>')[1]
    sum = 0
    for j in true_ids[1:]:
        cur_mask_token_id = mask_tokens[cur_mask_index]
        next_mask_token_id = mask_tokens[cur_mask_index + 1]
        if cur_mask_token_id not in im or next_mask_token_id not in im:
            if cur_mask_token_id not in im:  # Note The first segment delimiter is incorrect
                return None, mask_tokens, 0
            else:
                s = im.index(cur_mask_token_id) + 1
                for s1 in range(s, len(im)):
                    if im[s1] == 2 or tokenizer.decode([im[s1]]).startswith('<extra_id_'):
                        return None, mask_tokens[
                                     cur_mask_index + 1:], s1  # None,The list of remaining segment separators, the longest of which can be treated as the last position correctly generated
                return None, mask_tokens[cur_mask_index + 1:], len(im) - 1
        segment = im[im.index(cur_mask_token_id) + 1:im.index(next_mask_token_id)]
        sum += len(segment)
        tmp_segment = []
        for c, i in enumerate(segment):
            if i != empty_token_id:
                tmp_segment.append(i)
        target_ids += tmp_segment
        target_ids += j
        cur_mask_index += 1
    return target_ids, sum, None


def handle_segment_model_input(origin_source_ids, true_ids, tokenizer, arg):
    source_ids = copy.deepcopy(origin_source_ids)
    source_ids = source_ids[0].tolist()
    if 0 in source_ids:
        source_ids = source_ids[:source_ids.index(0)]
    source_ids += [tokenizer.encode('<seq>')[1]]
    target_ids = true_ids[0][1:]
    cur_mask_index = 0
    mask_tokens = []
    for j in true_ids[1:]:
        cur_mask_token = '<extra_id_{}>'.format(cur_mask_index)
        cur_mask_token_id = tokenizer.encode(cur_mask_token)[1]
        mask_tokens.append(cur_mask_token_id)
        target_ids += [cur_mask_token_id]
        target_ids += j
        cur_mask_index += 1
    source_ids += target_ids
    source_ids.extend([0] * (600 - len(source_ids)))
    return torch.tensor([source_ids], dtype=torch.long,
                        device='cuda:0'), mask_tokens + [2]


def handle_biSTM_model_output_Attention_Cache_forward(im, first_segment, true_ids, tokenizer, mask_tokens,
                                                      is_backward=False):
    # first_segment: The first segment is a modified verification segment for forward decoding and a new suffix for reverse decoding
    target_ids = copy.deepcopy(first_segment)
    cur_mask_index = 0
    empty_token_id = tokenizer.encode('<0_token>')[1]
    sum = 0
    for j in true_ids:
        cur_mask_token_id = mask_tokens[cur_mask_index]
        next_mask_token_id = mask_tokens[cur_mask_index + 1]
        if cur_mask_token_id not in im or next_mask_token_id not in im:
            if cur_mask_token_id not in im:  # That means the first segment separator is wrong
                return None, mask_tokens, 0
            else:
                s = im.index(cur_mask_token_id) + 1
                for s1 in range(s, len(im)):
                    if im[s1] == 2 or tokenizer.decode([im[s1]]).startswith('<extra_id_'):
                        return None, mask_tokens[cur_mask_index + 1:], s1  # None,剩余的段分隔符列表，最长的可视为正确生成的最后一个位置
                return None, mask_tokens[cur_mask_index + 1:], len(im) - 1
        segment = im[im.index(cur_mask_token_id) + 1:im.index(next_mask_token_id)]
        sum += len(segment)
        tmp_segment = []
        for c, i in enumerate(segment):
            if i != empty_token_id:
                tmp_segment.append(i)
        if is_backward:
            target_ids = tmp_segment + target_ids
            target_ids = j + target_ids
        else:
            target_ids += tmp_segment
            target_ids += j
        cur_mask_index += 1
    return target_ids, sum, None


def get_forward_inputs_and_type_ids(result, source_id, STM_tokenizer):
    forward_num = []
    forward_input = []
    forward_input_lable = []
    forward_true_ids = []
    segment_special_tokens = []
    # In each alternative, [[Prefix before validation segment][Modified validation segment][The suffix after the validation segment]]
    if 0 in source_id:
        source_id = source_id[:source_id.index(0)]
    forward_input += source_id
    forward_num += ['<extra_id_32103>'] * len(source_id)
    forward_input += STM_tokenizer.encode("<seq>")[1:-1]
    forward_num += ['<extra_id_32107>']
    for j in range(len(result[
                           0])):  # Get the prefix [[Validation segment 1] [Non-validation segment 1] [Validation segment 2] [Non-validation segment 2]
        forward_input += result[0][j][0]
        if len(result[0][j][0]) != 0:
            forward_num += ['<extra_id_32104>'] * len(result[0][j][0])
        forward_input += result[0][j][2]
        if len(result[0][j][2]) != 0:
            forward_num += ['<extra_id_32105>'] * len(result[0][j][2])

    forward_input += result[1]  # Plus the sequence corresponding to the [modified validation] segment
    forward_num += ['<extra_id_32106>'] * len(result[1])

    for k in range(len(result[2])):  # After the validation segment
        forward_input += STM_tokenizer.encode("<extra_id_" + str(k) + ">")[1:-1]
        segment_special_tokens += STM_tokenizer.encode("<extra_id_" + str(k) + ">")[1:-1]
        forward_num += ['<extra_id_32107>']
        forward_input += result[2][k][0]
        forward_true_ids.append(result[2][k][0])
        if len(result[2][k][0]) != 0:
            forward_num += ['<extra_id_32104>'] * len(result[2][k][0])

        forward_input_lable += STM_tokenizer.encode("<extra_id_" + str(k) + ">")[1:-1]
        if len(result[2][k][1]) == 0:
            forward_input_lable += STM_tokenizer.encode("<0_token>")[1:-1]
        else:
            forward_input_lable += result[2][k][1]

    forward_input += [0] * (600 - len(forward_input))
    forward_input_lable += [2]
    forward_input_lable += [0] * (80 - len(forward_input_lable))
    forward_num += ['<pad>'] * (600 - len(forward_num))
    segment_special_tokens += [2]
    return forward_input, forward_true_ids, \
           forward_num, segment_special_tokens


# get_backward_inputs_and_type_ids:
# Concatenates the reverse input, backward_true_ids, identification array and List of segment interval characters
# according to valid_segments and the correct suffix.
#
def get_backward_inputs_and_type_ids(result, source_id, STM_tokenizer, new_suffix):
    if 0 in source_id:
        source_id = source_id[:source_id.index(0)]
    reverse_num = []
    reverse_input = []
    reverse_input_lable = []
    backward_true_ids = []
    segment_special_tokens = []
    # reverse_input += STM_tokenizer.encode("<reversed>")[1:-1]
    reverse_input += STM_tokenizer.encode("<translate_prefix>")[1:-1]
    reverse_num += ['<extra_id_32107>']
    reverse_input += source_id
    reverse_num += ['<extra_id_32103>'] * len(source_id)
    reverse_input += STM_tokenizer.encode("<seq>")[1:-1]
    reverse_num += ['<extra_id_32107>']

    reverse_input += new_suffix
    reverse_num += ['<extra_id_32104>'] * len(new_suffix)

    m = 0
    for j in range(len(result[0]) - 1, -1,
                   -1):  # [<extra_id_99>]+[[verification segment n]]+[<extra_id_98>]+....+[verification segment 1]+[Complement 0 until up to maximum length]
        reverse_input += STM_tokenizer.encode("<extra_id_" + str(m) + ">")[1:-1]
        segment_special_tokens += STM_tokenizer.encode("<extra_id_" + str(m) + ">")[1:-1]
        reverse_input_lable += STM_tokenizer.encode("<extra_id_" + str(m) + ">")[1:-1]
        m += 1
        reverse_num += ['<extra_id_32107>']
        reverse_input += result[0][j][0]  # [verification segment n]
        backward_true_ids.append(result[0][j][0])
        if len(result[0][j][0]) != 0:
            reverse_num += ['<extra_id_32104>'] * len(result[0][j][0])

        if len(result[0][j][1]) == 0:
            reverse_input_lable += STM_tokenizer.encode("<0_token>")[1:-1]
        else:
            reverse_input_lable += result[0][j][1]

    reverse_input += [0] * (600 - len(reverse_input))
    reverse_input_lable += [2]
    reverse_input_lable += [0] * (80 - len(reverse_input_lable))
    reverse_num += ['<pad>'] * (600 - len(reverse_num))
    segment_special_tokens += [2]

    return reverse_input, backward_true_ids, reverse_num, segment_special_tokens


def get_BISTM_model():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    set_dist(args)
    set_seed(args)
    args.add_segment_ids = True
    if args.sub_task == 'python-java':
        args.trained_model_path = \
            r"./g4g/python2java/bistm/forward_bao/pytorch_model.bin"
    else:
        args.trained_model_path = \
            r"./g4g/java2python/bistm/forward_bao/pytorch_model.bin"
    config, forward_model, bistm_tokenizer = build_or_load_gen_model_biSTM(args)
    forward_model.to(args.device)
    forward_model.load_state_dict(torch.load(args.trained_model_path))
    if args.sub_task == 'python-java':
        args.trained_model_path = \
            r"./g4g/python2java/bistm/backward/pytorch_model.bin"
    else:
        args.trained_model_path = \
            r"./g4g/java2python/bistm/backward/pytorch_model.bin"
    logger.info(args)
    set_dist(args)
    set_seed(args)
    args.add_segment_ids = True
    _, backward_model, _ = build_or_load_gen_model_biSTM(args)
    backward_model.to(args.device)
    backward_model.load_state_dict(torch.load(args.trained_model_path))
    return forward_model, backward_model, bistm_tokenizer


def get_origin_false_and_true(valid_segment):
    origin_false = []
    origin_true = []
    for i in valid_segment[0]:
        origin_false += (i[0] + i[2])
        origin_true += (i[0] + i[1])
    origin_false += valid_segment[1]
    origin_true += valid_segment[1]
    for i in valid_segment[2]:
        origin_false += (i[2] + i[0])
        origin_true += (i[1] + i[0])
    return origin_false, origin_true


def handle_segment(prefix_valid_segment, valid_segment,
                   suffix_valid_segment, handled_segments, v, pre_answer, pre_false, is_prefix=True):
    # Divide the original validation segment into prefix_valid_segment,valid_segment,suffix_valid_segment
    # There may be empty strings. handled_segments represents out after processing, and pre_answer and pre_false are intermediate variables
    if is_prefix:
        handled_segments[-1][1] += prefix_valid_segment
        handled_segments[-1][2] += prefix_valid_segment
        if not valid_segment:
            handled_segments[-1][1] += (suffix_valid_segment + v[1])
            handled_segments[-1][2] += (suffix_valid_segment + v[2])
        else:
            handled_segments.append([valid_segment, suffix_valid_segment + v[1], \
                                     suffix_valid_segment + v[2]])
    else:
        # print('is_not_suffix')
        pre_answer += v[1] + prefix_valid_segment
        pre_false += v[2] + prefix_valid_segment
        if not valid_segment:
            pre_answer += suffix_valid_segment
            pre_false += suffix_valid_segment
        else:
            handled_segments.append([valid_segment, pre_answer, pre_false])
            pre_false = [i for i in suffix_valid_segment]
            pre_answer = [i for i in suffix_valid_segment]
    return handled_segments, pre_false, pre_answer


from codet5.AST.measure_ast_complete_rate import kmp


def handle_prefix_or_suffix_in_outs_small_range(valid_segment, ast_data, tokenizer, is_prefix=True):
    origin_a, origin_b = get_origin_false_and_true(valid_segment)
    origin_a = tokenizer.decode(origin_a)
    origin_b = tokenizer.decode(origin_b)
    # This is the case where the upper and lower bounds of the verification segment are reduced
    timinater = [274, 288, 289, 31, 95, 97, 1, 2, 5997]
    prefix_valid_segment = [[[], [], []]]
    pre_false = []
    pre_answer = []
    for v in (valid_segment[0] if is_prefix else valid_segment[2]):
        start_pos = 0
        end_pos = len(v[0])
        start_at_prop = 0.1

        if not (1 in v[0] or 2 in v[0] or start_at_prop < 0.25):
            if start_at_prop < 0.5:
                # Random length
                random_len = int(len(v[0]) * random.random())
                start_pos = len(v[0]) // 2 - random_len // 2
                end_pos = len(v[0]) // 2 + random_len // 2
            elif start_at_prop < 0.75:
                # Begins with a statement terminator
                for c, i in enumerate(v[0]):
                    if i in timinater:
                        start_pos = c + 1
                        break
            else:
                # res = get_AST_set(jparser, v[0], tokenizer)
                res = None
                for sub_ast in res:
                    start_pos, end_pos = kmp(v[0], sub_ast)
                    if start_pos != -1:
                        break
                if start_pos == -1:
                    start_pos = 0
                    end_pos = len(v[0])

        # AST Maximum subtree
        prefix_valid_segment1, pre_false1, pre_answer1 = handle_segment(v[0][:start_pos], v[0][start_pos:end_pos],
                                                                        v[0][end_pos:], prefix_valid_segment, v,
                                                                        pre_answer,
                                                                        pre_false, is_prefix)
        prefix_valid_segment, pre_false, pre_answer = prefix_valid_segment1, pre_false1, pre_answer1
    return prefix_valid_segment[1:]


def handle_prefix_or_suffix_in_valid_history(valid_segment, ast_data, tokenizer, valid_history, is_prefix=True):
    origin_a, origin_b = get_origin_false_and_true(valid_segment)
    origin_a = tokenizer.decode(origin_a)
    origin_b = tokenizer.decode(origin_b)
    prefix_valid_segment = [[[], [], []]]
    pre_false = []
    pre_answer = []
    for v in (valid_segment[0] if is_prefix else valid_segment[2]):
        if 1 in v[0] or 2 in v[0]:
            start_pos = 0
            end_pos = len(v[0])
        else:
            flag = True
            for history in valid_history:
                start_pos, end_pos = kmp(v[0], history)
                if start_pos != -1:
                    print('matched!!', start_pos, end_pos)
                    flag = False
                    break
            if flag:
                start_pos, end_pos = len(v[0]), len(v[0])
        # AST Maximum subtree
        prefix_valid_segment1, pre_false1, pre_answer1 = handle_segment(v[0][:start_pos], v[0][start_pos:end_pos],
                                                                        v[0][end_pos:], prefix_valid_segment, v,
                                                                        pre_answer,
                                                                        pre_false, is_prefix)
        prefix_valid_segment, pre_false, pre_answer = prefix_valid_segment1, pre_false1, pre_answer1
    return prefix_valid_segment[1:]


def handle_prefix_or_suffix_in_outs_discard(valid_segment, is_prefix=True):
    # Simulation of a missing validation segment
    prefix_valid_segment = [[[], [], []]]
    pre_false = []
    pre_answer = []
    for v in valid_segment:
        start_pos = 0
        end_pos = len(v[0])

        if not (1 in v[0] or 2 in v[0] or np.random.choice([True, False], p=[1, 0])):
            start_pos = end_pos
        prefix_valid_segment, pre_false, pre_answer = handle_segment(v[0][:start_pos], v[0][start_pos:end_pos],
                                                                     v[0][end_pos:], prefix_valid_segment, v,
                                                                     pre_answer,
                                                                     pre_false, is_prefix)
    return prefix_valid_segment[1:]


def get_human_segments_2_0(valid_segment, tokenizer):
    origin_false, origin_true = get_origin_false_and_true(valid_segment)
    ast_data = None
    # This is the case where the upper and lower bounds of the verification segment are reduced
    prefix_valid_segment = handle_prefix_or_suffix_in_outs_small_range(valid_segment, ast_data, tokenizer)
    suffix_valid_segment = handle_prefix_or_suffix_in_outs_small_range(valid_segment, ast_data, tokenizer,
                                                                       is_prefix=False)
    valid_segment1 = [prefix_valid_segment, valid_segment[1], suffix_valid_segment]
    origin_false1, origin_true1 = get_origin_false_and_true(valid_segment1)
    #  Simulation of a missing validation segment
    prefix_valid_segment = handle_prefix_or_suffix_in_outs_discard(valid_segment1[0])
    suffix_valid_segment = handle_prefix_or_suffix_in_outs_discard(valid_segment1[2], is_prefix=False)
    valid_segment2 = [prefix_valid_segment, valid_segment1[1], suffix_valid_segment]
    origin_false1, origin_true1 = get_origin_false_and_true(valid_segment2)
    print(origin_false == origin_false1, origin_true1 == origin_true)
    return valid_segment2


def get_human_segments_2_0_valid_history(valid_segment, tokenizer, valid_history):
    origin_false, origin_true = get_origin_false_and_true(valid_segment)
    ast_data = None
    # This is the case where the upper and lower bounds of the verification segment are reduced
    prefix_valid_segment = handle_prefix_or_suffix_in_valid_history(valid_segment, ast_data, tokenizer, valid_history)
    suffix_valid_segment = handle_prefix_or_suffix_in_valid_history(valid_segment, ast_data, tokenizer, valid_history,
                                                                    is_prefix=False)
    valid_segment1 = [prefix_valid_segment, valid_segment[1], suffix_valid_segment]
    origin_false1, origin_true1 = get_origin_false_and_true(valid_segment1)
    print(origin_false == origin_false1, origin_true1 == origin_true)
    return valid_segment1


def get_human_segments(valid_segment):
    origin_false, origin_true = get_origin_false_and_true(valid_segment)
    timinater = [274, 288, 289, 31, 95, 97, 1, 2, 5997]
    prefix_valid_segment = []
    for v in valid_segment[0]:
        # start_at_prop = random.random()
        start_at_prop = 0.1
        if 1 in v[0] or start_at_prop < 0.3:
            prefix_valid_segment.append(v)
        else:
            flag = False
            for c, i in enumerate(v[0]):
                if start_at_prop >= 0.2 and start_at_prop < 0.4:
                    c = int(len(v[0]) * random.random())
                    prefix_valid_segment[-1][1] += v[0][:c + 1]
                    prefix_valid_segment[-1][2] += v[0][:c + 1]
                    if not v[0][c + 1:]:
                        prefix_valid_segment[-1][1] += v[1]
                        prefix_valid_segment[-1][2] += v[2]
                    else:
                        prefix_valid_segment.append([v[0][c + 1:], v[1], v[2]])
                    flag = True
                    break
                if i in timinater:
                    prefix_valid_segment[-1][1] += v[0][:c + 1]
                    prefix_valid_segment[-1][2] += v[0][:c + 1]
                    if not v[0][c + 1:]:
                        prefix_valid_segment[-1][1] += v[1]
                        prefix_valid_segment[-1][2] += v[2]
                    else:
                        prefix_valid_segment.append([v[0][c + 1:], v[1], v[2]])
                    flag = True
                    break
            if not flag:
                prefix_valid_segment[-1][1] += (v[0] + v[1])
                prefix_valid_segment[-1][2] += (v[0] + v[2])
    suffix_valid_segment = []
    pre_answer = []
    pre_false = []
    for v in valid_segment[2]:
        # start_at_prop = random.random()
        start_at_prop = 0.1
        if 2 in v[0] or start_at_prop < 0.2:
            suffix_valid_segment.append([v[0], pre_answer + v[1], pre_false + v[2]])
            pre_answer = []
            pre_false = []
        else:
            flag = False
            for c, i in enumerate(v[0]):
                if start_at_prop < 0.4:
                    c = int(random.random() * len(v[0]))
                    if not v[0][c + 1:]:
                        pre_answer += v[1] + v[0]
                        pre_false += v[2] + v[0]
                    else:
                        suffix_valid_segment.append(
                            [v[0][c + 1:], pre_answer + v[1] + v[0][:c + 1], pre_false + v[2] + v[0][:c + 1]])
                        pre_answer = []
                        pre_false = []
                    flag = True
                    break
                if i in timinater:
                    if not v[0][c + 1:]:
                        pre_answer += v[1] + v[0]
                        pre_false += v[2] + v[0]
                    else:
                        suffix_valid_segment.append(
                            [v[0][c + 1:], pre_answer + v[1] + v[0][:c + 1], pre_false + v[2] + v[0][:c + 1]])
                        pre_answer = []
                        pre_false = []
                    flag = True
                    break
            if not flag:
                pre_answer += v[1] + v[0]
                pre_false += v[2] + v[0]
    valid_segment = [prefix_valid_segment, valid_segment[1], suffix_valid_segment]
    origin_false1, origin_true1 = get_origin_false_and_true(valid_segment)

    prefix_valid_segment = []
    for v in valid_segment[0]:
        if 1 in v[0] or np.random.choice([True, False], p=[0, 1]):
            # if 1 in v[0] or True:
            prefix_valid_segment.append(v)
        else:
            prefix_valid_segment[-1][1] += (v[0] + v[1])
            prefix_valid_segment[-1][2] += (v[0] + v[2])
    suffix_valid_segment = []
    pre_answer = []
    pre_false = []
    for v in valid_segment[2]:
        if 2 in v[0] or np.random.choice([True, False], p=[0, 1]):
            # if 2 in v[0] or True:
            suffix_valid_segment.append(
                [v[0], pre_answer + v[1], pre_false + v[2]])
            pre_answer = []
            pre_false = []
        else:
            pre_answer += v[1] + v[0]
            pre_false += v[2] + v[0]
    valid_segment = [prefix_valid_segment, valid_segment[1], suffix_valid_segment]
    origin_false1, origin_true1 = get_origin_false_and_true(valid_segment)
    print(origin_false == origin_false1, origin_true1 == origin_true)
    return valid_segment


def dfs_decode(valid_segment, tokenizer):
    res = []
    for i in valid_segment[0]:
        cur = []
        for j in i:
            cur.append(tokenizer.decode(j))
        res.append(cur)
    res.append(tokenizer.decode(valid_segment[1]))
    for i in valid_segment[2]:
        cur = []
        for j in i:
            cur.append(tokenizer.decode(j))
        res.append(cur)
    return res


def get_all_compute_index(dir_path):
    prefix = dir_path
    f0_compute = os.path.join(prefix, 'intermediate_results_compute_0.csv')
    f1_compute = os.path.join(prefix, 'intermediate_results_compute_1.csv')
    f2_compute = os.path.join(prefix, 'intermediate_results_compute_2.csv')
    f3_compute = os.path.join(prefix, 'intermediate_results_compute_3.csv')
    f4_compute = os.path.join(prefix, 'intermediate_results_compute_4.csv')
    f5_compute = os.path.join(prefix, 'intermediate_results_compute_5.csv')

    set_dic = {}
    import csv

    for i, f_compute in enumerate([f0_compute, f1_compute,
                                   f2_compute, f3_compute, f4_compute, f5_compute]):
        set_dic[i] = set()
        if not os.path.exists(f_compute):
            temp = csv.reader(open(f_compute, 'w', encoding='utf-8'))
        f = csv.reader(open(f_compute, encoding='utf-8'))
        for j in f:
            if len(j) < 1: continue
            set_dic[i].add(j[1])
    return set_dic


def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, segment_model, forward_BiSTM,
                    backward_BiSTM, BiSTM_tokenizer):
    logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, pin_memory=False
        )
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.eval()
    import csv
    dir_path = args.output_dir
    set_dic = get_all_compute_index(dir_path)
    # ex-config
    top_num = 1

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    # pc-based hmCodeTrans
    intermediate_result4 = csv.writer(
        open(os.path.join(dir_path, 'intermediate_results_0.csv'), 'a', encoding='utf-8', newline=''))
    intermediate_compute_result4 = csv.writer(
        open(os.path.join(dir_path, 'intermediate_results_compute_0.csv'), 'a', encoding='utf-8', newline=''))

    # pv-based hmCodeTrans
    intermediate_result5 = csv.writer(
        open(os.path.join(dir_path, 'intermediate_results_1.csv'), 'a', encoding='utf-8', newline=''))
    intermediate_compute_result5 = csv.writer(
        open(os.path.join(dir_path, 'intermediate_results_compute_1.csv'), 'a', encoding='utf-8', newline=''))

    # kv-based hmCodeTrans
    intermediate_result6 = csv.writer(
        open(os.path.join(dir_path, 'intermediate_results_2.csv'), 'a', encoding='utf-8', newline=''))
    intermediate_compute_result6 = csv.writer(
        open(os.path.join(dir_path, 'intermediate_results_compute_2.csv'), 'a', encoding='utf-8', newline=''))

    # seg-NLTrans
    intermediate_result7 = csv.writer(
        open(os.path.join(dir_path, 'intermediate_results_3.csv'), 'a', encoding='utf-8', newline=''))
    intermediate_compute_result7 = csv.writer(
        open(os.path.join(dir_path, 'intermediate_results_compute_3.csv'), 'a', encoding='utf-8', newline=''))

    # seg-NLTrans-greedy
    intermediate_result8 = csv.writer(
        open(os.path.join(dir_path, 'intermediate_results_4.csv'), 'a', encoding='utf-8', newline=''))
    intermediate_compute_result8 = csv.writer(
        open(os.path.join(dir_path, 'intermediate_results_compute_4.csv'), 'a', encoding='utf-8', newline=''))

    f_res = csv.reader(open(
        r'./old_data/prefix/ml_data/early_stop_525/intermediate_results_1.csv', encoding='utf-8'))
    dic_prefix = {}
    cur_dic = {}
    cur_id = 0
    for i in f_res:
        if int(i[0]) != cur_id and int(i[0]) != 0:
            dic_prefix[cur_id] = cur_dic
            cur_id = int(i[0])
            cur_dic = {}
        cur_dic[i[1]] = tokenizer.encode(i[2][3:-4])
    prefix_f = r'./old_data/prefix/ml_data/early_stop_g4g_p2j_total/intermediate_results_compute_1.csv'
    prefix_f = csv.reader(open(prefix_f, encoding='utf-8'))
    prefix_dic = {}
    for i in prefix_f:
        prefix_dic[(i[1])] = int(i[4])

    stm_dic = {}

    for count, batch in enumerate(
            tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag))):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        tgt = tokenizer.encode(eval_examples[count].target)
        source = tokenizer.encode(eval_examples[count].source)
        origin_source_ids = source_ids
        source_ids1 = batch[0].to(args.device)
        source_mask1 = source_ids.ne(tokenizer.pad_token_id)
        tgt1 = tokenizer.encode(eval_examples[count].target)
        source1 = tokenizer.encode(eval_examples[count].source)
        #
        if len(source) >= 509 or len(tgt) >= 509:
            continue

        try:
            with torch.no_grad():
                print("KV-based hmCodeTrans")
                for max_same_length in [2]:  # -1 indicates that using the Attention cache-based recovery algorithm
                    for p in [0]:  # 'segment_queue' in key means using a non-AC algorithm,
                        # p<0 based on edit distance mode,p>0 based on en-de attention score mode,p=0, pure same concept
                        if eval_examples[count].source in set_dic[2]:
                            continue
                        source_ids = copy.deepcopy(source_ids1)
                        tgt = copy.deepcopy(tgt1)
                        num_return_sequences = args.beam_size
                        cur_num = 0
                        preds = []
                        isFirst = True
                        first_instance = 0
                        tgt_len = get_First_diff(tgt, preds)
                        while tgt_len != -1:
                            start = time.time()
                            prefix = tokenizer.decode(tgt[:tgt_len + 1])
                            if isFirst:
                                preds = model.generate(source_ids,
                                                       attention_mask=source_mask,
                                                       use_cache=True,
                                                       num_beams=args.beam_size,
                                                       early_stopping=args.task == 'summarize',
                                                       max_length=args.max_target_length,
                                                       num_return_sequences=num_return_sequences,
                                                       prefix=[0] + tgt,
                                                       prefix_len=tgt_len + 1,
                                                       tokenizer=tokenizer,
                                                       least_past=None,
                                                       is_past=False,
                                                       is_concent_sufix=True,
                                                       greedy_mode=True
                                                       )
                                im = preds[0].tolist()[0][1:]
                                view_len = len(im)
                                success = -1
                                regenerate_ratio = 0
                            else:
                                all_valid_segments = get_all_candidate_valid_segments_in_BiModel(im, tgt, tokenizer,
                                                                                                 args)
                                candidate_im = []
                                for valid_segments in all_valid_segments:
                                    start = time.time()
                                    success = 1
                                    forward_input, forward_true_ids, BiSTM_type_ids, segment_special_tokens = get_forward_inputs_and_type_ids(
                                        valid_segments, source_ids[0].tolist(), BiSTM_tokenizer)
                                    forward_input_after_handle = handle_BiSTM_type_id_to_input(forward_input,
                                                                                               BiSTM_type_ids,
                                                                                               BiSTM_tokenizer,
                                                                                               False)
                                    forward_input_after_handle = torch.tensor([forward_input_after_handle],
                                                                              device='cuda:0')
                                    preds = forward_BiSTM.generate(forward_input_after_handle,
                                                                   attention_mask=forward_input_after_handle.ne(
                                                                       tokenizer.pad_token_id),
                                                                   use_cache=True,
                                                                   num_beams=args.beam_size,
                                                                   early_stopping=args.task == 'summarize',
                                                                   max_length=args.max_target_length,
                                                                   num_return_sequences=num_return_sequences,
                                                                   least_past=None,
                                                                   segment=True,
                                                                   BiSTM=True,
                                                                   BiSTM_type_ids=BiSTM_type_ids
                                                                   )
                                    im = list(preds[0][0][1:].cpu().numpy())
                                    im, view_len, tgt_len = handle_biSTM_model_output_Attention_Cache_forward(im,
                                                                                                              valid_segments[
                                                                                                                  1],
                                                                                                              forward_true_ids,
                                                                                                              BiSTM_tokenizer,
                                                                                                              segment_special_tokens,
                                                                                                              )
                                    if im is None:  # Forward generation failed, and the system resumed.
                                        print('forward is false!!!')
                                        success = 0
                                        past = preds[1]
                                        if past[1] is not None:
                                            res_tuple = []
                                            for i in past[1]:
                                                cur_tuple = []
                                                for index, j in enumerate(i):
                                                    if index in [0, 1]:
                                                        cur_tuple.append(j[:, :, :tgt_len + 1, :])
                                                    else:
                                                        cur_tuple.append(j)
                                                res_tuple.append(tuple(cur_tuple))
                                            past = tuple([past[0], tuple(res_tuple)])
                                        preds = forward_BiSTM.generate(forward_input_after_handle,
                                                                       attention_mask=forward_input_after_handle.ne(
                                                                           tokenizer.pad_token_id),
                                                                       use_cache=True,
                                                                       num_beams=args.beam_size,
                                                                       early_stopping=args.task == 'summarize',
                                                                       max_length=args.max_target_length,
                                                                       num_return_sequences=num_return_sequences,
                                                                       least_past=past,
                                                                       tokenizer=BiSTM_tokenizer,
                                                                       segment=True,
                                                                       prefix=list(preds[0][0].cpu().numpy())[
                                                                              :tgt_len + 1] + view_len[0:1],
                                                                       mask_tokens=view_len[1:])
                                        im = list(preds[0][0][1:].cpu().numpy())
                                        # print('false: ', BiSTM_tokenizer.decode(im))
                                        regenerate_ratio = (len(im) - tgt_len) / len(im)
                                        im, view_len, tgt_len = handle_biSTM_model_output_Attention_Cache_forward(im,
                                                                                                                  valid_segments[
                                                                                                                      1],
                                                                                                                  forward_true_ids,
                                                                                                                  BiSTM_tokenizer,
                                                                                                                  segment_special_tokens)
                                        if im is None:
                                            print('forward also is false!!!')
                                            break

                                    if im[0] != 1:
                                        new_suffix = im
                                        backward_input, backward_true_ids, BiSTM_type_ids, segment_special_tokens = get_backward_inputs_and_type_ids(
                                            valid_segments, source_ids[0].tolist(), BiSTM_tokenizer, new_suffix)
                                        backward_input = torch.tensor([backward_input], device='cuda:0')
                                        preds = backward_BiSTM.generate(backward_input,
                                                                        attention_mask=backward_input.ne(
                                                                            tokenizer.pad_token_id),
                                                                        use_cache=True,
                                                                        num_beams=args.beam_size,
                                                                        early_stopping=args.task == 'summarize',
                                                                        max_length=args.max_target_length,
                                                                        num_return_sequences=num_return_sequences,
                                                                        least_past=None,
                                                                        segment=True,
                                                                        BiSTM=True,
                                                                        BiSTM_type_ids=BiSTM_type_ids
                                                                        )
                                        im = list(preds[0][0][1:].cpu().numpy())
                                        im, view_len, tgt_len = handle_biSTM_model_output_Attention_Cache_forward(im,
                                                                                                                  new_suffix,
                                                                                                                  backward_true_ids,
                                                                                                                  BiSTM_tokenizer,
                                                                                                                  segment_special_tokens,
                                                                                                                  True)
                                        if im is None:  # Backward generation failed, and the system resumed.
                                            print('backward is false!!!')
                                            success = 0
                                            past = preds[1]
                                            if past[1] is not None:
                                                res_tuple = []
                                                for i in past[1]:
                                                    cur_tuple = []
                                                    for index, j in enumerate(i):
                                                        if index in [0, 1]:
                                                            cur_tuple.append(j[:, :, :tgt_len + 1, :])
                                                        else:
                                                            cur_tuple.append(j)
                                                    res_tuple.append(tuple(cur_tuple))
                                                past = tuple([past[0], tuple(res_tuple)])
                                            preds = backward_BiSTM.generate(backward_input,
                                                                            attention_mask=backward_input.ne(
                                                                                tokenizer.pad_token_id),
                                                                            use_cache=True,
                                                                            num_beams=args.beam_size,
                                                                            early_stopping=args.task == 'summarize',
                                                                            max_length=args.max_target_length,
                                                                            num_return_sequences=num_return_sequences,
                                                                            least_past=past,
                                                                            tokenizer=BiSTM_tokenizer,
                                                                            segment=True,
                                                                            prefix=list(preds[0][0].cpu().numpy())[
                                                                                   :tgt_len + 1] + view_len[0:1],
                                                                            mask_tokens=view_len[1:])
                                            im = list(preds[0][0][1:].cpu().numpy())
                                            # print('false: ', BiSTM_tokenizer.decode(im))
                                            regenerate_ratio = (len(im) - tgt_len) / len(im)
                                            im, view_len, tgt_len = handle_biSTM_model_output_Attention_Cache_forward(
                                                im,
                                                new_suffix,
                                                backward_true_ids,
                                                BiSTM_tokenizer,
                                                segment_special_tokens,
                                                True)
                                            if im is None:
                                                print('backward also is false!!!')
                                                break
                                    candidate_h_distance = minDistance(tgt, im)
                                    end = time.time()
                                    candidate_im.append([candidate_h_distance, im, valid_segments, end - start])
                            if isFirst:
                                end = time.time()
                                tgt_len = get_First_diff(tgt, im)
                                h_distance = minDistance(tgt, im)
                                first_instance = h_distance
                                isFirst = False
                                im_res = tokenizer.decode(im)
                                valid_segments = []
                                intermediate_result6.writerow(
                                    [count, tokenizer.decode(valid_segments[1]) if len(valid_segments) > 0 else '',
                                     im_res,
                                     eval_examples[count].target,
                                     h_distance, end - start, view_len,
                                     len(valid_segments[0]) + len(valid_segments[2]) if len(valid_segments) > 0 else 0,
                                     0,
                                     success, max_same_length, p,
                                     ''])
                            else:
                                if len(candidate_im) == 0:
                                    break
                                all_candidate_h_distance = ' '.join([str(i[0]) for i in candidate_im])
                                prefix_edit_h_distance = candidate_im[-1][0]
                                # ('all_candidate_h_distance: ', all_candidate_h_distance)
                                random.shuffle(candidate_im)
                                candidate_im.sort(key=lambda x: x[0])
                                for k in range(len(candidate_im)):
                                    if candidate_im[k][0] > prefix_edit_h_distance:
                                        candidate_im = candidate_im[:k]
                                        break
                                candidate_im[0] = random.choice(candidate_im[:top_num])
                                im = candidate_im[0][1]
                                valid_segments = candidate_im[0][2]
                                im_res = tokenizer.decode(im)
                                h_distance = candidate_im[0][0]
                                tgt_len = get_First_diff(tgt, im)

                                intermediate_result6.writerow(
                                    [count, tokenizer.decode(valid_segments[1]) if len(valid_segments) > 0 else '',
                                     im_res,
                                     eval_examples[count].target,
                                     h_distance, candidate_im[0][3], view_len,
                                     sum([1 if len(x[0]) != 0 else 0 for x in valid_segments[0]]) + sum(
                                         [1 if len(x[0]) != 0 else 0 for x in valid_segments[2]]) if len(
                                         valid_segments) > 0 else 0,
                                     0,
                                     success, max_same_length, p,
                                     all_candidate_h_distance])

                            if tgt_len == -1 or tgt_len >= len(tgt) or len(im) >= 509:
                                print('bistm_total_round:', cur_num)
                                if eval_examples[count].source in stm_dic:
                                    print('stm_total_round:', stm_dic[eval_examples[count].source])
                                if eval_examples[count].source in prefix_dic:
                                    print('prefix_total_round:', prefix_dic[eval_examples[count].source])
                                intermediate_compute_result6.writerow(
                                    [count, eval_examples[count].source, eval_examples[count].target, first_instance,
                                     cur_num, max_same_length, p])
                                break

                            print('\n')
                            print('now: epoch ' + str(count))
                            print('tgt_len: ' + str(tgt_len))
                            print('h_distance: ' + str(h_distance))
                            print('\n\n')
                            cur_num += 1
                            if cur_num * h_distance > 500:
                                break
                            # if cur_num > 50:
                            #     break

            #     print('seg_NLTrans')
            #     source_ids = batch[0].to(args.device)
            #     source_mask = source_ids.ne(tokenizer.pad_token_id)
            #     tgt = tokenizer.encode(eval_examples[count].target)
            #     source = tokenizer.encode(eval_examples[count].source)
            #     origin_source_ids = source_ids
            #     source_ids1 = batch[0].to(args.device)
            #     source_mask1 = source_ids.ne(tokenizer.pad_token_id)
            #     tgt1 = tokenizer.encode(eval_examples[count].target)
            #     source1 = tokenizer.encode(eval_examples[count].source)
            #     #
            #     if len(source) >= 509 or len(tgt) >= 509:
            #         continue
            #     with torch.no_grad():
            #         # [1,-1]
            #         # -1 indicates that using the Attention cache-based recovery algorithm
            #         # 'segment_queue' in key means using a non-AC algorithm,
            #         # p<0 based on edit distance mode,p>0 based on en-de attention score mode,p=0, pure same concept
            #         for max_same_length in [0]:
            #             for p in [0]:
            #                 if eval_examples[count].source in set_dic[3]:
            #                     continue
            #                 source_ids = copy.deepcopy(source_ids1)
            #                 tgt = copy.deepcopy(tgt1)
            #                 num_return_sequences = args.beam_size
            #                 cur_num = 0
            #                 preds = []
            #                 isFirst = True
            #                 first_instance = 0
            #                 tgt_len = get_First_diff(tgt, preds)
            #                 while tgt_len != -1:
            #                     start = time.time()
            #                     prefix = tokenizer.decode(tgt[:tgt_len + 1])
            #                     if isFirst:
            #                         preds = model.generate(source_ids,
            #                                                attention_mask=source_mask,
            #                                                use_cache=True,
            #                                                num_beams=args.beam_size,
            #                                                early_stopping=args.task == 'summarize',
            #                                                max_length=args.max_target_length,
            #                                                num_return_sequences=num_return_sequences,
            #                                                prefix=[0] + tgt,
            #                                                prefix_len=tgt_len + 1,
            #                                                tokenizer=tokenizer,
            #                                                least_past=None,
            #                                                is_past=False,
            #                                                is_concent_sufix=False
            #                                                )
            #                         im = preds[0].tolist()[0][1:]
            #                         view_len = len(im)
            #                         success = -1
            #                         regenerate_ratio = 0
            #                     else:
            #                         if max_same_length == 0:
            #                             preds, _ = model.generate(origin_source_ids,
            #                                                       attention_mask=source_mask,
            #                                                       use_cache=True,
            #                                                       num_beams=args.beam_size,
            #                                                       early_stopping=args.task == 'summarize',
            #                                                       max_length=args.max_target_length,
            #                                                       num_return_sequences=num_return_sequences,
            #                                                       segment_queue=compact_true_segment,
            #                                                       least_past=None,
            #                                                       propMax_failure=True,
            #                                                       max_same_length=max_same_length,
            #                                                       # segment_dic=segment_dic,
            #                                                       p=p,
            #                                                       max_segment_length=80,
            #                                                       )
            #                         elif max_same_length > 0:
            #                             preds, _ = model.generate(origin_source_ids,
            #                                                       attention_mask=source_mask,
            #                                                       use_cache=True,
            #                                                       num_beams=args.beam_size,
            #                                                       early_stopping=args.task == 'summarize',
            #                                                       max_length=args.max_target_length,
            #                                                       num_return_sequences=num_return_sequences,
            #                                                       segment_queue=compact_true_segment,
            #                                                       least_past=None,
            #                                                       max_same_length=max_same_length,
            #                                                       p=p,
            #                                                       )
            #                         im = list(preds[0][1:].cpu().numpy())
            #                     end = time.time()
            #                     tgt_len = get_First_diff(tgt, im)
            #                     h_distance = minDistance(tgt, im)
            #                     if isFirst:
            #                         first_instance = h_distance
            #                         isFirst = False
            #                     im_res = tokenizer.decode(im)
            #                     compact_true_segment, compact_false_segment, origin_first_segment_indexs = \
            #                         get_common_segment_only_insert(im, tgt, tgt[:tgt_len + 1], is_compact=False)
            #                     sum1 = 0  # The total number of correct fragments
            #                     for s in compact_true_segment:
            #                         sum1 += len(s)
            #                     source_ids, mask_tokens = \
            #                         handle_segment_model_input(origin_source_ids, compact_true_segment, tokenizer, args)
            #                     intermediate_result7.writerow(
            #                         [count, prefix, im_res, eval_examples[count].target, h_distance, end - start,
            #                          view_len,
            #                          len(compact_true_segment), sum, success, max_same_length, p, regenerate_ratio])
            #                     if tgt_len == -1 or tgt_len >= len(tgt) or len(im) >= 509:
            #                         intermediate_compute_result7.writerow(
            #                             [count, eval_examples[count].source, eval_examples[count].target,
            #                              first_instance,
            #                              cur_num, max_same_length, p])
            #                         break
            #                     print('\n')
            #                     print('now: epoch ' + str(count))
            #                     print('tgt_len: ' + str(tgt_len))
            #                     print('h_distance: ' + str(h_distance))
            #                     print('\n\n')
            #                     cur_num += 1
            #                     if cur_num * h_distance > 500:
            #                         break
            #     print('seg_NLTrans_greedy')
            #     source_ids = batch[0].to(args.device)
            #     source_mask = source_ids.ne(tokenizer.pad_token_id)
            #     tgt = tokenizer.encode(eval_examples[count].target)
            #     source = tokenizer.encode(eval_examples[count].source)
            #     origin_source_ids = source_ids
            #     source_ids1 = batch[0].to(args.device)
            #     source_mask1 = source_ids.ne(tokenizer.pad_token_id)
            #     tgt1 = tokenizer.encode(eval_examples[count].target)
            #     source1 = tokenizer.encode(eval_examples[count].source)
            #     #
            #     if len(source) >= 509 or len(tgt) >= 509:
            #         continue
            #     with torch.no_grad():
            #         # [1,-1]
            #         # -1 indicates that using the Attention cache-based recovery algorithm
            #         # 'segment_queue' in key means using a non-AC algorithm,
            #         # p<0 based on edit distance mode,p>0 based on en-de attention score mode,p=0, pure same concept
            #         for max_same_length in [1]:
            #             for p in [0]:
            #                 if eval_examples[count].source in set_dic[4]:
            #                     continue
            #                 source_ids = copy.deepcopy(source_ids1)
            #                 tgt = copy.deepcopy(tgt1)
            #                 num_return_sequences = args.beam_size
            #                 cur_num = 0
            #                 preds = []
            #                 isFirst = True
            #                 first_instance = 0
            #                 tgt_len = get_First_diff(tgt, preds)
            #                 while tgt_len != -1:
            #                     start = time.time()
            #                     prefix = tokenizer.decode(tgt[:tgt_len + 1])
            #                     if isFirst:
            #                         preds = model.generate(source_ids,
            #                                                attention_mask=source_mask,
            #                                                use_cache=True,
            #                                                num_beams=args.beam_size,
            #                                                early_stopping=args.task == 'summarize',
            #                                                max_length=args.max_target_length,
            #                                                num_return_sequences=num_return_sequences,
            #                                                prefix=[0] + tgt,
            #                                                prefix_len=tgt_len + 1,
            #                                                tokenizer=tokenizer,
            #                                                least_past=None,
            #                                                is_past=False,
            #                                                is_concent_sufix=False
            #                                                )
            #                         im = preds[0].tolist()[0][1:]
            #                         view_len = len(im)
            #                         success = -1
            #                         regenerate_ratio = 0
            #                     else:
            #                         if max_same_length == 0:
            #                             preds, _ = model.generate(origin_source_ids,
            #                                                       attention_mask=source_mask,
            #                                                       use_cache=True,
            #                                                       num_beams=args.beam_size,
            #                                                       early_stopping=args.task == 'summarize',
            #                                                       max_length=args.max_target_length,
            #                                                       num_return_sequences=num_return_sequences,
            #                                                       segment_queue=compact_true_segment,
            #                                                       least_past=None,
            #                                                       propMax_failure=True,
            #                                                       max_same_length=max_same_length,
            #                                                       # segment_dic=segment_dic,
            #                                                       p=p,
            #                                                       max_segment_length=80,
            #                                                       )
            #                         elif max_same_length > 0:
            #                             preds, _ = model.generate(origin_source_ids,
            #                                                       attention_mask=source_mask,
            #                                                       use_cache=True,
            #                                                       num_beams=args.beam_size,
            #                                                       early_stopping=args.task == 'summarize',
            #                                                       max_length=args.max_target_length,
            #                                                       num_return_sequences=num_return_sequences,
            #                                                       segment_queue=compact_true_segment,
            #                                                       least_past=None,
            #                                                       max_same_length=max_same_length,
            #                                                       # segment_dic=segment_dic,
            #                                                       p=p,
            #                                                       )
            #                         im = list(preds[0][1:].cpu().numpy())
            #                     end = time.time()
            #                     tgt_len = get_First_diff(tgt, im)
            #                     h_distance = minDistance(tgt, im)
            #                     if isFirst:
            #                         first_instance = h_distance
            #                         isFirst = False
            #                     im_res = tokenizer.decode(im)
            #                     compact_true_segment, compact_false_segment, origin_first_segment_indexs = \
            #                         get_common_segment_only_insert(im, tgt, tgt[:tgt_len + 1], is_compact=False)
            #                     sum1 = 0  # The total number of correct fragments
            #                     for s in compact_true_segment:
            #                         sum1 += len(s)
            #                     source_ids, mask_tokens = \
            #                         handle_segment_model_input(origin_source_ids, compact_true_segment, tokenizer, args)
            #                     intermediate_result8.writerow(
            #                         [count, prefix, im_res, eval_examples[count].target, h_distance, end - start,
            #                          view_len,
            #                          len(compact_true_segment), sum, success, max_same_length, p, regenerate_ratio])
            #                     if tgt_len == -1 or tgt_len >= len(tgt) or len(im) >= 509:
            #                         intermediate_compute_result8.writerow(
            #                             [count, eval_examples[count].source, eval_examples[count].target,
            #                              first_instance,
            #                              cur_num, max_same_length, p])
            #                         break
            #                     print('\n')
            #                     print('now: epoch ' + str(count))
            #                     print('tgt_len: ' + str(tgt_len))
            #                     print('h_distance: ' + str(h_distance))
            #                     print('\n\n')
            #                     cur_num += 1
            #                     if cur_num * h_distance > 500:
            #                         break
            #
            #     print('PV-based hmCodeTrans')
            #     source_ids = batch[0].to(args.device)
            #     source_mask = source_ids.ne(tokenizer.pad_token_id)
            #     tgt = tokenizer.encode(eval_examples[count].target)
            #     source = tokenizer.encode(eval_examples[count].source)
            #     origin_source_ids = source_ids
            #     source_ids1 = batch[0].to(args.device)
            #     source_mask1 = source_ids.ne(tokenizer.pad_token_id)
            #     tgt1 = tokenizer.encode(eval_examples[count].target)
            #     source1 = tokenizer.encode(eval_examples[count].source)
            #     #
            #     if len(source) >= 509 or len(tgt) >= 509:
            #         continue
            #     with torch.no_grad():
            #         # [1,-1]
            #         for max_same_length in [-1]:
            #             for p in [0]:
            #                 # -1 indicates that using the Attention cache-based recovery algorithm
            #                 # 'segment_queue' in key means using a non-AC algorithm,
            #                 # p<0 based on edit distance mode,p>0 based on en-de attention score mode,p=0, pure same concept
            #                 if eval_examples[count].source in set_dic[1]:
            #                     continue
            #                 source_ids = copy.deepcopy(source_ids1)
            #                 tgt = copy.deepcopy(tgt1)
            #                 num_return_sequences = args.beam_size
            #                 cur_num = 0
            #                 preds = []
            #                 isFirst = True
            #                 first_instance = 0
            #                 tgt_len = get_First_diff(tgt, preds)
            #                 while tgt_len != -1:
            #                     start = time.time()
            #                     prefix = tokenizer.decode(tgt[:tgt_len + 1])
            #                     if isFirst:
            #                         preds = model.generate(source_ids,
            #                                                attention_mask=source_mask,
            #                                                use_cache=True,
            #                                                num_beams=args.beam_size,
            #                                                early_stopping=args.task == 'summarize',
            #                                                max_length=args.max_target_length,
            #                                                num_return_sequences=num_return_sequences,
            #                                                prefix=[0] + tgt,
            #                                                prefix_len=tgt_len + 1,
            #                                                tokenizer=tokenizer,
            #                                                least_past=None,
            #                                                is_past=False,
            #                                                is_concent_sufix=False
            #                                                )
            #                         im = preds[0].tolist()[0][1:]
            #                         view_len = len(im)
            #                         success = -1
            #                         regenerate_ratio = 0
            #                     else:
            #                         preds = segment_model.generate(source_ids,
            #                                                        attention_mask=source_ids.ne(tokenizer.pad_token_id),
            #                                                        use_cache=True,
            #                                                        num_beams=args.beam_size,
            #                                                        early_stopping=args.task == 'summarize',
            #                                                        max_length=args.max_target_length,
            #                                                        num_return_sequences=num_return_sequences,
            #                                                        least_past=None,
            #                                                        segment=True)
            #                     end = time.time()
            #                     if not isFirst:
            #                         im = list(preds[0][0][1:].cpu().numpy())
            #                         im, view_len = handle_segment_model_output \
            #                             (im, compact_true_segment, tokenizer, mask_tokens)
            #                         success = 1
            #                         if im is None:  # Fails, enters segment-based autoregressive editing mode
            #                             segment_dic = {}
            #                             im1 = tokenizer.encode(im_res)[1:-1]
            #                             im1 = torch.tensor([im1 + [0] * (510 - len(im1))], device='cuda:0')
            #                             im1_mask = im1.ne(tokenizer.pad_token_id)
            #                             outputs = model(
            #                                 input_ids=origin_source_ids, attention_mask=source_mask,
            #                                 labels=im1, decoder_attention_mask=im1_mask, output_attentions=True
            #                             )
            #                             cross_attention_by_head = []
            #                             for attention_score in outputs[3]:
            #                                 cross_attention_by_head.append(attention_score.mean(dim=1)[0].tolist())
            #                             cross_attention_by_head = torch.tensor(cross_attention_by_head).mean(dim=0)
            #                             for segment, (index, _) in zip(compact_true_segment[1:], origin_first_segment_indexs):
            #                                 id = str(segment[:min(len(segment), max_same_length)])
            #                                 if id not in segment_dic: segment_dic[id] = []
            #                                 segment_dic[id].append([index, segment, cross_attention_by_head[index + 2]])
            #
            #                             print('fault!!!!')
            #                             success = 0
            #                             if max_same_length == 0:  # Probability maximum algorithm
            #                                 preds, view_len = model.generate(origin_source_ids,
            #                                                                  attention_mask=source_mask,
            #                                                                  use_cache=True,
            #                                                                  num_beams=args.beam_size,
            #                                                                  early_stopping=args.task == 'summarize',
            #                                                                  max_length=args.max_target_length,
            #                                                                  num_return_sequences=num_return_sequences,
            #                                                                  segment_queue=compact_true_segment,
            #                                                                  least_past=None,
            #                                                                  max_same_length=max_same_length,
            #                                                                  # segment_dic=segment_dic,
            #                                                                  p=p,
            #                                                                  )
            #                                 end = time.time()
            #                                 im = list(preds[0][1:].cpu().numpy())
            #                             elif max_same_length == -1:  # Recovery algorithm based on Attention Cache
            #                                 im = list(preds[0][0][1:].cpu().numpy())
            #                                 im, view_len, tgt_len = handle_segment_model_output_Attention_Cache \
            #                                     (im, compact_true_segment, tokenizer, mask_tokens)
            #                                 past = preds[1]
            #                                 if past[1] is not None:
            #                                     res_tuple = []
            #                                     for i in past[1]:
            #                                         cur_tuple = []
            #                                         for index, j in enumerate(i):
            #                                             if index in [0, 1]:
            #                                                 cur_tuple.append(j[:, :, :tgt_len + 1, :])
            #                                             else:
            #                                                 cur_tuple.append(j)
            #                                         res_tuple.append(tuple(cur_tuple))
            #                                     past = tuple([past[0], tuple(res_tuple)])
            #                                 preds = segment_model.generate(source_ids,
            #                                                                attention_mask=source_ids.ne(
            #                                                                    tokenizer.pad_token_id),
            #                                                                use_cache=True,
            #                                                                num_beams=args.beam_size,
            #                                                                early_stopping=args.task == 'summarize',
            #                                                                max_length=args.max_target_length,
            #                                                                num_return_sequences=num_return_sequences,
            #                                                                least_past=past,
            #                                                                tokenizer=tokenizer,
            #                                                                segment=True,
            #                                                                prefix=list(preds[0][0].cpu().numpy())[
            #                                                                       :tgt_len + 1] + view_len[0:1],
            #                                                                mask_tokens=view_len[1:])
            #                                 end = time.time()
            #                                 im = list(preds[0][0][1:].cpu().numpy())
            #                                 print('false: ', tokenizer.decode(im))
            #                                 regenerate_ratio = (len(im) - tgt_len) / len(im)
            #                                 im, view_len, tgt_len = handle_segment_model_output_Attention_Cache \
            #                                     (im, compact_true_segment, tokenizer, mask_tokens)
            #                                 if im is None:
            #                                     print('Also is false!!!')
            #                                     break
            #                     tgt_len = get_First_diff(tgt, im)
            #                     h_distance = minDistance(tgt, im)
            #                     if isFirst:
            #                         first_instance = h_distance
            #                         isFirst = False
            #                     im_res = tokenizer.decode(im)
            #
            #                     compact_true_segment, compact_false_segment, origin_first_segment_indexs = \
            #                         get_common_segment_only_insert(im, tgt, tgt[:tgt_len + 1], is_compact=False)
            #                     sum1 = 0  # The total number of correct fragments
            #                     for s in compact_true_segment:
            #                         sum1 += len(s)
            #                     source_ids, mask_tokens = \
            #                         handle_segment_model_input(origin_source_ids, compact_true_segment, tokenizer, args)
            #                     intermediate_result5.writerow(
            #                         [count, prefix, im_res, eval_examples[count].target, h_distance, end - start,
            #                          view_len,
            #                          len(compact_true_segment), sum, success, max_same_length, p, regenerate_ratio])
            #                     if tgt_len == -1 or tgt_len >= len(tgt) or len(im) >= 509:
            #                         intermediate_compute_result5.writerow(
            #                             [count, eval_examples[count].source, eval_examples[count].target,
            #                              first_instance,
            #                              cur_num, max_same_length, p])
            #                         break
            #                     print('\n')
            #                     print('now: epoch ' + str(count))
            #                     print('tgt_len: ' + str(tgt_len))
            #                     print('h_distance: ' + str(h_distance))
            #                     print('\n\n')
            #                     cur_num += 1
            #                     # if cur_num * h_distance > 500:
            #                     #     break
            #
            # print('PC-based hmCodeTrans')
            # if eval_examples[count].source in set_dic[0]:
            #     continue
            # with torch.no_grad():
            #     source_ids = copy.deepcopy(source_ids1)
            #     tgt = copy.deepcopy(tgt1)
            #     for pattern in [2]:
            #         for classfier in [2]:
            #             num_return_sequences = args.beam_size
            #             past = None
            #             cur_num = 0
            #             preds = []
            #             isFirst = True
            #             first_instance = 0
            #             tgt_len = get_First_diff(tgt, preds)
            #             while tgt_len != -1:
            #                 start = time.time()
            #                 prefix = tokenizer.decode(tgt[:tgt_len + 1])
            #                 # The past returned at this point may only be the local state corresponding to a prefix
            #                 if isFirst:
            #                     # Needs to get the first generated sequence
            #                     preds, past, sequence_outputs, first_output = model.generate(source_ids,
            #                                                                                  attention_mask=source_mask,
            #                                                                                  use_cache=True,
            #                                                                                  num_beams=args.beam_size,
            #                                                                                  early_stopping=args.task == 'summarize',
            #                                                                                  max_length=args.max_target_length,
            #                                                                                  num_return_sequences=num_return_sequences,
            #                                                                                  prefix=tgt,
            #                                                                                  prefix_len=tgt_len + 1,
            #                                                                                  # Prefix length for real user authentication
            #                                                                                  tokenizer=tokenizer,
            #                                                                                  least_past=None,
            #                                                                                  # Some actual local state, less than or equal to the prefix length -1
            #                                                                                  is_past=True,
            #                                                                                  is_concent_sufix=True,
            #                                                                                  token_dic=None,
            #                                                                                  greedy_mode=False,
            #                                                                                  dic_prefix=dic_prefix[
            #                                                                                      count] if count in dic_prefix.keys() else {},
            #                                                                                  pattern=pattern,
            #                                                                                  classfier=classfier
            #                                                                                  )
            #                     best_suffix = -1
            #                     has_contains_false = -1
            #                 else:
            #                     preds, past, best_suffix, has_contains_false = model.generate(source_ids,
            #                                                                                   attention_mask=source_mask,
            #                                                                                   use_cache=True,
            #                                                                                   num_beams=args.beam_size,
            #                                                                                   early_stopping=args.task == 'summarize',
            #                                                                                   max_length=args.max_target_length,
            #                                                                                   num_return_sequences=num_return_sequences,
            #                                                                                   prefix=tgt,
            #                                                                                   prefix_len=tgt_len + 1,
            #                                                                                   # Prefix length for real user authentication
            #                                                                                   tokenizer=tokenizer,
            #                                                                                   least_past=past,
            #                                                                                   # Some actual local state, less than or equal to the prefix length -1
            #                                                                                   is_past=True,
            #                                                                                   is_concent_sufix=True,
            #                                                                                   sequence_outputs=sequence_outputs,
            #                                                                                   first_output=first_output,
            #                                                                                   token_dic=token_dic,
            #                                                                                   greedy_mode=False,
            #                                                                                   dic_prefix=dic_prefix[
            #                                                                                       count] if count in dic_prefix.keys() else {},
            #                                                                                   pattern=pattern,
            #                                                                                   classfier=classfier
            #                                                                                   )
            #                 end = time.time()
            #                 im = list(preds[0][1:].cpu().numpy())
            #                 tgt_len = get_First_diff(tgt, im)
            #                 if past[1] is not None:
            #                     res_tuple = []
            #                     for i in past[1]:
            #                         cur_tuple = []
            #                         for index, j in enumerate(i):
            #                             if index in [0, 1]:
            #                                 cur_tuple.append(
            #                                     j[:, :, :min(tgt_len + 1, j.shape[2]), :])  # The value of the possible state is less than the length of the prefix -1
            #                             else:
            #                                 cur_tuple.append(j)
            #                         res_tuple.append(tuple(cur_tuple))
            #                     past = tuple([past[0], tuple(res_tuple)])
            #                 h_distance = minDistance(tgt, im)
            #                 if isFirst:
            #                     first_instance = h_distance
            #                     # Processing sequence_outputs, first_output
            #                     # Hash table :[key:token,value:[(index, sequence_output)...]
            #                     # The index should be arranged from smallest to largest
            #                     isFirst = False
            #                     cur_invalid_decode = 0
            #                     token_dic = {}
            #                     for c, (token, sequence_output) in enumerate(zip(first_output, sequence_outputs)):
            #                         if token not in token_dic.keys():
            #                             token_dic[token] = [(c, sequence_output)]
            #                         else:
            #                             token_dic[token].append((c, sequence_output))
            #                 im_res = tokenizer.decode(im, clean_up_tokenization_spaces=False)
            #                 intermediate_result4.writerow(
            #                     [count, prefix, im_res, eval_examples[count].target, h_distance, end - start,
            #                      best_suffix, has_contains_false, pattern, classfier])
            #                 if tgt_len == -1 or tgt_len >= len(tgt) or len(im) >= 509:
            #                     intermediate_compute_result4.writerow(
            #                         [count, eval_examples[count].source, eval_examples[count].target, first_instance,
            #                          cur_num, pattern, classfier])
            #                     break
            #                 print('\n')
            #                 print('now: epoch ' + str(count))
            #                 print('tgt_len: ' + str(tgt_len))
            #                 print('h_distance: ' + str(h_distance))
            #                 print('\n\n')
            #                 cur_num += 1


        except Exception as e:
            print(e)
            import traceback
            print(traceback.print_exc())
            print(traceback.format_exc())
            pass


def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    t0 = time.time()

    set_dist(args)
    set_seed(args)
    if args.do_train:
        config, model, tokenizer = build_or_load_gen_model_biSTM(args)
    else:
        config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)
    if args.n_gpu > 1:
        # for DataParallel
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(args.cpu_cont)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = './tensorboard/{}'.format('/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        train_examples, train_data = load_and_cache_gen_data_biSTM(args, args.train_filename, pool, tokenizer, 'train')
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            num_workers=4,
            pin_memory=True
        )

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_train_optimization_steps
        )

        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", int(np.ceil(args.train_batch_size / args.n_gpu)))
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps * (
                        torch.distributed.get_world_size() if args.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", num_train_optimization_steps)

        dev_dataset = {}
        global_step, best_bleu_em, best_ppl = 0, -1, 1e6
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(args.device) for t in batch)
                if len(batch) == 3:  # The BiSTM model has one more type identification sequence
                    source_ids, target_ids, BiSTM_type_ids = batch
                else:
                    source_ids, target_ids = batch
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                target_mask = target_ids.ne(tokenizer.pad_token_id)

                if args.model_type == 'roberta':
                    loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                       target_ids=target_ids, target_mask=target_mask)
                else:
                    if len(batch) == 3:
                        outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                        labels=target_ids, decoder_attention_mask=target_mask,
                                        BiSTM_type_ids=BiSTM_type_ids)
                    else:
                        outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                        labels=target_ids, decoder_attention_mask=target_mask,
                                        )
                    loss = outputs[0]

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

            if args.do_eval:
                # Eval model with dev dataset
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples, eval_data = load_and_cache_gen_data_biSTM(args, args.dev_filename, pool, tokenizer,
                                                                             'dev')
                    dev_dataset['dev_loss'] = eval_examples, eval_data

                eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer)
                result = {'epoch': cur_epoch, 'global_step': global_step, 'eval_ppl': eval_ppl}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)
                if args.data_num == -1:
                    tb_writer.add_scalar('dev_ppl', eval_ppl, cur_epoch)

                # save last checkpoint
                if args.save_last_checkpoints:
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the last model into %s", output_model_file)

                if eval_ppl < best_ppl:
                    not_loss_dec_cnt = 0
                    logger.info("  Best ppl:%s", eval_ppl)
                    logger.info("  " + "*" * 20)
                    fa.write("[%d] Best ppl changed into %.4f\n" % (cur_epoch, eval_ppl))
                    best_ppl = eval_ppl

                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if args.always_save_model:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the best ppl model into %s", output_model_file)
                else:
                    not_loss_dec_cnt += 1
                    logger.info("Ppl does not decrease for %d epochs", not_loss_dec_cnt)
                    if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                        early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                            cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt
                        )
                        logger.info(early_stop_str)
                        fa.write(early_stop_str)
                        break
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                if args.do_eval_bleu:
                    eval_examples, eval_data = load_and_cache_gen_data(
                        args, args.dev_filename, pool, tokenizer, 'dev', only_src=True, is_sample=True
                    )
                    result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'dev', 'e%d' % cur_epoch)
                    dev_bleu, dev_em = result['bleu'], result['em']
                    if args.task in ['summarize']:
                        dev_bleu_em = dev_bleu
                    elif args.task in ['defect']:
                        dev_bleu_em = dev_em
                    else:
                        dev_bleu_em = dev_bleu + dev_em
                    if args.data_num == -1:
                        tb_writer.add_scalar('dev_bleu_em', dev_bleu_em, cur_epoch)
                        # tb_writer.add_scalar('dev_em', dev_em, cur_epoch)
                    if dev_bleu_em > best_bleu_em:
                        not_bleu_em_inc_cnt = 0
                        logger.info("  [%d] Best bleu+em: %.2f (bleu: %.2f, em: %.2f)",
                                    cur_epoch, dev_bleu_em, dev_bleu, dev_em)
                        logger.info("  " + "*" * 20)
                        best_bleu_em = dev_bleu_em
                        fa.write("[%d] Best bleu+em changed into %.2f (bleu: %.2f, em: %.2f)\n" % (
                            cur_epoch, best_bleu_em, dev_bleu, dev_em))
                        # Save best checkpoint for best bleu
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or args.always_save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best bleu model into %s", output_model_file)
                    else:
                        not_bleu_em_inc_cnt += 1
                        logger.info("Bleu does not increase for %d epochs", not_bleu_em_inc_cnt)
                        if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                            stop_early_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                                cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                            logger.info(stop_early_str)
                            fa.write(stop_early_str)
                            break
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)
        trained_model_path = "./g4g/python2java/paraphrase/checkpoint-best-ppl/GTM/pytorch_model.bin"  # GTM model
        if os.path.isfile(trained_model_path):
            logger.info("Reload model from {}".format(trained_model_path))
            model.load_state_dict(torch.load(trained_model_path))

            forward_model, backward_model, bistm_tokenizer = get_BISTM_model()  # BISTM forward, backward translation model
            eval_examples, eval_data = load_and_cache_gen_data(
                args, args.test_filename, pool, tokenizer, 'test', only_src=True, is_sample=False
            )
            eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'test', None, forward_model,
                            backward_model, bistm_tokenizer)
    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()


if __name__ == "__main__":
    main()
