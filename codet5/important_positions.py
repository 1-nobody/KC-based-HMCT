import sys

sys.path.append(r'.')
from transformers import RobertaTokenizer
from AST.measure_ast_complete_rate import get_common_segment_only_insert_without_prefix
import torch


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


def codet5_tokenizer(load_extra_ids=True, add_lang_ids=False):
    tokenizer_path = r'./bpe'
    vocab_fn = '{}/codet5-vocab.json'.format(tokenizer_path)
    merge_fn = '{}/codet5-merges.txt'.format(tokenizer_path)
    tokenizer_class = RobertaTokenizer
    tokenizer = tokenizer_class(vocab_fn, merge_fn)

    tokenizer.add_special_tokens(
        {'additional_special_tokens': [
            "<pad>",
            "<s>",
            "</s>",
            "<unk>",
            "<mask>"
        ]})
    if load_extra_ids:
        tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<extra_id_{}>'.format(i) for i in range(99, -1, -1)]})
    if add_lang_ids:
        tokenizer.add_special_tokens({
            'additional_special_tokens': [
                '<en>', '<python>', '<java>', '<javascript>', '<ruby>', '<php>', '<go>', '<c>', '<c_sharp>'
            ]
        })
    # pdb.set_trace()
    tokenizer.model_max_len = 512
    return tokenizer


tokenizer = codet5_tokenizer()


def get_first_unmatch_res(program1, program2, i, j):
    tmp = [program2[j]]
    while program1[i] != 2 and program2[j] != 2 and program1[i] == program2[j]:
        i += 1
        j += 1
        tmp.append(program2[j])
    return tmp, i, j


import random


def get_all_edit_segment_and_pos(tokenizer, program1, program2):
    result = []
    n1 = len(program1)
    n2 = len(program2)
    dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
    for j in range(1, n2 + 1):
        dp[0][j] = dp[0][j - 1] + 1
    for i in range(1, n1 + 1):
        dp[i][0] = dp[i - 1][0] + 1
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            if program1[i - 1] == program2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]) + 1

    i = n1
    j = n2
    while i > 0 or j > 0:
        if dp[i][j] == dp[i - 1][j - 1] and program1[i - 1] == program2[j - 1]:
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:  # delete program[i-1]
            valid_token_after_del = i
            valid_token_before_del = i - 2
            # print('position "{}" delete "{}"'.format(tokenizer.decode(program1[:i]), str(tokenizer.decode(program1[i - 1]))))
            m = i - 2
            n = j - 1
            if program1[m] == 274 or program1[m] == 288 or program1[m] == 289 or program1[m] == 31 or program1[
                m] == 95 or program1[m] == 97 or program1[m] == 5997:  # 紧接着就是 ；（）{}
                if program1[m] == 5997:
                    m -= 2
                    n -= 2
                m -= 1
                n -= 1
            while dp[m][n] == dp[m - 1][n - 1] and program1[m] == program2[n] and program1[m] != 274 and program1[
                m] != 288 and program1[m] != 289 and program1[m] != 1 and program1[m] != 5997:
                m -= 1
                n -= 1
            if program1[m] == 274 or program1[m] == 288 or program1[m] == 289 or program1[m] == 31 or program1[
                m] == 95 or program1[m] == 97 and dp[m - 1][n] == dp[m - 2][n - 1] and program1[m - 2] == program2[
                n - 1] or program1[m] == 1 or program1[m] == 5997:
                while program1[valid_token_after_del] != program2[j]:
                    valid_token_after_del += 1
                if program1[m] == 1:
                    temp = program1[m:valid_token_before_del + 1]
                    tmp, end1, end2 = get_first_unmatch_res(program1, program2, valid_token_before_del + 1, j)
                    temp += tmp
                    result.append([temp, [m, n, valid_token_after_del + end1 - (valid_token_before_del + 1), end2]])
                else:
                    temp = program1[m + 1:valid_token_before_del + 1]
                    tmp, end1, end2 = get_first_unmatch_res(program1, program2, valid_token_before_del + 1, j)
                    temp += tmp
                    result.append(
                        [temp, [m + 1, n + 1, valid_token_after_del + end1 - (valid_token_before_del + 1), end2]])
                # print('valid segment: {}'.format(tokenizer.decode(temp)))
                # print('delete after valid segment of program1: {}'.format(
                #     tokenizer.decode(program1[valid_token_after_del + end1 - (valid_token_before_del + 1) + 1:])))
                # print('delete after valid segment of program2: {}'.format(tokenizer.decode(program2[end2 + 1:])))

            i -= 1
        elif dp[i][j] == dp[i][j - 1] + 1:  # insert program[j-1]
            valid_token_where_insert = i - 1
            # print('position "{}" insert "{}"'.format(tokenizer.decode(program1[:valid_token_where_insert + 1]),str(tokenizer.decode(program2[j - 1]))))
            m = i - 1
            n = j - 2
            if program1[m] == 274 or program1[m] == 288 or program1[m] == 289 or program1[m] == 5997 or program1[
                m] == 31 or program1[m] == 95 or program1[m] == 97:
                if program1[m] == 5997:  # python new line
                    m -= 2
                    n -= 2
                m -= 1
                n -= 1
            while dp[m][n] == dp[m - 1][n - 1] and program1[m] == program2[n] and program1[m] != 274 and program1[
                m] != 288 and program1[m] != 289 and program1[m] != 1 and program1[m] != 5997:
                m -= 1
                n -= 1
            if program1[m] == 274 or program1[m] == 288 or program1[m] == 289 or program1[m] == 31 or program1[
                m] == 95 or program1[m] == 97 and dp[m - 1][n] == dp[m - 2][n - 1] and program1[m - 2] == program2[
                n - 1] or program1[m] == 1 or program1[m] == 5997:
                if program1[m] == 1:
                    temp = program1[m:valid_token_where_insert + 1]
                    tmp, end1, end2 = get_first_unmatch_res(program1, program2, valid_token_where_insert + 1, j - 1)
                    temp += tmp
                    result.append([temp, [m, n, end1, end2]])
                else:
                    temp = program1[m + 1:valid_token_where_insert + 1]
                    tmp, end1, end2 = get_first_unmatch_res(program1, program2, valid_token_where_insert + 1, j - 1)
                    temp += tmp
                    result.append([temp, [m + 1, n + 1, end1 - 1, end2]])
                # print('valid segment: {}'.format(tokenizer.decode(temp)))
                # print('insert after valid segment of program1: {}'.format(
                #     tokenizer.decode(program1[end1:])))
                # print('insert after valid segment of program2: {}'.format(tokenizer.decode(program2[end2 + 1:])))
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:  # Exchange program1[i-1] and program2[j-1]
            valid_token_after_swap = i - 1
            # print('position "{}" swap '.format(tokenizer.decode(program1[:i])) + str(tokenizer.decode(program1[i - 1])) + '<->' + str(tokenizer.decode(program2[j - 1])))
            m = i - 2
            n = j - 2
            if program1[m] == 274 or program1[m] == 288 or program1[m] == 289 or program1[m] == 5997 or program1[
                m] == 31 or program1[m] == 95 or program1[m] == 97:
                if program1[m] == 5997:
                    m -= 2
                    n -= 2
                m -= 1
                n -= 1
            while dp[m][n] == dp[m - 1][n - 1] and program1[m] == program2[n] and program1[m] != 274 and program1[
                m] != 288 and program1[m] != 289 and program1[m] != 1 and program1[m] != 5997:
                m -= 1
                n -= 1
            if program1[m] == 274 or program1[m] == 288 or program1[m] == 289 or program1[m] == 31 or program1[
                m] == 95 or program1[m] == 97 or program1[m] == 1 or program1[m] == 5997:
                if program1[m] == 1:
                    temp = program1[m:valid_token_after_swap]
                    tmp, end1, end2 = get_first_unmatch_res(program1, program2, valid_token_after_swap, j - 1)
                    temp += tmp
                    result.append([temp, [m, n, end1, end2]])
                else:
                    temp = program1[m + 1:valid_token_after_swap]
                    tmp, end1, end2 = get_first_unmatch_res(program1, program2, valid_token_after_swap, j - 1)
                    temp += tmp
                    result.append([temp, [m + 1, n + 1, end1, end2]])
                # print('valid segment: {}'.format(tokenizer.decode(temp)))
                # print('swap after valid segment of program1: {}'.format(
                #     tokenizer.decode(program1[end1 + 1:])))
                # print('swap after valid segment of program2: {}'.format(tokenizer.decode(program2[end2 + 1:])))
            i -= 1
            j -= 1
    # if len(result) > 5:
    #     result = result[-1:] + random.sample(result, 4)
    return result


def get_res_by_biPrefixModel_suffix_splicing(args, model, biModel,
                                             tokenizer, source_ids, im, edit_segment, first_pos,
                                             token_dic, sequence_outputs, first_output,
                                             reversed_sequence_outputs,
                                             reversed_first_output, reversed_token_dic):
    num_return_sequences = args.beam_size
    prefix = im[:first_pos] + edit_segment  # [1,......,2]
    source_ids.extend([0] * (510 - len(source_ids)))
    source_ids = torch.tensor([source_ids], device='cuda:0')
    # Positive result
    preds, _, _, _ = model.generate(source_ids, attention_mask=source_ids.ne(tokenizer.pad_token_id),
                                    use_cache=False,
                                    num_beams=args.beam_size,
                                    early_stopping=args.task == 'summarize',
                                    max_length=args.max_target_length,
                                    num_return_sequences=num_return_sequences,
                                    prefix=prefix,
                                    prefix_len=-1,  # Indicates that AC is independent
                                    tokenizer=tokenizer,
                                    least_past=None,
                                    is_past=False,
                                    is_concent_sufix=True,
                                    sequence_outputs=sequence_outputs,
                                    first_output=first_output,
                                    token_dic=token_dic,
                                    greedy_mode=True,
                                    dic_prefix={}
                                    )
    im = list(preds[0][1:].cpu().numpy())  # [1,....,2]
    # print(tokenizer.decode(im))
    prefix = im[first_pos:-1]  # [....]
    prefix.reverse()  # [x1,x2]->[x2,x1]
    prefix = [1] + prefix
    if edit_segment[0] == 1:
        prefix[-1] = 2
    source_ids = source_ids.tolist()
    source_ids[0] = [tokenizer.encode('<extra_id_50>')[1]] + source_ids[0][:-1]
    source_ids = torch.tensor(source_ids, device='cuda:0')
    # 反向结果
    preds, _, _, _ = biModel.generate(source_ids, attention_mask=source_ids.ne(tokenizer.pad_token_id),
                                      use_cache=False,
                                      num_beams=args.beam_size,
                                      early_stopping=args.task == 'summarize',
                                      max_length=args.max_target_length,
                                      num_return_sequences=num_return_sequences,
                                      prefix=prefix,
                                      prefix_len=-1,  # Indicates that AC is independent
                                      tokenizer=tokenizer,
                                      least_past=None,
                                      is_past=False,
                                      is_concent_sufix=True,
                                      sequence_outputs=reversed_sequence_outputs,
                                      first_output=reversed_first_output,
                                      token_dic=reversed_token_dic,
                                      greedy_mode=True,
                                      dic_prefix={}
                                      )
    im = list(preds[0][1:].cpu().numpy())[1:-1]
    # print(preds[0][1:].cpu().numpy()[-5:])
    im.reverse()
    # print(tokenizer.decode(im))
    return [1] + im + [2]


def get_res_by_biPrefixModel(args, model, biModel, tokenizer, source_ids, im, edit_segment, first_pos):
    # edit_segment:The modification segment of the user.  first_pos：The starting position of the user's modification segment in the im
    num_return_sequences = args.beam_size
    prefix = im[:first_pos] + edit_segment  # [1,......,2]
    source_ids.extend([0] * (510 - len(source_ids)))
    source_ids = torch.tensor([source_ids], device='cuda:0')
    # Positive result
    preds, _, _, _ = model.generate(source_ids, attention_mask=source_ids.ne(tokenizer.pad_token_id),
                                    use_cache=False,
                                    num_beams=args.beam_size,
                                    early_stopping=args.task == 'summarize',
                                    max_length=args.max_target_length,
                                    num_return_sequences=num_return_sequences,
                                    prefix=prefix,
                                    prefix_len=-1,
                                    tokenizer=tokenizer,
                                    least_past=None,
                                    is_past=False,
                                    is_concent_sufix=False,
                                    dic_prefix={}
                                    )
    im = list(preds[0][1:].cpu().numpy())  # [1,....,2]
    # print(tokenizer.decode(im))
    prefix = im[first_pos:-1]  # [....]
    prefix.reverse()  # [x1,x2]->[x2,x1]
    prefix = [1] + prefix
    if edit_segment[0] == 1:
        prefix[-1] = 2
    source_ids = source_ids.tolist()
    source_ids[0] = [tokenizer.encode('<extra_id_50>')[1]] + source_ids[0][:-1]
    source_ids = torch.tensor(source_ids, device='cuda:0')
    # Reverse result
    preds, _, _, _ = biModel.generate(source_ids, attention_mask=source_ids.ne(tokenizer.pad_token_id),
                                      use_cache=False,
                                      num_beams=args.beam_size,
                                      early_stopping=args.task == 'summarize',
                                      max_length=args.max_target_length,
                                      num_return_sequences=num_return_sequences,
                                      prefix=prefix,
                                      prefix_len=-1,
                                      tokenizer=tokenizer,
                                      least_past=None,
                                      is_past=False,
                                      greedy_mode=False,
                                      is_concent_sufix=False,
                                      dic_prefix={}
                                      )
    im = list(preds[0][1:].cpu().numpy())[1:-1]
    # print(preds[0][1:].cpu().numpy()[-5:])
    im.reverse()
    # print(tokenizer.decode(im))
    return [1] + im + [2]


def get_valid_segments_in_BiModel_suffix_splicing(im, tgt, tokenizer, args, model, biModel, source_ids, token_dic,
                                                  sequence_outputs, first_output,
                                                  reversed_sequence_outputs,
                                                  reversed_first_output, reversed_token_dic, k=3):
    #
    #
    # Based on the current intermediate results and the target program, obtain the changes that the user may make in this round
    edit_segment_and_pos_list = get_all_edit_segment_and_pos(tokenizer, im, tgt)
    key_edit_list = []  # [[edit_distance,[edit_segment,pos]]]], Sort in reverse order by edit_distance
    for edit_segment, pos in edit_segment_and_pos_list:
        cur = get_res_by_biPrefixModel_suffix_splicing(args, model, biModel, tokenizer, source_ids, im, edit_segment,
                                                       pos[0], token_dic, sequence_outputs, first_output,
                                                       reversed_sequence_outputs,
                                                       reversed_first_output, reversed_token_dic)
        key_edit_list.append([minDistance(cur, tgt), [edit_segment, pos]])
    print('candidate edit distance:', str([i[0] for i in key_edit_list]))
    if len(key_edit_list) == 0: return []
    prefix_edit_distance = key_edit_list[-1][0]
    prefix_edit = key_edit_list[-1]
    key_edit_list.sort(key=lambda x: -1 * x[0])
    for i, j in enumerate(key_edit_list):
        if j[0] > prefix_edit_distance:
            key_edit_list = key_edit_list[i:]
            break
    # import random
    # select_edit_res = random.random.choice(key_edit_list[:k])
    select_edit_res = key_edit_list[0]
    if select_edit_res[0] == prefix_edit_distance:
        select_edit_res = prefix_edit

    true_segment, false_segment, origin_true_index, origin_flase_segment = get_common_segment_only_insert_without_prefix(
        im[:select_edit_res[1][1][0]], tgt[:select_edit_res[1][1][1]]
    )
    valid_segments = []
    prefix_valid_segments = []
    for i, j, k in zip(true_segment, false_segment[1:], origin_flase_segment[1:]):
        prefix_valid_segments.append([i, j, k])
    valid_segments.append(prefix_valid_segments)
    valid_segments.append(select_edit_res[1][0])
    true_segment, false_segment, origin_true_index, origin_flase_segment = get_common_segment_only_insert_without_prefix(
        im[select_edit_res[1][1][2] + 1:], tgt[select_edit_res[1][1][3] + 1:]
    )
    suffix_valid_segments = []
    for i, j, k in zip(true_segment, false_segment[:-1], origin_flase_segment[:-1]):
        suffix_valid_segments.append([i, j, k])
    valid_segments.append(suffix_valid_segments)

    return valid_segments

def get_im_in_BiModel_suffix_splicing(im, tgt, tokenizer, args, model, biModel, source_ids, token_dic,
                                                  sequence_outputs, first_output,
                                                  reversed_sequence_outputs,
                                                  reversed_first_output, reversed_token_dic, k=3):
    # Based on the current intermediate results and the target program, obtain the changes that the user may make in this round
    edit_segment_and_pos_list = get_all_edit_segment_and_pos(tokenizer, im, tgt)
    key_edit_list = []  # [[edit_distance,[edit_segment,pos]]]],Sort in reverse order by edit_distance
    for edit_segment, pos in edit_segment_and_pos_list:
        cur = get_res_by_biPrefixModel_suffix_splicing(args, model, biModel, tokenizer, source_ids, im, edit_segment,
                                                       pos[0], token_dic, sequence_outputs, first_output,
                                                       reversed_sequence_outputs,
                                                       reversed_first_output, reversed_token_dic)
        key_edit_list.append([minDistance(cur, tgt), cur])
    print('candidate edit distance:', str([i[0] for i in key_edit_list]))
    if len(key_edit_list) == 0: return []
    prefix_edit_distance = key_edit_list[-1][0]
    prefix_edit = key_edit_list[-1]
    key_edit_list.sort(key=lambda x: -1 * x[0])
    return key_edit_list[-1][1]
    for i, j in enumerate(key_edit_list):
        if j[0] > prefix_edit_distance:
            key_edit_list = key_edit_list[i:]
            break

    if select_edit_res[0] == prefix_edit_distance:
        select_edit_res = prefix_edit

    true_segment, false_segment, origin_true_index, origin_flase_segment = get_common_segment_only_insert_without_prefix(
        im[:select_edit_res[1][1][0]], tgt[:select_edit_res[1][1][1]]
    )
    valid_segments = []
    prefix_valid_segments = []
    for i, j, k in zip(true_segment, false_segment[1:], origin_flase_segment[1:]):
        prefix_valid_segments.append([i, j, k])
    valid_segments.append(prefix_valid_segments)
    valid_segments.append(select_edit_res[1][0])
    true_segment, false_segment, origin_true_index, origin_flase_segment = get_common_segment_only_insert_without_prefix(
        im[select_edit_res[1][1][2] + 1:], tgt[select_edit_res[1][1][3] + 1:]
    )
    suffix_valid_segments = []
    for i, j, k in zip(true_segment, false_segment[:-1], origin_flase_segment[:-1]):
        suffix_valid_segments.append([i, j, k])
    valid_segments.append(suffix_valid_segments)

    return valid_segments

def get_valid_segments_in_BiModel(im, tgt, tokenizer, args, model, biModel, source_ids, k=3):
    # Based on the current intermediate results and the target program, obtain the changes that the user may make in this round
    edit_segment_and_pos_list = get_all_edit_segment_and_pos(tokenizer, im, tgt)
    key_edit_list = []  # [[edit_distance,[edit_segment,pos]]]],Sort in reverse order by edit_distance
    for edit_segment, pos in edit_segment_and_pos_list:
        cur = get_res_by_biPrefixModel(args, model, biModel, tokenizer, source_ids, im, edit_segment, pos[0])
        key_edit_list.append([minDistance(cur, tgt), [edit_segment, pos]])
    if len(key_edit_list) == 0: return []
    prefix_edit_distance = key_edit_list[-1][0]
    prefix_edit = key_edit_list[-1]
    key_edit_list.sort(key=lambda x: -1 * x[0])
    for i, j in enumerate(key_edit_list):
        if j[0] > prefix_edit_distance:
            key_edit_list = key_edit_list[i:]
            break
    # import random
    # select_edit_res = random.random.choice(key_edit_list[:k])
    select_edit_res = key_edit_list[0]
    if select_edit_res[0] == prefix_edit_distance:
        select_edit_res = prefix_edit

    true_segment, false_segment, origin_true_index, origin_flase_segment = get_common_segment_only_insert_without_prefix(
        im[:select_edit_res[1][1][0]], tgt[:select_edit_res[1][1][1]]
    )
    valid_segments = []
    prefix_valid_segments = []
    for i, j, k in zip(true_segment, false_segment[1:], origin_flase_segment[1:]):
        prefix_valid_segments.append([i, j, k])
    valid_segments.append(prefix_valid_segments)
    valid_segments.append(select_edit_res[1][0])
    true_segment, false_segment, origin_true_index, origin_flase_segment = get_common_segment_only_insert_without_prefix(
        im[select_edit_res[1][1][2] + 1:], tgt[select_edit_res[1][1][3] + 1:]
    )
    suffix_valid_segments = []
    for i, j, k in zip(true_segment, false_segment[:-1], origin_flase_segment[:-1]):
        suffix_valid_segments.append([i, j, k])
    valid_segments.append(suffix_valid_segments)

    return valid_segments


def get_all_valid_segments_in_BiModel(im, tgt, tokenizer, args, model, biModel, source_ids, k=3):
    # Based on the current intermediate results and the target program, obtain the changes that the user may make in this round
    edit_segment_and_pos_list = get_all_edit_segment_and_pos(tokenizer, im, tgt)
    key_edit_list = []  # [[edit_distance,[edit_segment,pos]]]],Sort in reverse order by edit_distance
    for edit_segment, pos in edit_segment_and_pos_list:
        cur = get_res_by_biPrefixModel(args, model, biModel, tokenizer, source_ids, im, edit_segment, pos[0])
        key_edit_list.append([minDistance(cur, tgt), [edit_segment, pos]])
        # key_edit_list.append(minDistance(cur, tgt))
    # print(key_edit_list)
    # return key_edit_list
    if len(key_edit_list) == 0: return []
    prefix_edit_distance = key_edit_list[-1][0]
    res_segments = []
    for select_edit_res in key_edit_list:
        if select_edit_res[0] > prefix_edit_distance: continue
        valid_segments = []
        true_segment, false_segment, origin_true_index, origin_flase_segment = get_common_segment_only_insert_without_prefix(
            im[:select_edit_res[1][1][0]], tgt[:select_edit_res[1][1][1]]
        )
        prefix_valid_segments = []
        for i, j, k in zip(true_segment, false_segment[1:], origin_flase_segment[1:]):
            prefix_valid_segments.append([i, j, k])
        valid_segments.append(prefix_valid_segments)
        valid_segments.append(select_edit_res[1][0])
        true_segment, false_segment, origin_true_index, origin_flase_segment = get_common_segment_only_insert_without_prefix(
            im[select_edit_res[1][1][2] + 1:], tgt[select_edit_res[1][1][3] + 1:]
        )
        suffix_valid_segments = []
        for i, j, k in zip(true_segment, false_segment[:-1], origin_flase_segment[:-1]):
            suffix_valid_segments.append([i, j, k])
        valid_segments.append(suffix_valid_segments)
        # valid_segments.append(im[:select_edit_res[1][1][0]])
        # valid_segments.append(tgt[select_edit_res[1][1][1]+len(select_edit_res[1][0]):])
        res_segments.append(valid_segments)
    return res_segments


def get_all_candidate_valid_segments_in_BiModel(im, tgt, tokenizer, args, k=3):
    # Based on the current intermediate results and the target program, obtain the changes that the user may make in this round
    edit_segment_and_pos_list = get_all_edit_segment_and_pos(tokenizer, im, tgt)
    key_edit_list = []  # [[edit_distance,[edit_segment,pos]]]],Sort in reverse order by edit_distance
    for edit_segment, pos in edit_segment_and_pos_list:
        key_edit_list.append([-1, [edit_segment, pos]])
    if len(key_edit_list) == 0: return []
    res_segments = []
    for select_edit_res in key_edit_list:
        valid_segments = []
        true_segment, false_segment, origin_true_index, origin_flase_segment = get_common_segment_only_insert_without_prefix(
            im[:select_edit_res[1][1][0]], tgt[:select_edit_res[1][1][1]]
        )
        prefix_valid_segments = []
        for i, j, k in zip(true_segment, false_segment[1:], origin_flase_segment[1:]):
            prefix_valid_segments.append([i, j, k])
        valid_segments.append(prefix_valid_segments)
        valid_segments.append(select_edit_res[1][0])
        true_segment, false_segment, origin_true_index, origin_flase_segment = get_common_segment_only_insert_without_prefix(
            im[select_edit_res[1][1][2] + 1:], tgt[select_edit_res[1][1][3] + 1:]
        )
        suffix_valid_segments = []
        for i, j, k in zip(true_segment, false_segment[:-1], origin_flase_segment[:-1]):
            suffix_valid_segments.append([i, j, k])
        valid_segments.append(suffix_valid_segments)
        res_segments.append(valid_segments)
    return res_segments



import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



from utils import *

logger = logging.getLogger(__name__)


class BiSTMInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 BiSTM_type_id,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url
        self.BiSTM_type_id = BiSTM_type_id


def handle_BiSTM_type_id_to_input_all(input, BiSTM_type_id, is_revered):
    new_input = []
    if is_revered:
        new_input.append(input[0])
    cur_id = -1
    for i in range(len(BiSTM_type_id)):
        if BiSTM_type_id[i] != cur_id:
            if cur_id != -1:
                new_input.append(cur_id)
            cur_id = BiSTM_type_id[i]
            new_input.append(cur_id)
        if cur_id == 0: break
        new_input.append(input[i])
    return new_input

def handle_BiSTM_type_id_to_input(input, BiSTM_type_id, BISTM_tokenizer,is_revered):
    if is_revered:
        return input
    # Map the start and end of a special token
    special_token_start_end_dic={
        '<extra_id_32104>':'<extra_id_32107>'
    , '<extra_id_32105>':'<extra_id_32108>'
    , '<extra_id_32106>':'<extra_id_32109>'
    }
    new_input = []
    for i in range(len(BiSTM_type_id)-1):
        if BiSTM_type_id[i]=='<pad>':
            break
        new_input.append(input[i])
        if BiSTM_type_id[i]!=BiSTM_type_id[i+1]:
           if BiSTM_type_id[i] in special_token_start_end_dic:
               new_input.append(BISTM_tokenizer.encode(special_token_start_end_dic[BiSTM_type_id[i]])[1])
           if BiSTM_type_id[i + 1] in special_token_start_end_dic:
               new_input.append(BISTM_tokenizer.encode(BiSTM_type_id[i+1])[1])
    return new_input

def read_BiSTM_examples(args, tokenizer, filename, data_num, is_revered=False):
    """Read examples from filename."""
    examples = []
    row_data = np.load(filename, allow_pickle=True)  # [[[],[],[]],[[],[],[]],...,]
    for idx, group in enumerate(row_data):
        input, output, BiSTM_type_id = group[0 if not is_revered else 1]
        input = handle_BiSTM_type_id_to_input(input, BiSTM_type_id, tokenizer, is_revered)
        input = input + [0] * max(0, args.max_source_length - len(input))
        output = output + [0] * max(0, args.max_target_length - len(output))
        BiSTM_type_id = BiSTM_type_id + [0] * max(0, args.max_source_length - len(BiSTM_type_id))
        input = input[:args.max_source_length]
        BiSTM_type_id = BiSTM_type_id[:args.max_source_length]
        output = output[:args.max_target_length]
        examples.append(
            BiSTMInputFeatures(
                example_id=idx,
                source_ids=input,
                target_ids=output,
                BiSTM_type_id=BiSTM_type_id,
            )
        )
        if idx == data_num:
            break
    return examples


def load_and_cache_gen_data_biSTM(args, filename, pool, tokenizer, split_tag, only_src=False, is_sample=False):
    # filename is the npy file that holds the BiSTM data set
    # cache the data into args.cache_path except it is sampled
    # only_src: control whether to return only source ids for bleu evaluating (dev/test)
    # return: examples (Example object), data (TensorDataset)

    data_tag = '_all' if args.data_num == -1 else '_%d' % args.data_num
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + ('_src' if only_src else '') + data_tag)
    filename = args.data_dir
    if args.backward:
        is_revered = True
    elif args.forward:
        is_revered = False
    if split_tag == 'train':
        filename += '/dataset_train_p2j.npy'
    if split_tag == 'dev':
        filename += '/dataset_valid_p2j.npy'
    # examples = read_BiSTM_examples(args, tokenizer, filename, args.data_num, is_revered)
    examples = read_BiSTM_examples(args, tokenizer, filename, args.data_num, is_revered)
    if is_sample:
        examples = random.sample(examples, min(5000, len(examples)))

    if os.path.exists(cache_fn) and not is_sample and False:
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 5k data for computing bleu from %s", filename)
        else:
            logger.info("Create cache data into %s", cache_fn)
        # tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
        # features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        all_source_ids = torch.tensor([f.source_ids for f in examples], dtype=torch.long)
        if split_tag == 'test' or only_src:
            data = TensorDataset(all_source_ids)
        else:
            all_target_ids = torch.tensor([f.target_ids for f in examples], dtype=torch.long)
            all_BiSTM_type_ids = torch.tensor([tokenizer.encode(f.BiSTM_type_id)[1:-1] for f in examples],
                                              dtype=torch.long)
            data = TensorDataset(all_source_ids, all_target_ids, all_BiSTM_type_ids, )
        if args.local_rank in [-1, 0] and not is_sample:
            torch.save(data, cache_fn)
    return examples, data


class BiSTMModel(torch.nn.Module):
    def __init__(self, model, config, tokenizer, args):
        super(BiSTMModel, self).__init__()
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.emb = torch.nn.Embedding(config.vocab_size, config.d_model)
        self.emb.load_state_dict({'weight': model.state_dict()['shared.weight']})

    def forward(self, input_ids, attention_mask, labels, decoder_attention_mask, nums):
        inputs_embeds = self.emb(input_ids) + self.segment_emb(nums)
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                             decoder_attention_mask=decoder_attention_mask,
                             labels=labels)
        return outputs


def load_codet5_biSTM(config, model, tokenizer_class, load_extra_ids=True, add_lang_ids=False, load_extra_num=True,
                      tokenizer_path=''):
    # load_extra_num -> Whether additional id needs to be loaded
    vocab_fn = '{}/codet5-vocab.json'.format(tokenizer_path)
    merge_fn = '{}/codet5-merges.txt'.format(tokenizer_path)
    tokenizer = tokenizer_class(vocab_fn, merge_fn)

    tokenizer.add_special_tokens(
        {'additional_special_tokens': [
            "<pad>",
            "<s>",
            "</s>",
            "<unk>",
            "<mask>"
        ]})
    if load_extra_ids:
        tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<extra_id_{}>'.format(i) for i in range(99, -1, -1)]})
    if add_lang_ids:
        tokenizer.add_special_tokens({
            'additional_special_tokens': [
                '<en>', '<python>', '<java>', '<javascript>', '<ruby>', '<php>', '<go>', '<c>', '<c_sharp>'
            ]
        })
    tokenizer.add_special_tokens({'additional_special_tokens': ['<seq>', '<0_token>',
                                                                '<reversed>']})
    if load_extra_num:
        old_len_tokenizer = len(tokenizer)
        tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<extra_id_{}>'.format(i) for i in
                                           range(old_len_tokenizer, old_len_tokenizer + 6, 1)]})
    # pdb.set_trace()
    tokenizer.model_max_len = 512
    config.num_labels = 1
    config.vocab_size = len(tokenizer)
    config.pad_token_id = 0
    config.bos_token_id = 1
    config.eos_token_id = 2

    model.config = config  # changing the default eos_token_id from 1 to 2
    model.resize_token_embeddings(len(tokenizer))
    return config, model, tokenizer


def load_codet5_biSTM_Tokenizer():
    # load_extra_num -> Whether additional id needs to be loaded
    tokenizer_path = r'.\bpe'
    vocab_fn = '{}/codet5-vocab.json'.format(tokenizer_path)
    merge_fn = '{}/codet5-merges.txt'.format(tokenizer_path)
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer(vocab_fn, merge_fn)

    tokenizer.add_special_tokens(
        {'additional_special_tokens': [
            "<pad>",
            "<s>",
            "</s>",
            "<unk>",
            "<mask>"
        ]})
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<extra_id_{}>'.format(i) for i in range(99, -1, -1)]})

    tokenizer.add_special_tokens({'additional_special_tokens': ['<seq>', '<0_token>',
                                                                '<reversed>']})
    old_len_tokenizer = len(tokenizer)
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<extra_id_{}>'.format(i) for i in
                                       range(old_len_tokenizer, old_len_tokenizer + 6, 1)]})
    # pdb.set_trace()
    tokenizer.model_max_len = 600
    return tokenizer
