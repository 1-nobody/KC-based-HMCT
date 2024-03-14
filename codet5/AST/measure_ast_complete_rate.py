
def LCS(s1, s2):
    size1 = len(s1) + 1
    size2 = len(s2) + 1
    chess = [[["", 0] for j in list(range(size2))] for i in list(range(size1))]
    for i in list(range(1, size1)):
        chess[i][0][0] = s1[i - 1]
    for j in list(range(1, size2)):
        chess[0][j][0] = s2[j - 1]
    for i in list(range(1, size1)):
        for j in list(range(1, size2)):
            if s1[i - 1] == s2[j - 1]:
                chess[i][j] = ['↖', chess[i - 1][j - 1][1] + 1]
            elif chess[i][j - 1][1] > chess[i - 1][j][1]:
                chess[i][j] = ['←', chess[i][j - 1][1]]
            else:
                chess[i][j] = ['↑', chess[i - 1][j][1]]
    i = size1 - 1
    j = size2 - 1
    s3 = []
    res = []
    is_next_true_segment = False
    while i > 0 and j > 0:
        if is_next_true_segment:
            s3.reverse()
            if len(s3) !=0:
                res.append(s3)
            s3 = []
            is_next_true_segment = False
        if chess[i][j][0] == '↖':
            s3.append(chess[i][0][0])
            i -= 1
            j -= 1
        if chess[i][j][0] == '←':
            j -= 1
            is_next_true_segment = True
        if chess[i][j][0] == '↑':
            i -= 1
            is_next_true_segment = True
    if len(s3) != 0:
        s3.reverse()
        res.append(s3)
    l = 0
    str_res = []
    true_index = []
    res.reverse()
    for i in res:
        start = 0 if len(true_index) == 0 else true_index[-1][-1]
        true_index.append(list(kmp(s2, i,start)))
    origin_true_index = []
    for i in res:
        start = 0 if len(origin_true_index) == 0 else origin_true_index[-1][-1]
        origin_true_index.append(list(kmp(s1, i, start)))
    false_index = []
    mask_len = []
    pre_last_index = 0
    for i,j in true_index:
        mask_len.append(i-pre_last_index)
        false_index.append([pre_last_index,i])
        pre_last_index = j
    false_index.append([pre_last_index,len(s2)])
    mask_len.append(len(s2) - pre_last_index)
    return true_index,false_index,origin_true_index

def get_next(T):
    i = 0
    j = -1
    next_val = [-1] * len(T)
    while i < len(T) - 1:
        if j == -1 or T[i] == T[j]:
            i += 1
            j += 1
            # next_val[i] = j
            if i < len(T) and T[i] != T[j]:
                next_val[i] = j
            else:
                next_val[i] = next_val[j]
        else:
            j = next_val[j]
    return next_val

    # KMP算法
def kmp(S, T,start=0):
    i = start
    j = 0
    next = get_next(T)
    while i < len(S) and j < len(T):
        if j == -1 or S[i] == T[j]:
            i += 1
            j += 1
        else:
            j = next[j]
    if j == len(T):
        return i - j,i
    else:
        return -1,-1


def check(true_segment,false_segment,answer):
    res = []
    for c,i in enumerate(true_segment):
        res+=i
        if false_segment[c]!=[0]:
            res+=(false_segment[c])
    if not (res==answer):
        print(res)
        print(answer)
    return res==answer
import csv
import sys

from transformers import RobertaTokenizer
import numpy as np
mask_len_list = []
pre_intermediate_result = None

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

jprocessor = codet5_tokenizer()


def re_clean_up_tokenization(out_string: str) -> str:
    """ Clean up a list of simple English tokenization artifacts like spaces before punctuations and abreviated forms.
    """
    out_string = (
        out_string.replace(".", " .")
            .replace("?", " ?")
            .replace("!", " !")
            .replace(",", " ,")
            .replace("'", " ' ")
            .replace("n't", " n't")
            .replace("'m", " 'm")
            .replace("'s", " 's")
            .replace("'ve", " 've")
            .replace("'re", " 're")
    )
    return out_string


def clean_up_tokenization(out_string: str) -> str:
        """ Clean up a list of simple English tokenization artifacts like spaces before punctuations and abreviated forms.
        """
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string

res = []
insert_time = 0
delete_time = 0
swap_time = 0

def get_common_segment(src1,src2,prefix,is_compact = True,is_python_java = False):
    origin_src1 = [int(i) for i in src1][1:]
    src1 = origin_src1
    src2 = src2[1:]
    prefix = prefix[1:]
    answer = src2
    prefix = prefix
    strat_index = len(prefix)
    operation_type = 0 #[0:swap,1:insert,2:delete]
    try:
        bias = strat_index # Calculate the subscript offset at the beginning of the second paragraph
        if jprocessor.decode(src1[strat_index-1]).strip()\
            ==jprocessor.decode(src2[strat_index]).strip():
            src1 = jprocessor.encode(' '+jprocessor.decode(src1[strat_index-1]).strip())[1:-1]+ src1[max(strat_index-2,0):]
            src2 = src2[strat_index:]
            operation_type = 1
            bias = len(origin_src1)-len(src1)
        elif jprocessor.decode(prefix[-1]).strip()==jprocessor.decode(src1[strat_index]).strip():
           src1 = src1[strat_index+1:]
           src2 = src2[strat_index:]
           operation_type = 2
           bias = strat_index+1
        else:
            src1 = src1[strat_index:]
            src2 = src2[strat_index:]
    except IndexError:
        src1 = src1[strat_index :]
        src2 = src2[strat_index:]
        print('error')
    true_index,false_index,origin_true_index = LCS(src1, src2)
    true_segment = [src2[k[0]:k[1]] for k in true_index]
    false_segment = [src2[k[0]:k[1]] if k[0]!=k[1] else [] for k in false_index]
    print(operation_type)
    if is_python_java:
        return true_segment
    if not is_compact:
        if operation_type == 1:
           true_segment[0] = [0,1]+ prefix + true_segment[0]
           false_segment=false_segment[1:-1]
           return true_segment,false_segment,operation_type,[k[0]+bias for k in origin_true_index[1:]],[k[1]+bias for k in origin_true_index[:-1]]
        else:
            true_segment=[[0,1]+prefix]+true_segment
            false_segment=false_segment[:-1]
            return true_segment, false_segment, operation_type, [k[0] + bias for k in origin_true_index],[k[1]+bias for k in origin_true_index]
    true_segment = [prefix] + true_segment
    compact_true_segment= []
    compact_false_segment= []
    origin_first_segment_indexs = [] # The index at the beginning of the common segment in the original sequence
    cur_true = []
    for j, k in zip(true_segment, false_segment):
        if k == []:
            cur_true += [j]
        else:
            cur_true += [j]
            compact_true_segment.append(cur_true)
            if len(compact_true_segment)>1:
                l=0
                for k in compact_true_segment[:-1]:
                   l+=len(k)
                origin_first_segment_indexs.append(origin_true_index[l-1][0]+bias)
            cur_true = []
            compact_false_segment.append(k)
    if cur_true != []:
        compact_true_segment.append(cur_true)
        if len(compact_true_segment) > 1:
            l = 0
            for k in compact_true_segment[:-1]:
                l += len(k)
            origin_first_segment_indexs.append(origin_true_index[l - 1][0] + bias)
    res = []
    for j in compact_true_segment:
        tmp = []
        for k in j:
            tmp+=k
        res.append(tmp)
    compact_true_segment = res
    if origin_first_segment_indexs!=[]:
       #print(operation_type)
       a=[origin_src1[j] for j in origin_first_segment_indexs]
       b=[j[0] for j in compact_true_segment[1:]]

    compact_true_segment[0]=[0,1]+compact_true_segment[0] # Add the start character to the first paragraph
    return compact_true_segment,compact_false_segment,operation_type,origin_first_segment_indexs

def get_common_segment_only_insert(src1,src2,prefix,is_compact = True):
    origin_src1 = [int(i) for i in src1][1:]
    src1 = origin_src1
    src2 = src2[1:]
    prefix = prefix[1:]
    answer = src2
    strat_index = len(prefix)
    src1 = src1[strat_index:]
    src2 = src2[strat_index:]
    true_index,false_index,origin_true_index = LCS(src1, src2)
    true_segment = [src2[k[0]:k[1]] for k in true_index]
    false_segment = [src2[k[0]:k[1]] if k[0]!=k[1] else [] for k in false_index]
    if not is_compact:
       true_segment=[[0,1]+prefix]+true_segment
       false_segment=false_segment[:-1]
       origin_true_index = [[k[0] + strat_index, k[1] + strat_index] for k in origin_true_index]
       return true_segment, false_segment, origin_true_index

def get_common_segment_only_insert_without_prefix(src1,src2):
    s1,s2=src1,src2
    size1 = len(s1) + 1
    size2 = len(s2) + 1
    chess = [[["", 0] for j in list(range(size2))] for i in list(range(size1))]
    for i in list(range(1, size1)):
        chess[i][0][0] = s1[i - 1]
    for j in list(range(1, size2)):
        chess[0][j][0] = s2[j - 1]
    for i in list(range(1, size1)):
        for j in list(range(1, size2)):
            if s1[i - 1] == s2[j - 1]:
                chess[i][j] = ['↖', chess[i - 1][j - 1][1] + 1]
            elif chess[i][j - 1][1] > chess[i - 1][j][1]:
                chess[i][j] = ['←', chess[i][j - 1][1]]
            else:
                chess[i][j] = ['↑', chess[i - 1][j][1]]
    i = size1 - 1
    j = size2 - 1
    s3 = []
    res = []
    is_next_true_segment = False
    while i > 0 and j > 0:
        if is_next_true_segment:
            s3.reverse()
            if len(s3) != 0:
                res.append(s3)
            s3 = []
            is_next_true_segment = False
        if chess[i][j][0] == '↖':
            s3.append(chess[i][0][0])
            i -= 1
            j -= 1
        if chess[i][j][0] == '←':
            j -= 1
            is_next_true_segment = True
        if chess[i][j][0] == '↑':
            i -= 1
            is_next_true_segment = True
    if len(s3) != 0:
        s3.reverse()
        res.append(s3)
    l = 0
    str_res = []
    true_index = []
    res.reverse()
    for i in res:
        start = 0 if len(true_index) == 0 else true_index[-1][-1]
        true_index.append(list(kmp(s2, i, start)))
    origin_true_index = []
    for i in res:
        start = 0 if len(origin_true_index) == 0 else origin_true_index[-1][-1]
        origin_true_index.append(list(kmp(s1, i, start)))
    false_index = []
    mask_len = []
    pre_last_index = 0
    for i, j in true_index:
        mask_len.append(i - pre_last_index)
        false_index.append([pre_last_index, i])
        pre_last_index = j
    false_index.append([pre_last_index, len(s2)])
    mask_len.append(len(s2) - pre_last_index)
    true_segment = [src2[k[0]:k[1]] for k in true_index]
    false_segment = [src2[k[0]:k[1]] if k[0]!=k[1] else [] for k in false_index]
    false_segment=false_segment
    origin_true_index = [[k[0], k[1]] for k in origin_true_index]
    origin_flase_segment = []
    pre_last_index=0
    for i in origin_true_index:
        origin_flase_segment.append(src1[pre_last_index:i[0]])
        pre_last_index=i[1]
    origin_flase_segment.append(src1[pre_last_index:])

    return true_segment, false_segment, origin_true_index,origin_flase_segment