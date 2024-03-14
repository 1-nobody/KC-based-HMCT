import csv
import sys

from codet5.AST.measure_ast_complete_rate import jprocessor as tokenizer, clean_up_tokenization
import numpy as np
prefix = ''
prefix_res = prefix  # the file_path of results

def get_dic(f, m, pattern=-1, classifier=-1, has_pattern=False, is_segment=False, ):
    # Gets a dictionary of all iterations of a sample, where m is the subscript representing the minimum edit distance of the current intermediate result
    compute_transcoder = csv.reader(open(f, 'r', encoding='utf-8'))
    dic = {}
    cur_id = -1
    cur = []
    for i in compute_transcoder:
        if len(i) < 2: continue
        if has_pattern:
            if i[-2] != pattern or i[-1] != classifier:
                continue
        if int(i[0]) != cur_id:
            if cur != [] and cur[-1][-1] == '0':
                dic[cur_id] = cur
            cur = []
            cur_id = int(i[0])
        if not is_segment:
            cur.append([i[2][3:-4], i[1][3:], i[m]])
        else:
            cur.append([i[2][3:-4], i[1][3:], i[7], i[m]])
    dic[cur_id] = cur
    return dic

def get_dic_segment(f, m,pattern=0,p=0,only_false=False,only_true=False):
    # Gets a dictionary of all iterations of a sample, where m is the subscript representing the minimum edit distance of the current intermediate result
    compute_transcoder = csv.reader(open(f, 'r', encoding='utf-8'))
    dic = {}
    cur_id = -1
    cur = []
    has_false=False
    for i in compute_transcoder:
        if len(i) < 2 : continue
        if p==0 and pattern>0 and int(i[-1])!=pattern:continue
        if p>0 and ( i[-1]!=str(p) or int(i[-2])!=pattern):continue
        if int(i[0]) != cur_id:
            if cur != [] and cur[-1][-1] == '0':
                if not only_false:
                   dic[cur_id] = cur
                elif not has_false and only_true:
                    dic[cur_id] = cur
                elif has_false and only_false:
                    dic[cur_id] = cur
                    has_false=False
            cur = []
            cur_id = int(i[0])
        cur.append([i[9],i[5],i[1],i[2],i[3],i[7],i[m]])
        if int(i[9])==0:
            has_false=True
    if cur != [] and cur[-1][-1] == '0':
        if not only_false:
            dic[cur_id] = cur
        elif not has_false and only_true:
            dic[cur_id] = cur
        elif has_false and only_false:
            dic[cur_id] = cur
            has_false = False
    return dic

def get_dic_segment_with_p_pattern(f, m,pattern=0,p=0,p_list=[],only_false=False,only_true=False):
    # Gets a dictionary of all iterations of a sample, where m is the subscript representing the minimum edit distance of the current intermediate result
    compute_transcoder = csv.reader(open(f, 'r', encoding='utf-8'))
    dic = {}
    cur_id = -1
    cur = []
    has_false=False
    for i in compute_transcoder:
        if len(i) < 2 : continue
        if i[-2]!=str(pattern):continue
        if p!='ln' and i[-1]!=str(p):continue
        if p=='ln' and i[-1] in p_list:continue
        if int(i[0]) != cur_id:
            if cur != [] and cur[-1][-1] == '0':
                if not only_false:
                   dic[cur_id] = cur
                elif not has_false and only_true:
                    dic[cur_id] = cur
                elif has_false and only_false:
                    dic[cur_id] = cur
                    has_false=False
            cur = []
            cur_id = int(i[0])
        cur.append([i[9],i[5],i[1],i[2],i[3],i[7],i[m],i[12]])
        if int(i[9])==0:
            has_false=True
    if cur != [] and cur[-1][-1] == '0':
        if not only_false:
            dic[cur_id] = cur
        elif not has_false and only_true:
            dic[cur_id] = cur
        elif has_false and only_false:
            dic[cur_id] = cur
            has_false = False
    return dic


def get_dic_time(f, m, pattern, classifier, has_pattern):
    # Gets a dictionary of all iterations of a sample, where m is the subscript representing the minimum edit distance of the current intermediate result
    compute_transcoder = csv.reader(open(f, 'r', encoding='utf-8'))
    dic = {}
    cur_id = -1
    cur = []
    for i in compute_transcoder:
        if len(i) < 2: continue
        if has_pattern:
            if i[-2] != pattern or i[-1] != classifier:
                continue
        if int(i[0]) != cur_id:
            if cur != [] and cur[-1][-1] == '0':
                dic[cur_id] = cur
            cur_id = int(i[0])
            cur = []
        else:  # Remove the influence of the first translation
            cur.append([float(i[5]), i[1],i[m]])
    dic[cur_id] = cur
    return dic

def get_dic_contains_error_and_best_suffix(f, m, pattern, classifier, has_pattern):
    # Gets a dictionary of all iterations of a sample, where m is the subscript representing the minimum edit distance of the current intermediate result
    compute_transcoder = csv.reader(open(f, 'r', encoding='utf-8'))
    dic = {}
    cur_id = -1
    cur = []
    for i in compute_transcoder:
        if len(i) < 2: continue
        if has_pattern:
            if i[-2] != pattern or i[-1] != classifier:
                continue
        if int(i[0]) != cur_id:
            if cur != [] and cur[-1][-1] == '0':
                dic[cur_id] = cur
            cur_id = int(i[0])
            cur = []
        cur.append([int(i[6]),int(i[7]), i[m]])
    dic[cur_id] = cur
    return dic


def get_epoch_by_length(f, f_compute, j, k, m, pattern=0, classifier=0, has_pattern=False,is_segment=False,only_false=False,only_true=True):
    # j is the subscript of label,k is the subscript of the number of iterations, and m is the subscript representing the minimum edit distance of the current intermediate result
    if not is_segment or (not only_false and not only_true):
       dic = get_dic(f, m, pattern, classifier, has_pattern)
    else:
       dic = get_dic_segment(f, m,only_false=only_false,only_true=only_true)
    f_compute = csv.reader(open(f_compute, 'r', encoding='utf-8'))
    res = [[], [], []]
    for c, i in enumerate(f_compute):
        if c == 0 or int(i[0]) not in dic.keys(): continue
        if len(i) < 2: continue
        if has_pattern:
            if i[-2] != pattern or i[-1] != classifier:
                continue
        total_len = len(tokenizer.encode(i[j])[1:-1])  # Start and end characters are not considered
        round = int(i[k])
        if total_len <= 100:
            res[0].append(round)
        elif total_len <= 200:
            res[1].append(round)
        else:
            res[2].append(round)

    return np.mean(res[0]), np.mean(res[1]), np.mean(res[2])


def get_wsr_by_length(f, f_compute, j, k, m, pattern=0, classifier=0, has_pattern=False):
    # j is the subscript of the label,k is the subscript of the number of iterations and m is the subscript representing the minimum edit distance of the current intermediate result
    dic = get_dic(f, m, pattern, classifier, has_pattern)
    f_compute = csv.reader(open(f_compute, 'r', encoding='utf-8'))
    res = [[], [], []]
    for c, i in enumerate(f_compute):
        if c == 0 or int(i[0]) not in dic.keys(): continue
        if len(i) < 2: continue
        if has_pattern:
            if i[-2] != pattern or i[-1] != classifier:
                continue
        total_len = len(tokenizer.encode(i[j])[1:-1])  # Start and end characters are not considered
        wsr = int(i[k]) / total_len
        if total_len <= 100:
            res[0].append(wsr)
        elif total_len <= 200:
            res[1].append(wsr)
        else:
            res[2].append(wsr)

    return np.mean(res[0]), np.mean(res[1]), np.mean(res[2])


def get_segment_splice_success_rate(f, f_compute, m,pattern=0,p=0,p_list=[]):
    dic = get_dic_segment_with_p_pattern(f, m, pattern,p,p_list)
    success=0
    fault=0
    time_suc = []
    time_false = []

    ksr_suc = []
    ksr_false = []
    mar_suc = []
    mar_false = []
    ksmr_suc = []
    ksmr_false = []
    regenerate_ratio=[]

    rounds=[]
    for i,k in dic.items():
        rounds+=[len(k)]
        for j in k:
            l=len(tokenizer.encode(j[3][3:-4]))
            ksr = len(tokenizer.decode(tokenizer.encode(j[2])[-2]))/l
            mar = (1+int(j[5])*2)/l
            if int(j[0]) == 0:
                fault += 1
                time_false.append(float(j[1]))
                ksr_false.append(ksr)
                mar_false.append(mar)
                ksmr_false.append(ksr+mar)
                regenerate_ratio.append(float(j[7]))
            if int(j[0]) == 1:
                success += 1
                time_suc.append(float(j[1]))
                ksr_suc.append(ksr)
                mar_suc.append(mar)
                ksmr_suc.append(ksr + mar)

    return round(success/(success+fault),3), \
           round(np.mean(ksr_suc), 3), round(np.mean(ksr_false), 3), \
           round(np.mean(mar_suc), 3), round(np.mean(mar_false), 3), \
           round(np.mean(ksmr_suc), 3), round(np.mean(ksmr_false), 3), \
           round(np.mean(time_suc), 3), round(np.mean(time_false), 3),\
           round(np.mean(regenerate_ratio),3)

def get_segment_splice_in_mindistance(f, f_compute, m,pattern=0,p=0):
    dic = get_dic_segment(f, m, pattern,p,only_false=True)
    f_compute = csv.reader(open(f_compute, 'r', encoding='utf-8'))
    round=[]
    id=[]

    for c, i in enumerate(f_compute):
        if int(i[0]) not in dic.keys(): continue
        if len(i) < 2: continue
        if p==0 and int(i[-1]) != pattern:
            continue
        if p>0:
            if i[-1] !=str(p) or int(i[-2]) != pattern:
                continue
            round.append(int(i[4]))
        else:
            round.append(int(i[4]))
        id.append(int(i[0]))
    #print(id)
    print(round)
    return np.mean(round)





def get_kmsr_by_length(f, f_compute, j1, m, pattern=0, classifier=0, has_pattern=False, is_segment=False,only_false=False,only_true=False):
    # j is the subscript of the label, and m is the subscript representing the minimum edit distance of the current intermediate result
    if not is_segment or (not only_false and not only_true):
       dic = get_dic(f, m, pattern, classifier, has_pattern)
    else:
       dic = get_dic_segment(f, m,only_false=only_false,only_true=only_true)
    f_compute = csv.reader(open(f_compute, 'r', encoding='utf-8'))
    res_ksr = [[], [], []]
    res_mar = [[], [], []]
    res_kmsr = [[], [], []]
    ##Calculate KSR
    for c, i in enumerate(f_compute):
        if c == 0 or int(i[0]) not in dic.keys(): continue
        if len(i) < 2: continue
        if has_pattern:
            if i[-2] != pattern or i[-1] != classifier:
                continue
        i[2] = clean_up_tokenization(i[2])
        total_len = len(i[j1])  # Start and end characters are not considered
        total_len1 = len(tokenizer.encode(i[j1])[1:-1])  # Start and end characters are not considered
        last_token = []
        for j in dic[int(i[0])][1:]:
            if is_segment:
                last_token.append(tokenizer.encode(j[2])[-2])
            else:
                last_token.append(tokenizer.encode(j[1])[-2])
        last_token = [tokenizer.decode(j) for j in last_token]
        ks = 0
        for j in last_token: ks += len(j)
        if not is_segment:
            ms = len(last_token) * 2 + 1
        else:
            # 段模式下
            ms = 1
            for j in dic[int(i[0])]:
                ms += int(j[5]) * 2
        kms = ks + ms
        ksr = ks / total_len
        mar = ms / total_len
        kmsr = kms / total_len
        if total_len1 <= 100:
            res_ksr[0].append(ksr)
            res_mar[0].append(mar)
            res_kmsr[0].append(kmsr)
        elif total_len1 <= 200:
            res_ksr[1].append(ksr)
            res_mar[1].append(mar)
            res_kmsr[1].append(kmsr)
        else:
            res_ksr[2].append(ksr)
            res_mar[2].append(mar)
            res_kmsr[2].append(kmsr)
    return np.mean(res_ksr[0]), np.mean(res_ksr[1]), np.mean(res_ksr[2]), \
           np.mean(res_mar[0]), np.mean(res_mar[1]), np.mean(res_mar[2]), \
           np.mean(res_kmsr[0]), np.mean(res_kmsr[1]), np.mean(res_kmsr[2])


def get_mean_time_by_length(f, f_compute, j1, m, pattern=0, classifier=0, has_pattern=False, is_segment=False):
    # j is the subscript of the label, and m is the subscript representing the minimum edit distance of the current intermediate result
    dic = get_dic_time(f, m, pattern, classifier, has_pattern)
    f_compute = csv.reader(open(f_compute, 'r', encoding='utf-8'))
    res_mean_time = [[], [], []]
    ##Calculate KSR
    for c, i in enumerate(f_compute):
        if c == 0 or int(i[0]) not in dic.keys(): continue
        if len(i) < 2: continue
        if has_pattern:
            if i[-2] != pattern or i[-1] != classifier:
                continue
        total_len = len(tokenizer.encode(i[j1])[1:-1])  # Start and end characters are not considered
        for j in dic[int(i[0])]:
            if total_len <= 100:
                res_mean_time[0].append(j[0])
            elif total_len <= 200:
                res_mean_time[1].append(j[0])
            else:
                res_mean_time[2].append(j[0])
    return np.mean(res_mean_time[0]), np.mean(res_mean_time[1]), \
           np.mean(res_mean_time[2])

def get_mean_time_by_length_figure(f, f_compute, j1, m, pattern=0, classifier=0, has_pattern=False, is_segment=False):
    # j is the subscript of the label, and m is the subscript representing the minimum edit distance of the current intermediate result
    dic = get_dic_time(f, m, pattern, classifier, has_pattern)
    f_compute = csv.reader(open(f_compute, 'r', encoding='utf-8'))
    res_mean_time = []
    for i in range(40):
        res_mean_time.append([0])
    ##Calculate KSR
    for c, i in enumerate(f_compute):
        if c == 0 or int(i[0]) not in dic.keys(): continue
        if len(i) < 2: continue
        if has_pattern:
            if i[-2] != pattern or i[-1] != classifier:
                continue
        total_len = len(tokenizer.encode(i[j1])[1:-1])  # Start and end characters are not considered
        if total_len>=400:continue
        for j in dic[int(i[0])]:
            res_mean_time[total_len//10].append(j[0])
    for c,i in enumerate(res_mean_time):
        res_mean_time[c]=np.mean(i)
    return res_mean_time

def get_mean_time_by_suffix_figure(f, f_compute, j1, m, pattern=0, classifier=0, has_pattern=False, is_segment=False):
    # j is the subscript of the label, and m is the subscript representing the minimum edit distance of the current intermediate result
    dic = get_dic_time(f, m, pattern, classifier, has_pattern)
    f_compute = csv.reader(open(f_compute, 'r', encoding='utf-8'))
    res_mean_time = []
    for i in range(20):
        res_mean_time.append([0])
    ##Calculate KSR
    for c, i in enumerate(f_compute):
        if c == 0 or int(i[0]) not in dic.keys(): continue
        if len(i) < 2: continue
        if has_pattern:
            if i[-2] != pattern or i[-1] != classifier:
                continue
        total_len = len(tokenizer.encode(i[j1])[1:-1])  # Start and end characters are not considered
        for j in dic[int(i[0])]:
            suffix_len = total_len - len(tokenizer.encode(j[1])[1:-1])
            if suffix_len>=200:continue
            res_mean_time[suffix_len//10].append(j[0])
    for c,i in enumerate(res_mean_time):
        res_mean_time[c]=np.mean(i)
    return res_mean_time
def get_mean_time_by_epoch_figure(f, f_compute, j1, m, pattern=0, classifier=0, epoch=5,has_pattern=False):
    # j is the subscript of the label, and m is the subscript representing the minimum edit distance of the current intermediate result
    dic = get_dic_time(f, m, pattern, classifier, has_pattern)
    f_compute = csv.reader(open(f_compute, 'r', encoding='utf-8'))
    res_mean_time = []
    for i in range(epoch):
        res_mean_time.append([0])
    ##Calculate KSR
    for c, i in enumerate(f_compute):
        if c == 0 or int(i[0]) not in dic.keys(): continue
        if len(i) < 2: continue
        if has_pattern:
            if i[-2] != pattern or i[-1] != classifier:
                continue
        total_len = len(tokenizer.encode(i[j1])[1:-1])  # Start and end characters are not considered
        if len(dic[int(i[0])])!=epoch:continue
        for c,j in enumerate(dic[int(i[0])]):
            res_mean_time[c].append(j[0])
    for c,i in enumerate(res_mean_time):
        res_mean_time[c]=np.mean(i)
    return res_mean_time

def get_contains_error(f, f_compute, j1, m, pattern=0, classifier=0, has_pattern=False):
    # j is the subscript of the label, and m is the subscript representing the minimum edit distance of the current intermediate result
    dic = get_dic_contains_error_and_best_suffix(f, m, pattern, classifier, has_pattern)
    f_compute = csv.reader(open(f_compute, 'r', encoding='utf-8'))
    res_mean_time = []

    for c, i in enumerate(f_compute):
        if c == 0 or int(i[0]) not in dic.keys(): continue
        if len(i) < 2: continue
        if has_pattern:
            if i[-2] != pattern or i[-1] != classifier:
                continue
        for j in dic[int(i[0])]:
            res_mean_time.append(j[1])
    t=0
    f=0
    for i in res_mean_time:
        if i==0: f+=1
        else:t+=1

    return [t/(t+f)]


def get_contains_error_min_sentence(f, f_compute, j1, m, pattern=0, classifier=0, has_pattern=False):
    # j is the subscript of the label, and m is the subscript representing the minimum edit distance of the current intermediate result
    dic = get_dic_contains_error_and_best_suffix(f, m, pattern, classifier, has_pattern)
    f_compute = csv.reader(open(f_compute, 'r', encoding='utf-8'))
    dic_wrong_token_num = {}# Records the id and number of the wrong_token program
    dic_wrong_round_num = {}  # Records the id of the wrong_token program and the number of iterations
    for c, i in enumerate(f_compute):
        flag = [0, 0]
        if c == 0 or int(i[0]) not in dic.keys(): continue
        if len(i) < 2: continue
        if has_pattern:
            if i[-2] != pattern or i[-1] != classifier:
                continue
        for j in dic[int(i[0])]:
            if j[1] == 0:
                flag[0]=1
            if j[1] == 1:
                flag[1] =1
            if sum(flag)==2 and (j!= dic[int(i[0])][-1] and j[1]!=0):
                dic_wrong_token_num[int(i[0])] = len(tokenizer.encode(i[1]))
                dic_wrong_round_num[int(i[0])] = int(i[4])
                break
    dic_wrong_token_num=sorted(dic_wrong_token_num.items(),key=lambda x:x[1])
    #dic_wrong_round_num=sorted(dic_wrong_round_num.items(), key=lambda x: x[1])
    res=[]
    for i,j in dic_wrong_token_num:
        res+=[[i,j,dic_wrong_round_num[i]]]
    print(res)
    return res

def get_prefix_min_sentence(f, f_compute, j1, m, pattern=0, classifier=0, has_pattern=False):
    # j is the subscript of the label, and m is the subscript representing the minimum edit distance of the current intermediate result
    dic = get_dic(f, m, pattern, classifier, has_pattern=False)
    f_compute = csv.reader(open(f_compute, 'r', encoding='utf-8'))
    dic_wrong_token_num = {}# Record the id and token number of programs whose round=1 and first-round edit distance is greater than 1
    dic_wrong_round_num = {}  # Record the number of first edit distances
    for c, i in enumerate(f_compute):
        flag = [0, 0]
        if c == 0 or int(i[0]) not in dic.keys(): continue
        if len(i) < 2: continue
        if has_pattern:
            if i[-2] != pattern or i[-1] != classifier:
                continue
        if int(i[4])!=1 or int(dic[int(i[0])][0][2])<=1:continue
        dic_wrong_token_num[int(i[0])] = len(tokenizer.encode(i[1]))
        dic_wrong_round_num[int(i[0])] = int(dic[int(i[0])][0][2])
    dic_wrong_token_num=sorted(dic_wrong_token_num.items(),key=lambda x:x[1])
    #dic_wrong_round_num=sorted(dic_wrong_round_num.items(), key=lambda x: x[1])
    res=[]
    for i,j in dic_wrong_token_num:
        res+=[[i,j,dic_wrong_round_num[i]]]
    print(res)
    return res


def get_better_than_prefix_min_sentence_segment(f, f_compute, j1, m, pattern=0, classifier=0, has_pattern=False,dic_prefix=None):
    # j is the subscript of the label, and m is the subscript representing the minimum edit distance of the current intermediate result
    dic = get_dic_segment(f, m, only_false=False)
    f_compute = csv.reader(open(f_compute, 'r', encoding='utf-8'))
    dic_wrong_token_num = {}  # Records the id and number of the wrong_token program
    dic_wrong_round_num = {}  # Records the id of the wrong_token program and the number of iterations
    for c, i in enumerate(f_compute):
        if c == 0 or int(i[0]) not in dic.keys() or int(i[0]) not in dic_prefix.keys(): continue
        if len(i) < 2: continue
        if has_pattern:
            if i[-2] != pattern or i[-1] != classifier:
                continue
        if len(dic_prefix[int(i[0])])-int(i[4])<2:continue
        dic_wrong_token_num[int(i[0])] = len(tokenizer.encode(i[1]))
        dic_wrong_round_num[int(i[0])] = int(i[4])
    dic_wrong_token_num = sorted(dic_wrong_token_num.items(), key=lambda x: x[1])
    # dic_wrong_round_num=sorted(dic_wrong_round_num.items(), key=lambda x: x[1])
    res = []
    for i, j in dic_wrong_token_num:
        res += [[i, j, dic_wrong_round_num[i],len(dic_prefix[i])]]
    print(res)
    return res

def get_contains_error_min_sentence_segment(f, f_compute, j1, m, pattern=0, classifier=0, has_pattern=False):
    # j is the subscript of the label, and m is the subscript representing the minimum edit distance of the current intermediate result
    dic = get_dic_segment(f, m, only_false=True)
    f_compute = csv.reader(open(f_compute, 'r', encoding='utf-8'))
    dic_wrong_token_num = {}  # Records the id and number of the wrong_token program
    dic_wrong_round_num = {}  # Records the id of the wrong_token program and the number of iterations
    for c, i in enumerate(f_compute):
        if c == 0 or int(i[0]) not in dic.keys(): continue
        if len(i) < 2: continue
        if has_pattern:
            if i[-2] != pattern or i[-1] != classifier:
                continue
        dic_wrong_token_num[int(i[0])] = i[1]
        dic_wrong_round_num[int(i[0])] = int(i[4])
    print([i[1] for i in dic_wrong_token_num.items()])
    dic_wrong_token_num = sorted(dic_wrong_token_num.items(), key=lambda x: x[1])
    # dic_wrong_round_num=sorted(dic_wrong_round_num.items(), key=lambda x: x[1])
    res = []
    for i, j in dic_wrong_token_num:
        res += [[i, j, dic_wrong_round_num[i]]]
    print(res)
    return res

def get_best_suffix(f, f_compute, j1, m, pattern=0, classifier=0, has_pattern=False):
    # j is the subscript of the label, and m is the subscript representing the minimum edit distance of the current intermediate result
    dic = get_dic_contains_error_and_best_suffix(f, m, pattern, classifier, has_pattern)
    f_compute = csv.reader(open(f_compute, 'r', encoding='utf-8'))
    res_mean_time = []
    ##Calculate KSR
    for c, i in enumerate(f_compute):
        if c == 0 or int(i[0]) not in dic.keys(): continue
        if len(i) < 2: continue
        if has_pattern:
            if i[-2] != pattern or i[-1] != classifier:
                continue
        for j in dic[int(i[0])]:
            res_mean_time.append(j[0])
    t=0
    f=0
    for i in res_mean_time:
        if i==-2: f+=1
        elif i>=0:t+=1

    return [t/(t+f)]


import os
f0=os.path.join(prefix,'intermediate_results_0.csv')
f1=os.path.join(prefix,'intermediate_results_1.csv')
f2=os.path.join(prefix,'intermediate_results_2.csv')
f3=os.path.join(prefix,'intermediate_results_3.csv')
f4=os.path.join(prefix,'intermediate_results_4.csv')
f5=os.path.join(prefix,'intermediate_results_5.csv')
f0_compute=os.path.join(prefix,'intermediate_results_compute_0.csv')
f1_compute=os.path.join(prefix,'intermediate_results_compute_1.csv')
f2_compute=os.path.join(prefix,'intermediate_results_compute_2.csv')
f3_compute=os.path.join(prefix,'intermediate_results_compute_3.csv')
f4_compute=os.path.join(prefix,'intermediate_results_compute_4.csv')
f5_compute=os.path.join(prefix,'intermediate_results_compute_5.csv')

f_list=[f0,f1,f2,f3,f4,f5]
f_compute_list=[f0_compute,f1_compute,f2_compute,f3_compute,f4_compute,f5_compute]
clfs=['svm','gdbt','adboost']

def write_csv_file(name,res,is_figure=False,is_only_SS=False):
    method=['HMPO','HMPO-AC','HMPO-SS-SVM-20','HMPO-SS-GDBT-20','HMPO-SS-ADBOOST-20',
            'HMPO-ACSS-Greedy','HMPO-ACSS-ADBOOST-10','HMPO-ACSS-ADBOOST-20','HMPO-ACSS-ADBOOST-30','HMPO-ACSS-ADBOOST-40','HMPO-ACSS-ADBOOST-50',
            'HMPO-ACSS-SVM-20','HMPO-ACSS-GDBT-20','HM-Segment']
    if is_only_SS:
        method=['HMPO-SS-SVM-20','HMPO-SS-GDBT-20','HMPO-SS-ADBOOST-20',
            'HMPO-ACSS-Greedy','HMPO-ACSS-ADBOOST-10','HMPO-ACSS-ADBOOST-20','HMPO-ACSS-ADBOOST-30','HMPO-ACSS-ADBOOST-40','HMPO-ACSS-ADBOOST-50',
            'HMPO-ACSS-SVM-20','HMPO-ACSS-GDBT-20']
    f_res = csv.writer(open(os.path.join(prefix_res,'res_' + name + '_1.csv'), 'w', encoding='utf-8', newline=''))
    if not is_figure:
        if len(res) != len(method):
            print('错误的匹配.')
            return
        f_res.writerow(['Method', '(<=100)', '(100~200)', '(>200)'])
        for i, j in zip(method, res):
           f_res.writerow([i] + j)
    else:
         f_res.writerow(method)
         res=np.array(res).T.tolist()
         for j in res:
             f_res.writerow(j)



def write_kmsr_csv_file(name,res):
    method=['HMPO','HMPO-AC','HMPO-SS-SVM-20','HMPO-SS-GDBT-20','HMPO-SS-ADBOOST-20',
            'HMPO-ACSS-Greedy','HMPO-ADBOOST-10','HMPO-ADBOOST-20','HMPO-ADBOOST-30','HMPO-ADBOOST-40','HMPO-ADBOOST-50',
            'HMPO-ACSS-SVM-20','HMPO-ACSS-GDBT-20','HM-Segment']
    if len(res)!=len(method):
        print('错误的匹配.')
        return
    f_res = csv.writer(open(os.path.join(prefix_res,'res_'+name+'_1.csv'),'w',encoding='utf-8',newline=''))
    f_res.writerow(['Method','KSR(<=100)','KSR(100~200)','KSR(>200)',
                    'MAR(<=100)','MAR(100~200)','MAR(>200)'
                    ,'KSMR(<=100)','KSMR(100~200)','KSMR(>200)'])
    for i,j in zip(method,res):
        f_res.writerow([i]+j)

def get_round(l):
    return [round(i,3) for i in l]

def get_epoch_all(only_false = False,only_true = False):
    res_total=[]
    for c, (f, f_compute) in enumerate(zip(f_list, f_compute_list)):
        if c == 0:
            res = get_epoch_by_length(f, f_compute, 2, 4, 4)
            res_total.append(get_round(res))
        if c==1:
            res=get_epoch_by_length(f, f_compute, 2, 4, 4)
            res_total.append(get_round(res))
        if c==2:
            for i in range(3):
                res=get_epoch_by_length(f, f_compute, 2, 4, 4,pattern='2',classifier=str(i),has_pattern=True)
                res_total.append(get_round(res))
        if c==3:
            res=get_epoch_by_length(f, f_compute, 2, 4, 4)
            res_total.append(get_round(res))
        if c==4:
            for pattern in range(1, 6):
                    classfier=2
                    print(pattern,classfier)
                    pattern=str(pattern)
                    classfier=str(classfier)
                    res=get_epoch_by_length(f, f_compute, 2, 4, 4, pattern=pattern, classifier=classfier, has_pattern=True)
                    res_total.append(get_round(res))
            print('.....')
            for classfier in range(2):
                pattern=2
                print(pattern, classfier)
                pattern = str(pattern)
                classfier = str(classfier)
                res=get_epoch_by_length(f, f_compute, 2, 4, 4, pattern=pattern, classifier=classfier, has_pattern=True)
                res_total.append(get_round(res))
        if c == 5:
            res=get_epoch_by_length(f, f_compute, 2, 4, 4,is_segment=True,
                                    only_false=only_false,only_true=only_true)
            res_total.append(get_round(res))
    write_csv_file('epoch_p2j_transcoder',res_total)


def get_wsr_all():
    res_total=[]
    for c, (f, f_compute) in enumerate(zip(f_list, f_compute_list)):
        if c == 0:
            res = get_wsr_by_length(f, f_compute, 2, 4, 4)
            print(res)
            res_total.append(get_round(res))
        if c==1:
            res=get_wsr_by_length(f, f_compute, 2, 4, 4)
            print(res)
            res_total.append(get_round(res))
        if c==2:
            for i in range(3):
                res=get_wsr_by_length(f, f_compute, 2, 4, 4,pattern='2',classifier=str(i),has_pattern=True)
                print(res)
                res_total.append(get_round(res))
        if c==3:
            res=get_wsr_by_length(f, f_compute, 2, 4, 4)
            print(res)
            res_total.append(get_round(res))
        if c==4:
            for pattern in range(1, 6):
                    classfier=2
                    print(pattern,classfier)
                    pattern=str(pattern)
                    classfier=str(classfier)
                    res=get_wsr_by_length(f, f_compute, 2, 4, 4, pattern=pattern, classifier=classfier, has_pattern=True)
                    res_total.append(get_round(res))
            print('.....')
            for classfier in range(2):
                pattern=2
                print(pattern, classfier)
                pattern = str(pattern)
                classfier = str(classfier)
                res=get_wsr_by_length(f, f_compute, 2, 4, 4, pattern=pattern, classifier=classfier, has_pattern=True)
                res_total.append(get_round(res))
        if c == 5:
            res=get_wsr_by_length(f, f_compute, 2, 4, 4)
            res_total.append(get_round(res))
    write_csv_file('wsr_p2j_transcoder',res_total)


def get_kmsr_all(only_false = False,only_true = False):
    res_total=[]
    for c, (f, f_compute) in enumerate(zip(f_list, f_compute_list)):
        if c == 0:
            res = get_kmsr_by_length(f, f_compute, 2, 4)
            print(res)
            res_total.append(get_round(res))
        if c==1:
            res=get_kmsr_by_length(f, f_compute, 2, 4)
            print(res)
            res_total.append(get_round(res))
        if c==2:
            for i in range(3):
                res=get_kmsr_by_length(f, f_compute, 2,4,pattern='2',classifier=str(i),has_pattern=True)
                print(res)
                res_total.append(get_round(res))
        if c==3:
            res=get_kmsr_by_length(f, f_compute, 2,  4)
            print(res)
            res_total.append(get_round(res))
        if c==4:
            for pattern in range(1, 6):
                    classfier=2
                    print(pattern,classfier)
                    pattern=str(pattern)
                    classfier=str(classfier)
                    res=get_kmsr_by_length(f, f_compute, 2, 4,  pattern=pattern, classifier=classfier, has_pattern=True)
                    res_total.append(get_round(res))
            print('.....')
            for classfier in range(2):
                pattern=2
                print(pattern, classfier)
                pattern = str(pattern)
                classfier = str(classfier)
                res=get_kmsr_by_length(f, f_compute, 2, 4,  pattern=pattern, classifier=classfier, has_pattern=True)
                res_total.append(get_round(res))
        if c == 5:
            res=get_kmsr_by_length(f, f_compute, 2, 4, is_segment=True,only_false = only_false,only_true = only_true)
            res_total.append(get_round(res))
    write_kmsr_csv_file('kmsr_p2j_transcoder',res_total)



def get_time_all():
    res_total=[]
    for c, (f, f_compute) in enumerate(zip(f_list, f_compute_list)):
        if c == 0:
            res = get_mean_time_by_length(f, f_compute, 2, 4)
            print(res)
            res_total.append(get_round(res))
        if c==1:
            res=get_mean_time_by_length(f, f_compute, 2, 4)
            print(res)
            res_total.append(get_round(res))
        if c==2:
            for i in range(3):
                res=get_mean_time_by_length(f, f_compute, 2,4,pattern='2',classifier=str(i),has_pattern=True)
                print(res)
                res_total.append(get_round(res))
        if c==3:
            res=get_mean_time_by_length(f, f_compute, 2,  4)
            print(res)
            res_total.append(get_round(res))
        if c==4:
            for pattern in range(1, 6):
                    classfier=2
                    print(pattern,classfier)
                    pattern=str(pattern)
                    classfier=str(classfier)
                    res=get_mean_time_by_length(f, f_compute, 2, 4,  pattern=pattern, classifier=classfier, has_pattern=True)
                    res_total.append(get_round(res))
            print('.....')
            for classfier in range(2):
                pattern=2
                print(pattern, classfier)
                pattern = str(pattern)
                classfier = str(classfier)
                res=get_mean_time_by_length(f, f_compute, 2, 4,  pattern=pattern, classifier=classfier, has_pattern=True)
                res_total.append(get_round(res))
        if c == 5:
            res=get_mean_time_by_length(f, f_compute, 2, 4, is_segment=True)
            res_total.append(get_round(res))
    write_csv_file('mean_time_p2j_transcoder',res_total)

def get_mean_time_all_by_length():
    res_total=[]
    for c, (f, f_compute) in enumerate(zip(f_list, f_compute_list)):
        if c == 0:
            res = get_mean_time_by_length_figure(f, f_compute, 2, 4)
            print(res)
            res_total.append(get_round(res))
        if c==1:
            res=get_mean_time_by_length_figure(f, f_compute, 2, 4)
            print(res)
            res_total.append(get_round(res))
        if c==2:
            for i in range(3):
                res=get_mean_time_by_length_figure(f, f_compute, 2,4,pattern='2',classifier=str(i),has_pattern=True)
                print(res)
                res_total.append(get_round(res))
        if c==3:
            res=get_mean_time_by_length_figure(f, f_compute, 2,  4)
            print(res)
            res_total.append(get_round(res))
        if c==4:
            for pattern in range(1, 6):
                    classfier=2
                    print(pattern,classfier)
                    pattern=str(pattern)
                    classfier=str(classfier)
                    res=get_mean_time_by_length_figure(f, f_compute, 2, 4,  pattern=pattern, classifier=classfier, has_pattern=True)
                    res_total.append(get_round(res))
            print('.....')
            for classfier in range(2):
                pattern=2
                print(pattern, classfier)
                pattern = str(pattern)
                classfier = str(classfier)
                res=get_mean_time_by_length_figure(f, f_compute, 2, 4,  pattern=pattern, classifier=classfier, has_pattern=True)
                res_total.append(get_round(res))
        if c == 5:
            res=get_mean_time_by_length_figure(f, f_compute, 2, 4, is_segment=True)
            res_total.append(get_round(res))
    write_csv_file('mean_time_by_length_p2j_transcoder',res_total,is_figure=True)


def get_mean_time_all_by_suffix():
    res_total=[]
    for c, (f, f_compute) in enumerate(zip(f_list, f_compute_list)):
        if c == 0:
            res = get_mean_time_by_suffix_figure(f, f_compute, 2, 4)
            print(res)
            res_total.append(get_round(res))
        if c==1:
            res=get_mean_time_by_suffix_figure(f, f_compute, 2, 4)
            print(res)
            res_total.append(get_round(res))
        if c==2:
            for i in range(3):
                res=get_mean_time_by_suffix_figure(f, f_compute, 2,4,pattern='2',classifier=str(i),has_pattern=True)
                print(res)
                res_total.append(get_round(res))
        if c==3:
            res=get_mean_time_by_suffix_figure(f, f_compute, 2,  4)
            print(res)
            res_total.append(get_round(res))
        if c==4:
            for pattern in range(1, 6):
                    classfier=2
                    print(pattern,classfier)
                    pattern=str(pattern)
                    classfier=str(classfier)
                    res=get_mean_time_by_suffix_figure(f, f_compute, 2, 4,  pattern=pattern, classifier=classfier, has_pattern=True)
                    res_total.append(get_round(res))
            print('.....')
            for classfier in range(2):
                pattern=2
                print(pattern, classfier)
                pattern = str(pattern)
                classfier = str(classfier)
                res=get_mean_time_by_suffix_figure(f, f_compute, 2, 4,  pattern=pattern, classifier=classfier, has_pattern=True)
                res_total.append(get_round(res))
        if c == 5:
            res=get_mean_time_by_suffix_figure(f, f_compute, 2, 4)
            res_total.append(get_round(res))
    write_csv_file('mean_time_by_suffix_p2j_transcoder',res_total,is_figure=True)

def get_mean_time_all_by_epoch(epoch=5):
    res_total=[]
    for c, (f, f_compute) in enumerate(zip(f_list, f_compute_list)):
        if c == 0:
            res = get_mean_time_by_epoch_figure(f, f_compute, 2, 4,epoch=epoch)
            print(res)
            res_total.append(get_round(res))
        if c==1:
            res=get_mean_time_by_epoch_figure(f, f_compute, 2, 4,epoch=epoch)
            print(res)
            res_total.append(get_round(res))
        if c==2:
            for i in range(3):
                res=get_mean_time_by_epoch_figure(f, f_compute, 2,4,pattern='2',classifier=str(i),has_pattern=True,epoch=epoch)
                print(res)
                res_total.append(get_round(res))
        if c==3:
            res=get_mean_time_by_epoch_figure(f, f_compute, 2,  4,epoch=epoch)
            print(res)
            res_total.append(get_round(res))
        if c==4:
            for pattern in range(1, 6):
                    classfier=2
                    print(pattern,classfier)
                    pattern=str(pattern)
                    classfier=str(classfier)
                    res=get_mean_time_by_epoch_figure(f, f_compute, 2, 4,  pattern=pattern, classifier=classfier, has_pattern=True,epoch=epoch)
                    res_total.append(get_round(res))
            print('.....')
            for classfier in range(2):
                pattern=2
                print(pattern, classfier)
                pattern = str(pattern)
                classfier = str(classfier)
                res=get_mean_time_by_epoch_figure(f, f_compute, 2, 4,  pattern=pattern, classifier=classfier, has_pattern=True,epoch=epoch)
                res_total.append(get_round(res))
        if c == 5:
            res=get_mean_time_by_epoch_figure(f, f_compute, 2, 4,epoch=epoch)
            res_total.append(get_round(res))
    write_csv_file('mean_time_by_epoch_p2j_transcoder_'+str(epoch),res_total,is_figure=True)


def get_check_contains_error_all():
    res_total = []
    for c, (f, f_compute) in enumerate(zip(f_list, f_compute_list)):
        if c == 2:
            for i in range(3):
                res = get_contains_error(f, f_compute, 2, 4,  pattern='2', classifier=str(i), has_pattern=True)
                print(res)
                res_total.append(get_round(res))
        if c==3:
            res=get_contains_error(f, f_compute, 2,  4)
            print(res)
            res_total.append(get_round(res))
        if c == 4:
            for pattern in range(1, 6):
                classfier = 2
                print(pattern, classfier)
                pattern = str(pattern)
                classfier = str(classfier)
                res = get_contains_error(f, f_compute, 2, 4,  pattern=pattern, classifier=classfier, has_pattern=True)
                res_total.append(get_round(res))
            print('.....')
            for classfier in range(2):
                pattern = 2
                print(pattern, classfier)
                pattern = str(pattern)
                classfier = str(classfier)
                res = get_contains_error(f, f_compute, 2, 4, pattern=pattern, classifier=classfier, has_pattern=True)
                res_total.append(get_round(res))
    write_csv_file('contains_error_p2j_transcoder', res_total,is_only_SS=True)

def get_check_best_suffix_all():
    res_total = []
    for c, (f, f_compute) in enumerate(zip(f_list, f_compute_list)):
        if c == 2:
            for i in range(3):
                res = get_best_suffix(f, f_compute, 2, 4,  pattern='2', classifier=str(i), has_pattern=True)
                print(res)
                res_total.append(get_round(res))
        if c==3:
            res=get_contains_error(f, f_compute, 2,  4)
            print(res)
            res_total.append(get_round(res))
        if c == 4:
            for pattern in range(1, 6):
                classfier = 2
                print(pattern, classfier)
                pattern = str(pattern)
                classfier = str(classfier)
                res = get_best_suffix(f, f_compute, 2, 4,  pattern=pattern, classifier=classfier, has_pattern=True)
                res_total.append(get_round(res))
            print('.....')
            for classfier in range(2):
                pattern = 2
                print(pattern, classfier)
                pattern = str(pattern)
                classfier = str(classfier)
                res = get_best_suffix(f, f_compute, 2, 4, pattern=pattern, classifier=classfier, has_pattern=True)
                res_total.append(get_round(res))
    write_csv_file('best_suffix_p2j_transcoder', res_total,is_only_SS=True)

def get_check_segment_success(pattern=0):
    res_total = []
    for c, (f, f_compute) in enumerate(zip(f_list, f_compute_list)):
        if c == 5:
            res=get_segment_splice_success_rate(f, f_compute, 4,0)
            print(res)

def get_check_segment_success(pattern=0,p=0):
    res_total = []
    for c, (f, f_compute) in enumerate(zip(f_list, f_compute_list)):
        if c == 5:
            res=get_segment_splice_success_rate(f, f_compute, 4,pattern,p,[0.99,0.95,0.9,0,-5,-10])
            print(str(list(res))+',')


def get_check_segment_splice_in_mindistance(pattern,p=0):
    res_total = []
    for c, (f, f_compute) in enumerate(zip(f_list, f_compute_list)):
        if c == 5:
            res=get_segment_splice_in_mindistance(f, f_compute, 4,pattern,p=p)
            print(res)

def get_contains_error_min_sentence_all():
    res_total = []
    for c, (f, f_compute) in enumerate(zip(f_list, f_compute_list)):
        if c == 4:
            for pattern in range(2, 6):
                classfier = 2
                print(pattern, classfier)
                pattern = str(pattern)
                classfier = str(classfier)
                res = get_contains_error_min_sentence(f, f_compute, 2, 4, pattern=pattern, classifier=classfier, has_pattern=True)
                break
            print('.....')

def get_prefix_min_sentence_all():
    res_total = []
    for c, (f, f_compute) in enumerate(zip(f_list, f_compute_list)):
        if c == 1:
            res = get_prefix_min_sentence(f, f_compute, 2, 4, has_pattern=False)
        print('.....')


def get_better_than_prefix_min_sentence_all_segment():
    res_total = []
    for c, (f, f_compute) in enumerate(zip(f_list, f_compute_list)):
        if c == 1:
            dic_prefix = get_dic(f, 4)
        if c == 5:
            res = get_contains_error_min_sentence_segment(f, f_compute, 2,4,dic_prefix=dic_prefix)
            print(res)

def get_contains_error_min_sentence_all_segment():
    res_total = []
    for c, (f, f_compute) in enumerate(zip(f_list, f_compute_list)):
        if c == 5:
            res = get_contains_error_min_sentence_segment(f, f_compute, 2,4)
            print(res)



# get_epoch_all(only_true=True,only_false=False)
# get_wsr_all()
get_kmsr_all(only_true=True,only_false=False)
# get_time_all()
# get_mean_time_all_by_length()
# get_mean_time_all_by_epoch(epoch=5)
# get_mean_time_all_by_epoch(epoch=10)

# get_check_contains_error_all()
# get_check_best_suffix_all()
# get_mean_time_all_by_suffix()
# get_check_segment_success(pattern=2)
# get_check_segment_success(pattern=3)
# get_check_segment_splice_in_mindistance(pattern=1,p=0.99)
# get_check_segment_splice_in_mindistance(pattern=1,p=0.98)
# get_check_segment_splice_in_mindistance(pattern=1,p=0.95)
#get_contains_error_min_sentence_all()
#get_contains_error_min_sentence_all_segment()
#get_prefix_min_sentence_all()
# for p in [-9999]:
#     for max_same_length in [1]:
#         get_check_segment_success(pattern=max_same_length,p=p)

