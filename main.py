import math
import numpy as np
from collections import Counter
import copy


def calc_prec_or_rec(a, b):
    """
    if a == generated text and b == reference text
    then precision will be returned
    otherwise
    recall will be returned
    """
    r = sum(a.values())
    for k in a:
        if k in b:
            if a[k] > b[k]:
                a[k] = b[k]
        else:
            a[k] = 0
    #print(a)
    score = sum(a.values()) / r
    #print(score)
    return score


def upto_n(clipped_prec):
    if not np.nonzero(clipped_prec)[0].size > 0:
        return [0], 1
    final_prec = [x for x in clipped_prec if x != 0]
    len_fin = len(final_prec)
    return final_prec, max(len_fin, 1)


def generate_ngram(text, n=2, n_gram=False):
    """
    N-Gram generator with parameters sentence
    n is for the number of n_grams
    The n_gram parameter removes repeating n_grams
    """
    text = text.lower()  # converting to lower case
    str_arr = np.array(text.split())  # split to string arrays
    length = len(str_arr)

    word_list = []
    for i in range(length + 1):
        if i < n:
            continue
        word_range = list(range(i - n, i))
        s_list = str_arr[word_range]
        string = ' '.join(s_list)  # converting list to strings
        word_list.append(string)  # append to word_list
        if n_gram:
            word_list = list(set(word_list))
    return word_list


def g_penalty_func(list_size, n=4):
    if list_size == n:
        return 1
    l = list_size / n
    g_pen = 1 - math.exp(-3 * l)
    #print(g_pen)
    return g_pen


def mod_bleu_score(reference, generated):
    """
    Modified Bleu score function is given the reference, or original text, and generated/machine-translated texts.
    Uses both precision and recall
    """
    gen_length = len(generated.split())
    ref_length = len(reference.split())

    # Brevity Penalty
    if gen_length > ref_length:
        BP = 1
    else:
        penalty = 1 - (ref_length / gen_length)
        BP = np.exp(penalty)

    n = 5  # upto which n-gram
    clipped_precision_score = []
    recall_score = []
    for i in range(1, min(gen_length + 1, n+1)):
        ref_n_gram = Counter(generate_ngram(reference, i))
        gen_n_gram = Counter(generate_ngram(generated, i))

        ref_n_gram2 = copy.deepcopy(ref_n_gram)
        gen_n_gram2 = copy.deepcopy(gen_n_gram)

        # Calculating the recall
        recall_score.append(calc_prec_or_rec(ref_n_gram2, gen_n_gram2))

        # Calculating the clipped-precision
        clipped_precision_score.append(calc_prec_or_rec(gen_n_gram, ref_n_gram))

    print(clipped_precision_score, recall_score)
    final_prec, l1 = upto_n(clipped_precision_score)  # fetches a list of non-zero BLEU scores and also the length
    final_rec, l2 = upto_n(recall_score)  # fetches a list of non-zero BLEU scores and also the length
    g_penalty_prec = g_penalty_func(l1,n)
    g_penalty_rec = g_penalty_func(l2,n)
    # of that list so the weight can be calculated
    # weights = [0.5] * 2  # Modifying for bleu-2, would be [0.25]*4 for bleu-4     <-----Not a useful way
    # s = (w_i * math.log(p_i) for w_i, p_i in zip(weights, clipped_precision_score))     of calculating BLEU score
    # s = BP * math.exp(math.fsum(s))
    wt_prec = 1 / l1
    wt_rec = 1 / l2
    global_avg_prec1 = (math.prod(final_prec) ** wt_prec)
    global_avg_rec1 = (math.prod(final_rec) ** wt_rec)
    global_avg_prec = g_penalty_prec * (math.prod(final_prec) ** wt_prec)
    global_avg_rec = g_penalty_rec * (math.prod(final_rec) ** wt_rec)
    if global_avg_rec == global_avg_prec == 0:
        f1_score = 0
    else:
        f1_score = (2 * global_avg_rec * global_avg_prec) / (global_avg_rec + global_avg_prec)
    print(f"Precision and recall without considering g-penalty:{global_avg_prec1, global_avg_rec1}")
    print(f"Precision and recall after considering g-penalty:{global_avg_prec, global_avg_rec}")
    print(f1_score)
    s = BP * f1_score
    print(f"Original bleu-score: {BP * global_avg_prec1}")
    return s


reference = 'I go to the school'
candidate = input("Type candidate translation: ")
print(f"Modified bleu-score: {mod_bleu_score(reference, candidate)}")
