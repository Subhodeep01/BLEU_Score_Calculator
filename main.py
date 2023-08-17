import math
import numpy as np
from collections import Counter


def upto_n(clipped_prec):
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


def bleu_score(reference, generated):
    """
    Bleu score function is given the reference, or original text, and generated/machine-translated texts.
    """
    gen_length = len(generated.split())
    ref_length = len(reference.split())

    # Brevity Penalty
    if gen_length > ref_length:
        BP = 1
    else:
        penalty = 1 - (ref_length / gen_length)
        BP = np.exp(penalty)

    # Clipped precision
    clipped_precision_score = []
    for i in range(1, min(gen_length + 1, 5)):
        ref_n_gram = Counter(generate_ngram(reference, i))
        gen_n_gram = Counter(generate_ngram(generated, i))
        c = sum(gen_n_gram.values())
        for j in gen_n_gram:
            if j in ref_n_gram:
                if gen_n_gram[j] > ref_n_gram[j]:
                    gen_n_gram[j] = ref_n_gram[j]
            else:
                gen_n_gram[j] = 0
        clipped_precision_score.append(sum(gen_n_gram.values()) / c)

    final_prec, l = upto_n(clipped_precision_score)  # fetches a list of non-zero BLEU scores and also the length
                                                     # of that list so the weight can be calculated
    # weights = [0.5] * 2  # Modifying for bleu-2, would be [0.25]*4 for bleu-4     <-----Not a useful way
    # s = (w_i * math.log(p_i) for w_i, p_i in zip(weights, clipped_precision_score))     of calculating BLEU score
    # s = BP * math.exp(math.fsum(s))
    weight = 1 / l
    s = BP * (math.prod(final_prec) ** weight)
    return s


reference = 'I go to the school'
candidate = input("Type candidate translation: ")
print(bleu_score(reference, candidate))
