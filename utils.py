import gensim
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import average_precision_score, matthews_corrcoef, f1_score, balanced_accuracy_score


def pattern_to_number(text):
    length = len(text)
    number = 0
    for i in range(length):
        if text[i] == "A":
            number += 0 * 4 ** (length - i - 1)
        elif text[i] == "C":
            number += 1 * 4 ** (length - i - 1)
        elif text[i] == "G":
            number += 2 * 4 ** (length - i - 1)
        elif text[i] == "T":
            number += 3 * 4 ** (length - i - 1)
        else:
            pass
    return number


def seq_to_tensor(seq, k=4, stride=1):
    """ Converts sequence to tensor.

    The original sequence is nucleotides whose bases are A, T, C and G. This
    function converts it to PyTorch 1-dim tensor(dtype=torch.long) with k-pts
    and stride stride.

    Example:
        >>> seq_to_tensor('ATCGATCG', k=4, stride=1)
        tensor([ 54, 216,  99, 141,  54])

    Args:
        seq: nucleotides sequence, which is supposed to be a string.
        k: length of nucleotide combination. E.g. k of 'ATCGAT' should be 6.
        stride: step when moving the k window on sequence.

    Returns:
        tensor: Pytorch 1-dim tensor representing the sequence.

    """
    seq_length = len(seq)
    tensor = []

    while seq_length > k:
        tensor.append(pattern_to_number(seq[-seq_length:-seq_length + k]))
        seq_length -= stride
    tensor.append(pattern_to_number(seq[-k:]))
    # tensor = torch.IntTensor(tensor)
    tensor = torch.tensor(tensor, dtype=torch.long)

    return tensor


def collate_fn(data_list):
    code = [i['code'] for i in data_list]
    feature = [i['feature'] for i in data_list]
    label = [i['CNRCI'] for i in data_list]
    return code, feature, label


def evaluate_metrics_sklearn(y_score, y_true, threshold=0.5):
    # y_true = y_true.astype(np.int)
    y_true = (y_true > threshold).astype(np.int)
    y_pred = (y_score > threshold).astype(np.int)
    roc_auc = metrics.roc_auc_score(y_true, y_score)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    report = metrics.classification_report(y_true, y_pred, digits=5)
    report_dict = metrics.classification_report(
        y_true, y_pred, output_dict=True)
    support = y_true.shape[0]
    # print(report)
    return accuracy, precision, recall, roc_auc, report, support, report_dict

def evaluate_metrics_sklearn_new(y_score, y_true, threshold=0.5):
    # y_true = y_true.astype(np.int)

    y_true = (y_true > threshold).astype(np.int)
    y_pred = (y_score > threshold).astype(np.int)
    roc_auc = metrics.roc_auc_score(y_true, y_score)
    bacc = balanced_accuracy_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_score)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    # print(report)
    return roc_auc, bacc, auprc, mcc, f1


def number_to_pattern(num, base):
    pattern = ""
    for i in range(base):
        div = num // 4 ** (base - i - 1)
        num = num - 4 ** (base - i - 1) * div
        if div == 0:
            pattern += "A"
        elif div == 1:
            pattern += "C"
        elif div == 2:
            pattern += "G"
        elif div == 3:
            pattern += "T"
        else:
            pass
    return pattern


def embed_from_pretrained(args):
    model = gensim.models.word2vec.Word2Vec.load('./save/' + args.cell_line + '_k=' + str(args.k) + '_vs=' + str(args.embed_num) + '.w2v')
    # model = gensim.models.word2vec.Word2Vec.load('./save/HUVEC.w2v')
    embed = torch.zeros(4 ** args.k, args.embed_num)
    for i in range(4 ** args.k):
        try:
            pattern = number_to_pattern(i, args.k)
            embed[i, :] = torch.Tensor(model.wv[pattern])
        except:
            print("Pattern {:s} not found.".format(pattern))
    return embed
