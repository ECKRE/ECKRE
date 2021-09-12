import time
import os
import random
import logging
import torch
import numpy as np
from transformers import BertTokenizer, BertConfig, AlbertConfig, AlbertTokenizer, RobertaConfig, RobertaTokenizer
from sklearn.metrics import matthews_corrcoef, f1_score, classification_report, accuracy_score

# from official_eval import official_f1
from model import ECKRE
from collections import Counter

MODEL_CLASSES = {
    'albert': (AlbertConfig, ECKRE, AlbertTokenizer)
}

MODEL_PATH_MAP = {
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'albert': 'albert-xxlarge-v1'
}

ADDITIONAL_SPECIAL_TOKENS = ["<e1s>", "<e1e>", "<e2s>", "<e2e>"]


def get_label(args):
    # get labels of dataset
    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file), 'r', encoding='utf-8')]


def get_entity_pair_type(args):
    return [label.strip() for label in
            open(os.path.join(args.data_dir, args.entity_pair_type_file), 'r', encoding='utf-8')]


def load_tokenizer(args):
    # load tokenizer from ALBERT.tokenizer
    tokenizer = MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(task, preds, labels):
    '''
    compute acc and f1 from acc_and_f1()
    :param task: task of dataset
    :param preds: prediction
    :param labels: label
    :return:
    '''
    assert len(preds) == len(labels)
    return acc_and_f1(task, preds, labels)


def simple_accuracy(preds, labels):
    '''
    acc
    '''
    return (preds == labels).mean()


def acc_and_f1(task, preds, labels):
    '''
    the process of computing acc and f1
    :param task,preds,labels:
    :return: acc and f1
    '''
    acc = simple_accuracy(preds, labels)
    if (task == "semeval"):
        no_relation = 0
        class_num = 19
        pre, recall, f1 = score(labels, preds, no_relation, class_num)
        labels_ = [i for i in range(class_num)]
        labels_.remove(no_relation)
        f1 = f1_score(y_true=labels, y_pred=preds, labels=labels_, average='macro')
    else:
        no_relation = 0
        class_num = 42
        pre, recall, f1 = score(labels, preds, no_relation, class_num)
        labels_ = [i for i in range(class_num)]
        labels_.remove(no_relation)
        f1 = f1_score(y_true=labels, y_pred=preds, labels=labels_, average='micro')

    return {
        "acc": acc,
        "pre": pre,
        "recall": recall,
        "f1": f1,
    }


def score(key, prediction, no_relation, class_num):
    '''
    Detailed steps of computing f1
    :param key: true labels
    :param prediction: predict labels
    :param no_relation: the id of "no_relation" or "other"
    :param class_num: number of classes
    :return: pre ,recall and f1
    '''

    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == no_relation and guess == no_relation:
            pass
        elif gold == no_relation and guess != no_relation:
            guessed_by_relation[guess] += 1
        elif gold != no_relation and guess == no_relation:
            gold_by_relation[gold] += 1
        elif gold != no_relation and guess != no_relation:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1
    prec_micro = 1.0
    recall_micro = 0.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    prec_macro = 0.0
    recall_macro = 0.0
    f1_macro = 0.0
    for item in gold_by_relation.keys():
        prec_macro_ = 1.0
        recall_macro_ = 0.0
        if item in correct_by_relation.keys():
            if item in guessed_by_relation.keys():
                prec_macro_ = correct_by_relation[item] / guessed_by_relation[item]
            recall_macro_ = correct_by_relation[item] / gold_by_relation[item]
        prec_macro = prec_macro + prec_macro_
        recall_macro = recall_macro + recall_macro_
        f1_macro = f1_macro + 2.0 * prec_macro_ * recall_macro_ / (prec_macro_ + recall_macro_)
    prec_macro = prec_macro / (class_num - 1)
    recall_macro = recall_macro / (class_num - 1)
    f1_macro = f1_macro / (class_num - 1)

    return prec_micro, recall_micro, f1_micro


def json_list2tensor(json_list):
    '''
    convert json list to tensor
    '''
    json_tensor = {}
    for key in json_list.keys():
        json_tensor[key] = torch.tensor(json_list[key]).cuda().float()

    return json_tensor


def json_tensor2list(json_tensor):
    '''
       convert tensor to json list
    '''
    json_list = {}
    for key in json_tensor.keys():
        json_list[key] = json_tensor[key].cpu().detach().numpy().tolist()
    return json_list


def pos_neg_pair(num_labels, features, args):
    '''
    Pair Positive and negative samples
    :param num_labels: number of label
    :param features:features(InputFeatures)
    '''

    pos_pair_ids = [] # pairs of positive samples
    neg_pair_ids = [] # pairs of positive samples
    pos_neg_pair_ids = []
    class_idx_to_sample_ids = {}
    for idx, f in enumerate(features):
        if not class_idx_to_sample_ids.__contains__(f.label_id):
            class_idx_to_sample_ids[f.label_id]=[]
        class_idx_to_sample_ids[f.label_id].append(idx)
    for l in class_idx_to_sample_ids:
        random.shuffle(class_idx_to_sample_ids[l])
        m=0
        for i in range(0, len(class_idx_to_sample_ids[l]), 2):
            m=m+1
            pair = class_idx_to_sample_ids[l][i:i + 2]
            if len(pair)<2:
                pair.append(class_idx_to_sample_ids[l][i-1])
            if l == 0:
                neg_pair_ids.append(pair)
            else:
                pos_pair_ids.append(pair)
    max_len=len(neg_pair_ids)
    if len(pos_pair_ids)>len(neg_pair_ids):
        max_len=len(pos_pair_ids)
    for i in range(max_len):
        pos_neg_pair_id=[]
        i_pos = i
        i_neg = i
        if i >= len(pos_pair_ids):
            i_pos = i % len(pos_pair_ids)
        if i >= len(neg_pair_ids):
            i_neg = i % len(neg_pair_ids)
        pos_neg_pair_id.extend(pos_pair_ids[i_pos])
        pos_neg_pair_id.extend(neg_pair_ids[i_neg])
        pos_neg_pair_ids.append(pos_neg_pair_id)

    pos_neg_pair_num_in_batch = int(args.batch_size / 4)
    batch_ids=[]
    for i in range(0, len(pos_neg_pair_ids), pos_neg_pair_num_in_batch):
        batch_id = []
        batch = pos_neg_pair_ids[i:i+pos_neg_pair_num_in_batch]
        m=1
        while len(batch) < pos_neg_pair_num_in_batch:
            batch.append(pos_neg_pair_ids[i-m])
            m = m+1
        for j in range(len(batch)):
            batch_id.extend(batch[j])
        batch_ids.append(batch_id)

    features_subs = []
    for i in range(len(batch_ids)):
        features_sub = []
        for j in range(len(batch_ids[i])):
            features_sub.append(features[batch_ids[i][j]])
        features_subs.append(features_sub)
    return  features_subs
