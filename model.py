import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
from torch.nn.parameter import Parameter
from transformers import BertModel, BertPreTrainedModel, RobertaModel, AlbertModel
from torch.nn.modules.module import Module

PRETRAINED_MODEL_MAP = {
    'bert': BertModel,
    'roberta': RobertaModel,
    'albert': AlbertModel
}


class ECKRE(BertPreTrainedModel):
    def __init__(self, bert_config, args):
        super(ECKRE, self).__init__(bert_config)
        self.albert = PRETRAINED_MODEL_MAP[args.model_type].from_pretrained(args.model_name_or_path,
                                                                            config=bert_config)  # Load pretrained bert
        self.num_labels = bert_config.num_labels
        self.cls_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        self.k_fc_layer = FCLayer(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        self.label_classifier = FCLayer(bert_config.hidden_size * 4, bert_config.num_labels, args.dropout_rate,
                                        use_activation=False)
        self.ss_gcn = SSGCN(bert_config.hidden_size, bert_config.hidden_size, args.dropout_rate)
        self.center = []

    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask,
                e1_ids, e2_ids, graph, edge_feature, edge_type, alpha, beta, mode):
        """
        forward process of SRGLHRE
        process: ALBERT -> SRG -> classification
        :param input_ids,attention_mask,token_type_ids,labels,e1_mask,e2_mask,e1_ids,e2_ids,graph,edge_feature
        :return: loss the final loss
        :return: Pr the corrected prediction results
        """
        # get output from ALBERT(hidden states of tokens and CLS)
        outputs = self.albert(input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]
        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        length = len(e1_ids)
        batch_k = torch.Tensor().cuda()
        batch_pr_ = torch.Tensor().cuda()

        # mine knowledge about each sample in batch
        for i in range(length):
            e1_id = int(e1_ids[i])
            e2_id = int(e2_ids[i])
            label = torch.zeros((self.num_labels))
            label[labels[i]] = 1
            label = label.cuda()
            self.update(pooled_output[i], e1_id, e2_id, graph, edge_feature, edge_type, label)
            e1_neighbors = graph[str(e1_id)]
            e2_neighbors = graph[str(e2_id)]
            adj_edges_h = []
            adj_edges_id = []

            for e_id in e1_neighbors:
                str_e1_id = str(e1_id)
                str_e_id = str(e_id)
                if e_id < e1_id:
                    str_e1_id, str_e_id = str_e_id, str_e1_id
                adj_edges_h.append(edge_feature[str_e1_id + "-" + str_e_id])
                adj_edges_id.append(str_e1_id + "-" + str_e_id)

            for e_id in e2_neighbors:
                str_e2_id = str(e2_id)
                str_e_id = str(e_id)
                if e_id < e2_id:
                    str_e2_id, str_e_id = str_e_id, str_e2_id
                adj_edges_h.append(edge_feature[str_e2_id + "-" + str_e_id])
                adj_edges_id.append(str_e2_id + "-" + str_e_id)
            adj_edges_h = torch.from_numpy(np.array(adj_edges_h)).cuda().float()
            k, most_sim = self.ss_gcn(pooled_output[i], adj_edges_h)
            pr_ = edge_type[adj_edges_id[most_sim]]
            batch_k = torch.cat((batch_k, k), 0)
            batch_pr_ = torch.cat((batch_pr_, pr_), 0)
        knowledge = batch_k.reshape(length, -1)
        knowledge = self.k_fc_layer(knowledge)
        batch_pr_ = batch_pr_.reshape(length, -1)
        pooled_output = self.cls_fc_layer(pooled_output)
        context = torch.cat([pooled_output, e1_h, e2_h], dim=-1)

        feature = torch.cat((context, knowledge), dim=-1)
        logits = self.label_classifier(feature)

        if mode == 'train':
            dot_product = torch.matmul(feature, torch.transpose(feature, 0, 1))
            square_norm = torch.diag(dot_product)
            square_norm = torch.sqrt(square_norm)
            embeddings = torch.div(feature, square_norm.unsqueeze(1))
            tri_loss = triplet_loss(embeddings, labels)
            CE = nn.CrossEntropyLoss()
            c_loss = CE(logits.view(-1, self.num_labels), labels.view(-1))
            loss = alpha * c_loss + (1 - alpha) * tri_loss
            softmax = nn.Softmax(dim=1)
            pr = softmax(logits)
        else:
            loss = 0
            softmax = nn.Softmax(dim=1)
            pr = softmax(logits)
            pr = (1 - beta) * pr + beta * batch_pr_
            for i in range(length):
                e1_id = int(e1_ids[i])
                e2_id = int(e2_ids[i])
                self.update_type(e1_id, e2_id, edge_type, pr[i])
        return loss, pr

    def update(self, cls_h, e1_id, e2_id, graph, edge_feature, edge_type, label):

        '''
        update the ERG
        :param label:
        :param edge_type:
        :param cls_h: hidden state of "[CLS]" (sentence)
        :param e1_id: the id of entity 1
        :param e2_id:the id of entity 2
        :param graph: ERG adjacency list
        :param edge_feature: features of edges
        '''

        # update ERG adjacency list
        if graph.__contains__(str(e1_id)):
            if int(e2_id) not in graph[str(e1_id)]:
                graph[str(e1_id)].append(int(e2_id))
        else:
            graph[str(e1_id)] = [int(e2_id)]

        if graph.__contains__(str(e2_id)):
            if int(e1_id) not in graph[str(e2_id)]:
                graph[str(e2_id)].append(int(e1_id))
        else:
            graph[str(e2_id)] = [int(e1_id)]

        # update edge_feature
        e1_id_ = e1_id
        e2_id_ = e2_id
        if e1_id > e2_id:
            e2_id_, e1_id_ = e1_id, e2_id
        edge = str(e1_id_) + "-" + str(e2_id_)
        edge_feature[edge] = cls_h.cpu().clone().detach().numpy().tolist()
        edge_type[edge] = label

    def update_type(self, e1_id, e2_id, edge_type, label):
        '''
        update the prediction result to ERG
        :param e1_id:
        :param e2_id:
        :param edge_type:
        :param label:
        :return:
        '''

        e1_id_ = e1_id
        e2_id_ = e2_id
        if e1_id > e2_id:
            e2_id_, e1_id_ = e1_id, e2_id
        edge = str(e1_id_) + "-" + str(e2_id_)
        edge_type[edge] = label



    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(
            1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector



class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        fully-connected layer
        :param x:
        :return:
        '''
        x = self.dropout(x)
        if self.use_activation:
            x = self.relu(x)
        return self.linear(x)


class SSGCN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super(SSGCN, self).__init__()
        self.fc = FCLayer(input_dim, output_dim, dropout_rate)

    def forward(self, hs, hs_k):
        '''
        the Semantic Similarity-based Graph Convolutional Network
        :param hs the representation of sentence s
        :param hs_k the representation of sentence sk in E_adj(s)
        :return: knowledge
        '''
        d_k = hs.size(-1)
        sim = torch.matmul(hs, hs_k.transpose(-2, -1)) / math.sqrt(d_k)
        w_ = F.softmax(sim, dim=-1)
        most_sim = torch.argmax(w_)
        k = torch.matmul(w_, self.fc(hs_k))
        return k, most_sim



def pairwise_distance(embeddings):
    dot_product = torch.matmul(embeddings, torch.transpose(embeddings, 0, 1))
    square_norm = torch.diag(dot_product)
    distances = square_norm.unsqueeze(1) - 2.0 * dot_product + square_norm.unsqueeze(0)
    return distances


def positive_mask(label):
    indices_not_equal = ~torch.eye(label.size()[0]).cuda().bool()
    labels_equal = torch.eq(label.unsqueeze(0), label.unsqueeze(1)).cuda()
    mask = indices_not_equal & labels_equal
    return mask


def negative_mask(label):
    mask = ~torch.eq(label.unsqueeze(0), label.unsqueeze(1)).cuda()
    return mask


def triplet_loss(embeddings, label):
    '''
    triplet loss
    '''
    dot_product = torch.matmul(embeddings, torch.transpose(embeddings, 0, 1))
    mask_positive = positive_mask(label).cuda().float()
    mask_negative = negative_mask(label).cuda().float()
    positive_dist = torch.mul(mask_positive, dot_product)
    negative_dist = torch.mul(mask_negative, dot_product)
    sig = nn.Sigmoid()

    hardest_positive_dist = torch.log(sig(torch.max(positive_dist, 1).values))
    hardest_negative_dist = torch.sum(torch.log(sig(negative_dist)), 1)
    tri_loss = -hardest_positive_dist - hardest_negative_dist
    tri_loss = torch.mean(tri_loss)
    return tri_loss


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        fully-connected layer
        :param x:
        :return:
        '''
        x = self.dropout(x)
        if self.use_activation:
            x = self.relu(x)
        return self.linear(x)


class SSGCN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super(SSGCN, self).__init__()
        self.fc = FCLayer(input_dim, output_dim, dropout_rate)

    def forward(self, hs, hs_k):
        '''
        the Semantic Similarity-based Graph Convolutional Network
        :param hs the representation of sentence s
        :param hs_k the representation of sentence sk in E_adj(s)
        :return: knowledge
        '''
        d_k = hs.size(-1)
        sim = torch.matmul(hs, hs_k.transpose(-2, -1)) / math.sqrt(d_k)
        w_ = F.softmax(sim, dim=-1)
        most_sim = torch.argmax(w_)
        k = torch.matmul(w_, self.fc(hs_k))
        return k, most_sim

