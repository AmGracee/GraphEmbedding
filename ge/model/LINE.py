# -*- coding:utf-8 -*-
# https://github.com/shenweichen/GraphEmbedding/blob/master/ge/models/line.py


import math
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Embedding,Input,Lambda
from tensorflow.python.keras.models import Model

from ge.utils import preprocess_nxgraph

'''首先输入就是两个顶点的编号，然后分别拿到各自对应的embedding向量，
最后输出内积的结果。 真实label定义为1或者-1，通过模型输出的内积和line_loss就可以优化使用了负采样技巧的目标函数了'''
# 定义损失函数
def line_loss(y_true,y_pred):
    return -K.mean(K.log(K.sigmoid(y_true*y_pred)))
# 定义模型
def create_model(numNodes, embedding_size, order='second'):

    v_i = Input(shape=(1,))
    v_j = Input(shape=(1,))

    first_emb = Embedding(numNodes,embedding_size,name='first_emb')
    second_emb = Embedding(numNodes,embedding_size,name='second_emb')  # 2阶相似度
    context_emb = Embedding(numNodes,embedding_size,name='context_emb')

    v_i_emb = first_emb(v_i) #本身的embedding
    v_j_emb = first_emb(v_j)

    v_i_emb_second = second_emb(v_i)
    v_j_context_emb = context_emb(v_j)

    first = Lambda(lambda x :tf.reduce_sum(
        x[0] * x[1], axis=-1, keepdims=False), name='first_order')([v_i_emb,v_j_emb])
    second = Lambda(lambda x: tf.reduce_sum(
        x[0] * x[1], axis=-1, keep_dims=False), name='second_order')([v_i_emb_second, v_j_context_emb])

    if order == 'first':
        output_list = [first]
    elif order == 'second':
        output_list = [second]
    else:
        output_list = [first,second]

    model = Model(inputs=[v_i,v_j],outputs=output_list )
    return model,{'first': first_emb,'second':second_emb}




class LINE:
    def __init__(self,graph,embedding_size=8,negative_ratio=5,order='second'):
        """
        :param graph:
        :param embedding_size:
        :param negative_ratio:
        :param order: 'first','second','all'
        """
        if order not in ['first','second','all']:
            raise ValueError('mode must be first,second,or all')

        self.graph = graph
        self.idx2node, self.node2idx = preprocess_nxgraph(graph)
        self.use_alias = True

        self.rep_size = embedding_size
        self.order = order

        self._embeddings = {}
        self.negative_ratio = negative_ratio
        self.order = order

        self.node_size = graph.number_of_nodes()
        self.edge_size = graph.number_of_edges()
        self.samples_per_epoch = self.edge_size*(1+negative_ratio)

        self._gen_sampling_table()# 负采样 边采样
        self.reset_model()

    def reset_training_config(self,batch_size,times):
        self.batch_size = batch_size
        self.step_per_epoch = (
            (self.samples_per_epoch-1) // self.batch_size+1)*times

    def reset_model(self, opt='adam'):
        self.model, self.embedding_dict = create_model(
            self.node_size, self.rep_size, self.order)
        self.model.compile(opt, line_loss)
        self.batch_it = self.batch_iter(self.node2idx)

    #顶点负采样和边采样
    #下面的函数功能是创建顶点负采样和边采样需要的采样表。中规中矩，主要就是做一些预处理，然后创建alias算法需要的两个表。
    def _gen_sampling_table(self):
        # create sampling table for vertex
        power = 0.75
        numNodes = self.node_size
        node_degree = np.zeros(numNodes) #节点的出度[0,0,0,0,...]
        node2idx = self.node2idx

        for edge in self.graph.edges():
            node_degree[node2idx[edge[0]]] += self.graph[edge[0]][edge[1]].get('weight',1.0)

        for i in range(numNodes):
            total_sum = sum([math.pow(node_degree[i],power)])
            norm_prob = [float(math.pow(node_degree[i],power)) / total_sum for j in range()]

