

import tensorflow as tf
import torch
from tensorflow.contrib import rnn
from tensorflow.python.layers import core as layers_core


# def getLayeredCell(layer_size, num_units, input_keep_prob,
#         output_keep_prob=1.0):
#     return rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.BasicLSTMCell(num_units),
#         input_keep_prob, output_keep_prob) for i in range(layer_size)])

def getLayeredCell(layer_size, num_units, input_keep_prob,
        output_keep_prob=1.0):
    return rnn.MultiRNNCell([rnn.DropoutWrapper(rnn.GRUCell(num_units),
        input_keep_prob, output_keep_prob) for i in range(layer_size)])

# 使用一个双胞胎网络来进行encoder，返回双胞胎网络的最后一次循环的输出，和双胞胎网络最后的的状态
# def bi_encoder(embed_input, in_seq_len, num_units, layer_size, input_keep_prob):
#     # encode input into a vector
#     bi_layer_size = int(layer_size / 2)
#     encode_cell_fw = getLayeredCell(bi_layer_size, num_units, input_keep_prob)
#     encode_cell_bw = getLayeredCell(bi_layer_size, num_units, input_keep_prob)
#     bi_encoder_output, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
#             cell_fw = encode_cell_fw,
#             cell_bw = encode_cell_bw,
#             inputs = embed_input,
#             sequence_length = in_seq_len,
#             dtype = embed_input.dtype,
#             time_major = False)
#
#     # concat encode output and state
#     encoder_output = tf.concat(bi_encoder_output, -1)
#     encoder_state = []
#     for layer_id in range(bi_layer_size):
#         encoder_state.append(bi_encoder_state[0][layer_id])
#         encoder_state.append(bi_encoder_state[1][layer_id])
#     encoder_state = tuple(encoder_state)
#     return encoder_output, encoder_state

# def bi_encoder(embed_input, in_seq_len, num_units, layer_size, input_keep_prob,max_len):
#     encoder_output, encoder_state=BertSim(max_seq_length=max_len)
#     return encoder_output, encoder_state
def attention_decoder_cell(encoder_output, in_seq_len, num_units, layer_size,
        input_keep_prob):
    attention_mechanim = tf.contrib.seq2seq.BahdanauAttention(num_units,
            encoder_output, in_seq_len, normalize = True)
    # attention_mechanim = tf.contrib.seq2seq.LuongAttention(num_units,
    #         encoder_output, in_seq_len, scale = True)
    cell = getLayeredCell(layer_size, num_units, input_keep_prob)
    cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanim,
            attention_layer_size=num_units)
    return cell

# decoder的映射，将decoder的输出利用一个全连接网络再映射
def decoder_projection(output, output_size):
    return tf.layers.dense(output, output_size, activation=None,
            use_bias=False, name='output_mlp')



# 训练时候decoder 需要
def train_decoder(encoder_output, in_seq_len, target_seq, target_seq_len,
        encoder_state, num_units, layers, embedding, output_size,
        input_keep_prob, projection_layer):
    decoder_cell = attention_decoder_cell(encoder_output, in_seq_len, num_units,
            layers, input_keep_prob)
    batch_size = tf.shape(in_seq_len)[0]
    init_state = decoder_cell.zero_state(batch_size, tf.float32).clone(
            cell_state=encoder_state)
    helper = tf.contrib.seq2seq.TrainingHelper(
                target_seq, target_seq_len, time_major=False)
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
            init_state, output_layer=projection_layer)
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
            maximum_iterations=100)
    return outputs.rnn_output

# 预测时候的decoder，需要一个注意力层，然后再加一个一个解码器
def infer_decoder(encoder_output, in_seq_len, encoder_state, num_units, layers,
        embedding, output_size, input_keep_prob, projection_layer):
    decoder_cell = attention_decoder_cell(encoder_output, in_seq_len, num_units,
            layers, input_keep_prob)

    batch_size = tf.shape(in_seq_len)[0]
    init_state = decoder_cell.zero_state(batch_size, tf.float32).clone(
            cell_state=encoder_state)

    # TODO: start tokens and end tokens are hard code
    """
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding, tf.fill([batch_size], 0), 1)
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
            init_state, output_layer=projection_layer)
    """

    decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        cell=decoder_cell,
        embedding=embedding,
        start_tokens=tf.fill([batch_size], 0),
        end_token=1,
        initial_state=init_state,
        beam_width=10,
        output_layer=projection_layer,
        length_penalty_weight=1.0)

    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
            maximum_iterations=100)
    return outputs.sample_id


# 参数，输入序列，输入序列长度，目标序列，目标序列长度，词汇表大小，隐藏层神经元数量，隐藏层层数，dropout比例
def seq2seq(in_seq, in_seq_len, in_seq_mask,in_seq_segment,
            target_seq, target_seq_len, target_seq_mask,target_seq_segment,
            vocab_size,
        num_units, layers, dropout,max_len,bert_model,init_train):
    in_shape = tf.shape(in_seq)
    batch_size = in_shape[0]
    #
    if target_seq != None:
        input_keep_prob = 1 - dropout
    else:
        input_keep_prob = 1

    # 输出层的定义
    projection_layer=layers_core.Dense(vocab_size, use_bias=False)

    # embedding input and target sequence
    # with tf.device('/gpu:0'):
    #     embedding = tf.get_variable(
    #             name = 'embedding',
    #             shape = [vocab_size, num_units])
    # embed_input=tf.nn.embedding_lookup(embedding,in_seq,name='embed_input')
    # encode and decode
    encoder_output, encoder_state =bert_model.get_encoder(in_seq,in_seq_mask,in_seq_segment,init_train,False)

    # encoder_output, encoder_state =bi_encoder(embed_input, in_seq_len, num_units, layers, input_keep_prob)
    # 对encode部分进行decode，参数为encode最后一层的输出，输入序列的大小，隐藏层神经元的数量，层数
    decoder_cell = attention_decoder_cell(encoder_output, in_seq_len, num_units,
            layers, input_keep_prob)
    batch_size = tf.shape(in_seq_len)[0]
    init_state = decoder_cell.zero_state(batch_size, tf.float32)
    # init_state = decoder_cell.zero_state(batch_size, tf.float32).clone(
    #         cell_state=encoder_state)

    if target_seq != None:
        embed_target = bert_model.get_Bert_embedding(target_seq, target_seq_mask, target_seq_segment, False,init_train, False)
        # embed_target = tf.nn.embedding_lookup(embedding, target_seq,
        #         name='embed_target')
        helper = tf.contrib.seq2seq.TrainingHelper(
                    embed_target, target_seq_len, time_major=False)
    else:
        # TODO: start tokens and end tokens are hard code
        embedding=bert_model.get_Bert_embedding(in_seq,in_seq_mask,in_seq_segment, True,init_train, False)
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding, tf.fill([batch_size], 0), 1)
    # decode解码
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
            init_state, output_layer=projection_layer)
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
            maximum_iterations=100)
    if target_seq != None:
        return outputs.rnn_output
    else:
        return outputs.sample_id

# decode的损失函数定义
def seq_loss(output, target, seq_len):
    # 获取target第一列以后的所有数据
    print('output:shape:',output.shape[1])
    # target = target[:,output.shape[1]:]
    target=target[:,1:]
    # target=tf.reshape(target, tf.shape(16,output.shape[1]))
    # output=output[:,:output.shape[1],:]
    print(target)
    # _cost = tf.nn.ctc_loss(labels=target,inputs=output,sequence_length=seq_len,time_major=False)
    # cost=tf.nn.softmax_cross_entropy_with_logits_v2
    _cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,
            labels=target)
    batch_size = tf.shape(target)[0]
    loss_mask = tf.sequence_mask(seq_len, tf.shape(output)[1])
    cost = _cost * tf.to_float(loss_mask)
    return tf.reduce_sum(cost) / tf.to_float(batch_size),_cost

