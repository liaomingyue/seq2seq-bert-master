import tensorflow as tf
from queue import Queue
from threading import Thread
import bertmodel.tokenization as tokenization
import random
import bertmodel.args as args
# 定义一个最大的序列，其他用<s>来填充
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
def padding_seq(seq):
    results = []
    max_len = 0
    for s in seq:
        if max_len < len(s):
            max_len = len(s)
    for i in range(0, len(seq)):
        l = max_len - len(seq[i])
        results.append(seq[i] + [0 for j in range(l)])
    return results
# print(padding_seq([['我','不是','你哥'],['我','是','你哥','你姐']]))

# 将words句子编码成其字典的小标
def encode_text(words, vocab_indices):
    return [vocab_indices[word] for word in words if word in vocab_indices]

# print(encode_text(['我','想','回家','玩'],{'我':1,'想':2,'回家':3}))

# 将字典下标编码成数字
def decode_text(labels, vocabs, end_token = '[SEP]'):
    results = []
    for idx in labels:
        word = vocabs[idx]
        if word == end_token:
            return ' '.join(results)
        results.append(word)
    return ' '.join(results)

# 获取字典列表
def read_vocab(vocab_file):
     f = open(vocab_file, 'rb')
     vocabs = [line.decode('utf8')[:-1] for line in f]
     f.close()
     return vocabs
# print(read_vocab('data\dl-data\couplet/vocabs'))
# class FeatureGet():
#     def __init__(self):
class SeqReader():
    def __init__(self, input_file, target_file, vocab_file, batch_size,
            queue_size = 2048, worker_size = 8, end_token = '[SEP]',
            padding = True, max_len = 32):
        self.input_file = input_file
        self.target_file = target_file
        self.end_token = end_token
        self.batch_size = batch_size
        self.tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
        self.padding = padding
        self.max_len = max_len
        # self.vocabs = read_vocab(vocab_file) + [end_token]
        self.vocabs = read_vocab(vocab_file)
        self.vocab_indices = dict((c, i) for i, c in enumerate(self.vocabs))
        self.data_queue = Queue(queue_size)
        self.worker_size = worker_size
        with open(self.input_file,encoding='utf-8') as f:
            for i, l in enumerate(f):
                pass
            f.close()
            self.single_lines = i+1
        self.data_size = int(self.single_lines / batch_size)
        self.data_pos = 0
        self._init_reader()

    # 定义一个来多线程获取数据
    def start(self):
        pass
        # for i in range(self.worker_size):
        #     t = Thread(target=self._init_reader())
        #     t.daemon = True
        #     t.start()
        # return
    '''
        for i in range(self.worker_size):
            t = Thread(target=self._init_reader())
            t.daemon = True
            t.start()
    '''
    # 真正实现句子编码方式
    def convert_single_example(self, ex_index, example, label_list, max_seq_length, tokenizer):
        """Converts a single `InputExample` into a single `InputFeatures`."""
        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            pass
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("guid: %s" % (example.guid))
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids)
        return feature
    #返回BERT  句子编码方式
    def get_input_ids(self,input_text):
        label_list = ['1']
        train_examples = []
        for i, text in enumerate(input_text):
            train_examples.append(InputExample('tarin-' + str(i), text))
        features_list = self.file_based_convert_examples_to_features(train_examples, label_list, self.max_len,
                                                                     self.tokenizer,
                                                                     output_file='')
        input_ids = []
        input_mask = []
        segment_ids = []
        for features in features_list:
            input_ids.append(features[0])
            input_mask.append(features[1])
            segment_ids.append(features[2])
        # input_ids = tf.convert_to_tensor(input_ids)
        # input_mask = tf.convert_to_tensor(input_mask)
        # segment_ids = tf.convert_to_tensor(segment_ids)
        return [input_ids,input_mask,segment_ids]
    #
    def file_based_convert_examples_to_features(self, examples, label_list, max_seq_length, tokenizer, output_file):
        """Convert a set of `InputExample`s to a TFRecord file."""
        features_list=[]
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
            feature = self.convert_single_example(ex_index, example, label_list,
                                                  max_seq_length, tokenizer)
            features_list.append([feature.input_ids,feature.input_mask,feature.segment_ids])
        return features_list

    def read_single_data(self):
        if self.data_pos >= len(self.data):
            random.shuffle(self.data)
            self.data_pos = 0
        result = self.data[self.data_pos]
        self.data_pos += 1
        return result

    # 返回一个批次的数据
    def read(self):
        while True:
            batch = {'in_seq_ids': [],
                    'in_seq_mask': [],
                    'in_seq_segment': [],
                    'in_seq_len': [],
                    'target_seq_ids': [],
                    'target_seq_mask': [],
                    'target_seq_segment': [],
                    'target_seq_len': []}
            for i in range(0, self.batch_size):
                item = self.read_single_data()
                batch['in_seq_ids'].append(item['in_seq_ids'])
                batch['in_seq_len'].append(item['in_seq_len'])
                batch['in_seq_mask'].append(item['in_seq_mask'])
                batch['in_seq_segment'].append(item['in_seq_segment'])
                batch['target_seq_ids'].append(item['target_seq_ids'])
                batch['target_seq_len'].append(item['target_seq_len'])
                batch['target_seq_mask'].append(item['target_seq_mask'])
                batch['target_seq_segment'].append(item['target_seq_segment'])
                if(len(item['target_seq_ids'])<31):
                    while(True):
                        print("caonima")
            if self.padding:
                batch['in_seq_ids'] = padding_seq(batch['in_seq_ids'])
                batch['target_seq_ids'] = padding_seq(batch['target_seq_ids'])
            yield batch
    def get_seq_len(self,seq):
        index=0
        for word in seq:
            if(word==0):
                return seq[:index]
            index+=1
        return seq
    # 定义一个全部数据的data
    def _init_reader(self):
        self.data = []
        input_f = open(self.input_file, 'rb')
        target_f = open(self.target_file, 'rb')
        count=1
        for input_line in input_f:
            count+=1
            input_line=self.get_input_ids([input_line.decode('utf-8')[:-1]])
            # input_line = input_line.decode('utf-8')[:-1]
            # target_line = target_f.readline().decode('utf-8')[:-1]
            target_line=self.get_input_ids([target_f.readline().decode('utf-8')[:-1]])
            # input_words = [x for x in input_line.split(' ') if x != '']
            # if len(input_words) >= self.max_len:
            #     input_words = input_words[:self.max_len-1]
            # input_words.append(self.end_token)
            # target_words = [x for x in target_line.split(' ') if x != '']
            # if len(target_words) >= self.max_len:
            #     target_words = target_words[:self.max_len-1]
            # target_words = ['<s>',] + target_words
            # target_words.append(self.end_token)
            # in_seq = encode_text(input_words, self.vocab_indices)
            # target_seq = encode_text(target_words, self.vocab_indices)
            in_seq_ids=input_line[0]
            in_seq_mask=input_line[1]
            in_seq_segment = input_line[2]
            target_seq_ids=target_line[0]
            target_seq_mask=target_line[1]
            target_seq_segment = target_line[2]
            # print(self.get_seq_len(in_seq_ids[0]),self.get_seq_len(target_seq_ids[0])-1)
            self.data.append({
                'in_seq_ids': in_seq_ids[0],
                'in_seq_mask': in_seq_mask[0],
                'in_seq_segment': in_seq_segment[0],
                'in_seq_len': len(in_seq_ids[0]),
                'target_seq_ids': target_seq_ids[0],
                'target_seq_mask': target_seq_mask[0],
                'target_seq_segment': target_seq_segment[0],
                'target_seq_len': len(target_seq_ids[0])-1
            })
            print(count)
        input_f.close()
        target_f.close()
        self.data_pos = len(self.data)
        # for i in self.data:
        #     print(i)
