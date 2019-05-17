
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from model import Model
# from gevent.wsgi import WSGIServer
import logging

# app = Flask(__name__)
# CORS(app)

vocab_file = 'data/dl-data/couplet/vocabs'
# vocab_file = 'data\dl-data\AutomaticSummarization/vocab_dict.txt'
# model_dir = 'data/dl-data/models/tf-lib/output_couplet_prod'
model_dir = 'data/dl-data/models/tf-lib/output_couplet'

m = Model(
        None, None, None, None, vocab_file,
        num_units=512, layers=4, dropout=0.2,
        batch_size=32, learning_rate=0.0001,
        output_dir=model_dir,
        restore_model=True, init_train=False, init_infer=True)


# @app.route('/chat/couplet/<in_str>')
def chat_couplet(in_str):
    if len(in_str) == 0 or len(in_str) > 50:
        output = u'您的输入太长了'
    else:
        output = m.infer(' '.join(in_str))
        output = ''.join(output.split(' '))
    print('上联：%s；下联：%s' % (in_str, output))
    # return jsonify({'output': output})

chat_couplet('帅气天籁')
while True:
    sentence1 = input('sentence1: ')
    chat_couplet(sentence1)

# http_server = WSGIServer(('', 5000), app)
# http_server.serve_forever()
