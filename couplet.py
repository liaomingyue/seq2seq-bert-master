
from model import Model

m = Model(
        'data\dl-data\AutomaticSummarization/test/in.txt',
        'data\dl-data\AutomaticSummarization/test/out.txt',
        'data\dl-data\AutomaticSummarization/train/in.txt',
        'data\dl-data\AutomaticSummarization/train/out.txt',
        'bertmodel\chinese_L-12_H-768_A-12/vocab.txt',
        num_units=128, layers=4, dropout=0.2,
        batch_size=16, learning_rate=0.0001,max_len=32,
        output_dir='data/dl-data/models/tf-lib/output_couplet',
        restore_model=False)

m.train(5000000)
