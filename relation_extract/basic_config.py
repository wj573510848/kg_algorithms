import os

class Config:
    def __init__(self):

        self.do_train=True
        self.do_eval=True
        self.do_predict=True

        # bert-RoBERTa-wwm
        self.bert_model_dir = '/home/wangjian0110/myWork/chinese_roberta_wwm_ext_L-12_H-768_A-12'
        self.bert_config_file = os.path.join(self.bert_model_dir,
                                             'bert_config.json')
        self.bert_init_checkpoint = os.path.join(self.bert_model_dir,
                                                 'bert_model.ckpt')
        self.vocab_file = os.path.join(self.bert_model_dir, 'vocab.txt')


        # files
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.train_file = os.path.join(self.cur_dir, 'train_data/train.txt')
        self.eval_file = os.path.join(self.cur_dir, 'data/dev.txt')
        self.predict_file = None

        # bert pre-trained model

        self.output_dir = './out_1'
        self.task_name = 'basic'
        self.max_seq_length = 256
        self.keep_prob = 0.9
        self.log_steps = 200
        self.gpu_device_id = '2'

        self.train_batch_size = 64
        self.eval_batch_size = 64
        self.num_train_epochs = 4
        self.warmup_proportion = 0.1
        self.learning_rate = 5e-5