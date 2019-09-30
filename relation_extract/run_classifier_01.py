import tensorflow as tf
import os
import collections
from sklearn.metrics import f1_score,accuracy_score,recall_score,classification_report

from basic_models import modeling,tokenization,optimization


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, e1_pos, e2_pos, label=None):
        self.guid = guid
        self.text_a = text_a
        self.e1_pos = e1_pos
        self.e2_pos = e2_pos
        self.label = label


class BasicProcessor:
    def __init__(self, train_file=None, eval_file=None, predict_file=None):
        self.train_file = train_file
        self.eval_file = eval_file
        self.predict_file = predict_file
        self.get_train_examples()
        self.get_eval_examples()
        self.get_predict_examples()

    def get_train_examples(self):
        tf.logging.info("Process train file:{}".format(self.train_file))
        self.train_examples = []
        self.labels = set()
        if self.train_file:
            if tf.gfile.Exists(self.train_file):
                lines = self._read_lines(self.train_file)
                for (i, line) in enumerate(lines):
                    line = line.split('\t')
                    assert len(line) == 2
                    guid = "train-%d" % (i)
                    text_a = line[0]
                    relations = line[1].split()
                    e1_pos = (relations[0], relations[1])
                    e2_pos = (relations[2], relations[3])
                    label = relations[4]
                    self.train_examples.append(
                        InputExample(guid=guid,
                                     text_a=text_a,
                                     e1_pos=e1_pos,
                                     e2_pos=e2_pos,
                                     label=label))
                    self.labels.add(label)
        self.labels = sorted(list(self.labels))

    def get_eval_examples(self):
        tf.logging.info("Process eval file:{}".format(self.eval_file))
        self.eval_examples = []
        if self.eval_file and tf.gfile.Exists(self.eval_file):
            lines = self._read_lines(self.eval_file)
            for (i, line) in enumerate(lines):
                line = line.split('\t')
                assert len(line) == 2
                guid = "eval-{}".format(i)
                text_a = line[0]
                relations = line[1].split()
                e1_pos = (relations[0], relations[1])
                e2_pos = (relations[2], relations[3])
                label = relations[4]
                self.eval_examples.append(
                    InputExample(guid=guid,
                                 text_a=text_a,
                                 e1_pos=e1_pos,
                                 e2_pos=e2_pos,
                                 label=label))

    def get_predict_examples(self):
        self.predict_examples = []
        if self.predict_file and tf.gfile.Exists(self.predict_file):
            lines = self._read_lines(self.predict_file)
            for (i, line) in enumerate(lines):
                line = line.split('\t')
                assert len(line) == 2
                guid = "predict-{}".format(i)
                text_a = line[0]
                relations = line[1].split()
                e1_pos = (relations[0], relations[1])
                e2_pos = (relations[2], relations[3])
                label = relations[4]
                self.predict_examples.append(
                    InputExample(guid=guid,
                                 text_a=text_a,
                                 e1_pos=e1_pos,
                                 e2_pos=e2_pos,
                                 label=label))

    @classmethod
    def _read_lines(cls, input_file):
        with tf.gfile.Open(input_file, "r") as f:
            lines = set()  # 去重
            for line in f:
                line = line.strip()
                if line:
                    line = line.split('\t')
                    text_a = line[0]
                    relations = line[1].split()
                    num_relations = int(len(relations) / 5)
                    assert len(relations) % 5 == 0
                    for j in range(num_relations):
                        r = relations[5 * j:5 * j + 5]
                        new_line = text_a + '\t' + ' '.join(r)
                        lines.add(new_line)
            tf.logging.info('Filter lines:{}'.format(len(lines)))
            return lines


def create_model(bert_config,
                 is_training,
                 input_ids,
                 input_mask,
                 segment_ids,
                 e1_mas,
                 e2_mas,
                 labels,
                 num_labels,
                 use_one_hot_embeddings,
                 keep_prob=0.9):

    model = modeling.BertModel(config=bert_config,
                               is_training=is_training,
                               input_ids=input_ids,
                               input_mask=input_mask,
                               token_type_ids=segment_ids,
                               use_one_hot_embeddings=use_one_hot_embeddings)
    cls_layer = model.get_pooled_output()
    seq_layer = model.get_sequence_output()
    seq_length = seq_layer.shape[1].value
    hidden_size = seq_layer.shape[2].value

    if is_training:
        cls_layer = tf.nn.dropout(cls_layer, keep_prob=keep_prob)

    def get_entity_layer(mask):
        e_mas = tf.to_float(tf.reshape(mask, [-1, seq_length, 1]))
        e = tf.multiply(seq_layer, e_mas)
        e = tf.reduce_sum(e, axis=-2) / tf.maximum(
            1.0, tf.reduce_sum(tf.to_float(e_mas), axis=-2))
        e = tf.reshape(e, [-1, hidden_size])
        if is_training:
            e = tf.nn.dropout(e, keep_prob=keep_prob)
        return e

    e1 = get_entity_layer(e1_mas)
    e2 = get_entity_layer(e2_mas)

    cls_layer = tf.layers.dense(cls_layer,
                                hidden_size,
                                activation=tf.nn.tanh,
                                use_bias=True)
    e1 = tf.layers.dense(e1, hidden_size, activation=tf.nn.tanh, use_bias=True)
    e2 = tf.layers.dense(e2, hidden_size, activation=tf.nn.tanh, use_bias=True)

    output_layer = tf.concat([cls_layer, e1, e2], axis=-1)

    if is_training:
        output_layer = tf.nn.dropout(output_layer, keep_prob=keep_prob)

    logits = tf.layers.dense(output_layer, num_labels, use_bias=True)

    with tf.variable_scope('loss'):
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config,
                     num_labels,
                     init_checkpoint,
                     learning_rate,
                     num_train_steps,
                     num_warmup_steps,
                     use_tpu=False,
                     use_one_hot_embeddings=False,
                     keep_prob=0.9):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" %
                            (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        e1_mas = features['e1_mas']
        e2_mas = features['e2_mas']
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"],
                                      dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits,
         probabilities) = create_model(bert_config, is_training, input_ids,
                                       input_mask, segment_ids, e1_mas, e2_mas,
                                       label_ids, num_labels,
                                       use_one_hot_embeddings, keep_prob)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(
                 tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint,
                                                  assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(total_loss, learning_rate,
                                                     num_train_steps,
                                                     num_warmup_steps, use_tpu)

            output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                     loss=total_loss,
                                                     train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits,
                          is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(labels=label_ids,
                                               predictions=predictions,
                                               weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss,
                                       weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

                #eval_metrics = (metric_fn, [
                per_example_loss, label_ids, logits, is_real_example

            #])
            eval_metrics = metric_fn(per_example_loss, label_ids, logits,
                                     is_real_example)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, loss=total_loss, eval_metric_ops=eval_metrics)
        else:
            predict_dict={
                'true_labels':label_ids,
                'predict_labels':tf.argmax(probabilities,-1)
            }
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, predictions=predict_dict)
        return output_spec

    return model_fn


class InputFeatures:
    def __init__(self, input_ids, input_mask, segment_ids, e1_mas, e2_mas,
                 label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.e1_mas = e1_mas
        self.e2_mas = e2_mas
        self.label_id = label_id


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    tokens_a = example.text_a.split()
    e1_pos_start = int(example.e1_pos[0])
    e1_pos_end = int(example.e1_pos[1])
    e2_pos_start = int(example.e2_pos[0])
    e2_pos_end = int(example.e2_pos[1])
    if len(tokens_a) > max_seq_length - 6:
        end_max=max(e1_pos_end,e2_pos_end)
        if end_max>max_seq_length+6:
            tokens_a=tokens_a[len(tokens_a)-max_seq_length+6:]
        else:
            tf.logging.warning("Too long:{}".format(tokens_a))
            tokens_a = tokens_a[0:max_seq_length - 6]
        tf.logging.warning("Too long:{}".format(tokens_a))
    tokens = []
    segment_ids = []
    e1_mas = []
    e2_mas = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    e1_mas.append(0)
    e2_mas.append(0)
    for i, token in enumerate(tokens_a):
        if i == e1_pos_start:
            tokens.append('$')
            segment_ids.append(0)
            e1_mas.append(0)
            e2_mas.append(0)
        elif i == e2_pos_start:
            tokens.append('#')
            segment_ids.append(0)
            e1_mas.append(0)
            e2_mas.append(0)

        tokens.append(token)
        segment_ids.append(0)
        if i >= e1_pos_start and i < e1_pos_end:
            e1_mas.append(1)
        else:
            e1_mas.append(0)
        if i >= e2_pos_start and i < e2_pos_end:
            e2_mas.append(1)
        else:
            e2_mas.append(0)

        if i + 1 == e1_pos_end:
            tokens.append('$')
            segment_ids.append(0)
            e1_mas.append(0)
            e2_mas.append(0)
        elif i + 1 == e2_pos_end:
            tokens.append('#')
            segment_ids.append(0)
            e1_mas.append(0)
            e2_mas.append(0)
    #print("".join(tokens_a[e1_pos_start:e1_pos_end]))
    #print("".join([i for i,j in zip(tokens,e1_mas) if j==1]))
    #print("".join(tokens_a[e2_pos_start:e2_pos_end]))
    #print("".join([i for i,j in zip(tokens,e2_mas) if j==1]))

    tokens.append("[SEP]")
    segment_ids.append(0)
    e1_mas.append(0)
    e2_mas.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        e1_mas.append(0)
        e2_mas.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(e1_mas) == max_seq_length
    assert len(e2_mas) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info(
            "tokens: %s" %
            " ".join([tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x)
                                                    for x in input_ids]))
        tf.logging.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" %
                        " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
        tf.logging.info("e1 mask:{}".format(e1_mas))
        tf.logging.info("e2 mask:{}".format(e2_mas))

    feature = InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id,
                            e1_mas=e1_mas,
                            e2_mas=e2_mas)
    return feature


def file_based_convert_examples_to_features(examples, label_list,
                                            max_seq_length, tokenizer,
                                            output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" %
                            (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(
                value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["e1_mas"] = create_int_feature(feature.e1_mas)
        features["e2_mas"] = create_int_feature(feature.e2_mas)

        tf_example = tf.train.Example(features=tf.train.Features(
            feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                batch_size, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "e1_mas": tf.FixedLenFeature([seq_length], tf.int64),
        "e2_mas": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn():
        """The actual input function."""

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def run(config=None):
    tf.logging.set_verbosity(tf.logging.INFO)
    if config is None:
        tf.logging.info("Use default config")
        import basic_config
        config = basic_config.Config()

    if not config.do_train and not config.do_eval and not config.do_predict:
        raise ValueError(
            "At least one of 'do_train ,do_eval or do_predict' must be True")

    bert_config = modeling.BertConfig.from_json_file(config.bert_config_file)

    if config.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (config.max_seq_length, bert_config.max_position_embeddings))
    tf.gfile.MakeDirs(config.output_dir)

    processors = {"basic": BasicProcessor}

    task_name = config.task_name.lower()
    processor = processors[task_name](config.train_file, config.eval_file,
                                      config.predict_file)

    label_list = processor.labels
    tokenizer = tokenization.FullTokenizer(config.vocab_file)

    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        model_dir=config.output_dir,
        log_step_count_steps=config.log_steps,
        session_config=session_config,
    )
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_device_id

    num_train_steps = None
    num_warmup_steps = None
    if config.do_train:
        num_train_steps = len(
            processor.train_examples
        ) / config.train_batch_size * config.num_train_epochs
        num_warmup_steps = int(num_train_steps * config.warmup_proportion)

    init_checkpoint = config.bert_init_checkpoint
    model_fn = model_fn_builder(bert_config,
                                len(label_list),
                                init_checkpoint,
                                config.learning_rate,
                                num_train_steps,
                                num_warmup_steps,
                                False,
                                False,
                                keep_prob=config.keep_prob)
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=config.output_dir,
                                       config=run_config)

    if config.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(processor.train_examples))
        tf.logging.info("  Batch size = %d", config.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_file = os.path.join(config.output_dir, 'train.tfrecord')
        file_based_convert_examples_to_features(processor.train_examples,
                                                label_list,
                                                config.max_seq_length,
                                                tokenizer, train_file)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=config.max_seq_length,
            is_training=True,
            batch_size=config.train_batch_size,
            drop_remainder=False)
        estimator.train(input_fn=train_input_fn, steps=num_train_steps)
    if config.do_eval:
        tf.logging.info("***** Running eval *****")
        tf.logging.info("  Num examples = %d", len(processor.eval_examples))
        tf.logging.info("  Batch size = %d", config.eval_batch_size)
        eval_file = os.path.join(config.output_dir, 'eval.tfrecord')
        file_based_convert_examples_to_features(processor.eval_examples,
                                                label_list,
                                                config.max_seq_length,
                                                tokenizer, eval_file)
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=config.max_seq_length,
            is_training=False,
            batch_size=config.eval_batch_size,
            drop_remainder=False)

        result = estimator.evaluate(input_fn=eval_input_fn)

        # output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        # with tf.gfile.GFile(output_eval_file, "w") as writer:
        tf.logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))
            # writer.write("%s = %s\n" % (key, str(result[key])))
    if config.do_predict:
        tf.logging.info("***** Running predict *****")
        tf.logging.info("  Num examples = %d", len(processor.eval_examples))
        tf.logging.info("  Batch size = %d", config.eval_batch_size)
        eval_file = os.path.join(config.output_dir, 'eval.tfrecord')
        file_based_convert_examples_to_features(processor.eval_examples,
                                                label_list,
                                                config.max_seq_length,
                                                tokenizer, eval_file)
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=config.max_seq_length,
            is_training=False,
            batch_size=config.eval_batch_size,
            drop_remainder=False)

        result = estimator.predict(input_fn=eval_input_fn)

        tf.logging.info("***** predict results *****")
        true_id=[]
        predict_id=[]
        for i in result:
            true_id.append(i['true_labels'])
            predict_id.append(i['predict_labels'])
        tf.logging.info('\n'+classification_report(y_true=true_id,y_pred=predict_id))

if __name__ == "__main__":
    run()