import tensorflow as tf
import os
import collections
from sklearn.metrics import f1_score,accuracy_score,recall_score,classification_report

from basic_models import cnn,tokenization


def get_train_op(init_lr,loss,num_train_steps,num_warmup_steps):
    global_step=tf.train.get_or_create_global_step()
    #learning_rate = tf.train.exponential_decay(learning_rate, global_step, self.decay_steps, self.decay_rate, staircase=True)
    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=1.0,
      cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if num_warmup_steps:
      global_steps_int = tf.cast(global_step, tf.int32)
      warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

      global_steps_float = tf.cast(global_steps_int, tf.float32)
      warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

      warmup_percent_done = global_steps_float / warmup_steps_float
      warmup_learning_rate = init_lr * warmup_percent_done

      is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
      learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
    tf.summary.scalar('learning rate',learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #ADD 2018.06.01
    with tf.control_dependencies(update_ops):  #ADD 2018.06.01
        train_op = optimizer.apply_gradients(zip(gradients, variables))
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op

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


def create_model(input_ids,label_ids,e1_pos,e2_pos,vocab_size,num_labels,is_training,config):

    logits=cnn.text_cnn(
        input_ids=input_ids,
        e1_pos=e1_pos,
        e2_pos=e2_pos,
        max_distance=config.max_distance,
        keep_prob=config.keep_prob,
        filter_sizes=config.filter_sizes,
        num_filters=config.num_filters,
        pos_embedding_size=config.pos_embedding_size,
        sequence_length=config.max_seq_length,
        num_classes=num_labels,
        is_training=is_training,
        hidden_size=config.char_embedding_size,
        vocab_size=vocab_size,
        l2_reg_lambda=0.0)
    
    per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label_ids)
    loss = tf.reduce_mean(per_example_loss) 
    probabilities = tf.nn.softmax(logits, axis=-1)
    return loss,per_example_loss,logits,probabilities

def model_fn_builder(vocab_size,num_labels,config,num_train_steps,num_warmup_steps):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" %
                            (name, features[name].shape))

        input_ids = features["input_ids"]
        e1_pos = features['e1_pos']
        e2_pos = features['e2_pos']
        label_ids = features["label_ids"]
        
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits,
         probabilities) = create_model(input_ids,label_ids,e1_pos,e2_pos,vocab_size,num_labels,is_training,config)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = get_train_op(config.learning_rate,total_loss,num_train_steps,num_warmup_steps)

            output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                     loss=total_loss,
                                                     train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits,
                          is_real_example=False):
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

                
            eval_metrics = metric_fn(per_example_loss, label_ids, logits)
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
def get_relative_distance(mapping,lo_start,lo_end,max_distance,max_seq_length):
  res=[max_distance]*max_seq_length
  mask=[0]*max_seq_length
  for i in range(max_seq_length):
    if i<len(mapping):
      val=mapping[i]
      if val<lo_start-max_distance:
        res[i]=max_distance
      elif val<lo_start:
        res[i]=lo_start-val
      elif val<lo_end:
        res[i]=0
      elif val<=lo_end+max_distance:
        res[i]=val-lo_end+max_distance
      else:
        res[i]=2*max_distance
    else:
      res[i]=2*max_distance
  return res
def prepare_extra_data(mapping_a,locs,max_distance,max_seq_length):
  entity1_loc,entity2_loc=locs
  e1_pos=get_relative_distance(mapping_a,entity1_loc[0],entity1_loc[1],max_distance,max_seq_length)
  e2_pos=get_relative_distance(mapping_a,entity2_loc[0],entity2_loc[1],max_distance,max_seq_length)
  return e1_pos,e2_pos

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer,max_distance):
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
        tokens_a = tokens_a[0:max_seq_length - 6]
        tf.logging.warning("Too long:{}".format(tokens_a))

    e1_start=e1_pos_start
    e1_end=e1_pos_end
    e2_start=e2_pos_start
    e2_end=e2_pos_end

    def update_pos(index,e1_start,e1_end,e2_start,e2_end):
        if index<=e1_pos_start:
            e1_start+=1
        if index<e1_pos_end:
            e1_end+=1
        if index<=e2_pos_start:
            e2_start+=1
        if index<e2_pos_end:
            e2_end+=1
        return e1_start,e1_end,e2_start,e2_end

    tokens = []
    segment_ids = []
    e1_mas = []
    e2_mas = []
    tokens.append("[CLS]")
    e1_start,e1_end,e2_start,e2_end=update_pos(0,e1_start,e1_end,e2_start,e2_end)
    segment_ids.append(0)
    e1_mas.append(0)
    e2_mas.append(0)

    for i, token in enumerate(tokens_a):
        if i == e1_pos_start:
            tokens.append('$')
            segment_ids.append(0)
            e1_mas.append(0)
            e2_mas.append(0)
            e1_start,e1_end,e2_start,e2_end=update_pos(i,e1_start,e1_end,e2_start,e2_end)
        elif i == e2_pos_start:
            tokens.append('#')
            segment_ids.append(0)
            e1_mas.append(0)
            e2_mas.append(0)
            e1_start,e1_end,e2_start,e2_end=update_pos(i,e1_start,e1_end,e2_start,e2_end)

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
            e1_start,e1_end,e2_start,e2_end=update_pos(i+1,e1_start,e1_end,e2_start,e2_end)
          
        elif i + 1 == e2_pos_end:
            tokens.append('#')
            segment_ids.append(0)
            e1_mas.append(0)
            e2_mas.append(0)
            e1_start,e1_end,e2_start,e2_end=update_pos(i+1,e1_start,e1_end,e2_start,e2_end)
     
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

    mapping_a=list(range(len(input_ids)))
    e1_pos,e2_pos = prepare_extra_data(mapping_a, [(e1_start,e1_end),(e2_start,e2_end)], max_distance,max_seq_length)
    
    #e1=''
    #for i,j in zip(e1_pos,tokens):
    #    if i==0:
    #        e1+=j
    #print(e1)
    #e2=''
    #for i,j in zip(e2_pos,tokens):
    #    if i==0:
    #        e2+=j
    #print(e2)
    #exit()

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
                            e1_mas=e1_pos,
                            e2_mas=e2_pos)
    return feature


def file_based_convert_examples_to_features(examples, label_list,
                                            max_seq_length, tokenizer,
                                            output_file,max_distance):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" %
                            (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer,max_distance)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(
                value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["e1_pos"] = create_int_feature(feature.e1_mas)
        features["e2_pos"] = create_int_feature(feature.e2_mas)

        tf_example = tf.train.Example(features=tf.train.Features(
            feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                batch_size, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "e1_pos": tf.FixedLenFeature([seq_length], tf.int64),
        "e2_pos": tf.FixedLenFeature([seq_length], tf.int64),
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
        config = basic_config.CnnConfig()

    if not config.do_train and not config.do_eval and not config.do_predict:
        raise ValueError(
            "At least one of 'do_train ,do_eval or do_predict' must be True")

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

    model_fn = model_fn_builder(
        vocab_size=len(tokenizer.vocab),
        num_labels=len(processor.labels),
        config=config,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)
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
                                                tokenizer, train_file,config.max_distance)
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
                                                tokenizer, eval_file,config.max_distance)
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
                                                tokenizer, eval_file,config.max_distance)
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