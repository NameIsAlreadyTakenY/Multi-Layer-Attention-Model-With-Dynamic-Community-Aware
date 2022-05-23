from .layers import *
# import keras.backend as K

flags = tf.app.flags
FLAGS = flags.FLAGS

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging
        self.vars = {}
        self.placeholders = {}
        self.layers = []
        self.activations = []
        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}
        # Build metrics
        self._loss()
        self._accuracy()
        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class DyMADC(Model):
    def _accuracy(self):
        pass

    def __init__(self, placeholders, num_features, num_features_nonzero, degrees, **kwargs):
        super(DyMADC, self).__init__(**kwargs)
        self.attn_wts_all = []
        self.temporal_attention_layers = []
        self.structural_attention_layers = []
        self.placeholders = placeholders
        if FLAGS.ttl < 0:
            self.num_time_steps = len(placeholders['features'])
        else:
            self.num_time_steps = min(len(placeholders['features']), FLAGS.ttl + 1)  # ttl = 0 => only self.
        self.num_time_steps_train = self.num_time_steps - 1
        self.num_features = num_features
        self.num_features_nonzero = num_features_nonzero
        self.degrees = degrees
        self.num_features = num_features
        self.structural_head_config = map(int, FLAGS.structural_head_config.split(","))
        self.structural_layer_config = map(int, FLAGS.structural_layer_config.split(","))
        self.temporal_head_config = map(int, FLAGS.temporal_head_config.split(","))
        self.temporal_layer_config = map(int, FLAGS.temporal_layer_config.split(","))
        self._build()

    def _build(self):
        proximity_labels = [tf.expand_dims(tf.cast(self.placeholders['node_2'][t], tf.int64), 1) #nodes type cast to int
                            for t in range(0, len(self.placeholders['features']))]  # [B, 1]

        self.proximity_neg_samples = []
        for t in range(len(self.placeholders['features']) - 1 - self.num_time_steps_train,len(self.placeholders['features']) - 1):
            # print(len(self.degrees[t]))
            # print(self.degrees[t].tolist())
            temp=tf.nn.fixed_unigram_candidate_sampler(
                true_classes=proximity_labels[t], #一个int64类型的Tensor,具有shape [batch_size, num_true].目标类
                num_true=1, # int,每个训练示例的目标类数.
                num_sampled=FLAGS.neg_sample_size, # int,随机抽样的类数.
                unique=False, #bool,确定批处理中的所有采样类是否都是唯一的
                range_max=len(self.degrees[t]), #int,可能的类数. 这里用的节点个数
                distortion=0.75, #distortion(失真)用于扭曲unigram概率分布.在添加到内部unigram分布之前,首先将每个权重提升到失真的幂.结果,distortion = 1.0给出常规的unigram采样(由vocab文件定义),并且distortion = 0.0给出均匀分布.
                unigrams=self.degrees[t].tolist())#unigram计数或概率的列表,这里用节点度来作为权重
            self.proximity_neg_samples.append(temp[0])

        self.final_output_embeddings = self.build_net(self.structural_head_config, self.structural_layer_config,
                                                      self.temporal_head_config,
                                                      self.temporal_layer_config,
                                                      self.placeholders['spatial_drop'],
                                                      self.placeholders['temporal_drop'],
                                                      self.placeholders['adjs'])
        self._loss()
        self.init_optimizer() #run

    def build_net(self, attn_head_config, attn_layer_config, temporal_head_config, temporal_layer_config,
                  spatial_drop, temporal_drop, adjs):
        input_dim = self.num_features
        sparse_inputs = True
        attn_layer_config = list(attn_layer_config)
        attn_head_config = list(attn_head_config)
        temporal_head_config = list(temporal_head_config)
        temporal_layer_config = list(temporal_layer_config)

        # 1:添加一层 Structural Attention Layers
        for i in range(0, len(attn_layer_config)):
            if i > 0:
                input_dim = attn_layer_config[i - 1]
                sparse_inputs = False
            self.structural_attention_layers.append(StructuralAttentionLayer(input_dim=input_dim,
                                                                             output_dim=attn_layer_config[i],
                                                                             n_heads=attn_head_config[i],
                                                                             attn_drop=spatial_drop,
                                                                             ffd_drop=spatial_drop,
                                                                             act=tf.nn.elu,
                                                                             sparse_inputs=sparse_inputs,
                                                                             residual=False))
        # 2:添加一层 Temporal Attention Layers 输入128维，heads=16
        input_dim = attn_layer_config[-1]
        for i in range(0, len(temporal_layer_config)):
            if i > 0:
                input_dim = temporal_layer_config[i - 1]
            temporal_layer = TemporalAttentionLayer(input_dim=input_dim, n_heads=temporal_head_config[i],
                                                    attn_drop=temporal_drop, num_time_steps=self.num_time_steps,
                                                    residual=False)
            self.temporal_attention_layers.append(temporal_layer)
        input_list = self.placeholders['features']  # List of t feature matrices. [N x F] N个节点，每个节点F维
        for layer in self.structural_attention_layers:
            attn_outputs = []
            for t in range(0, self.num_time_steps):
                out = layer([input_list[t], adjs[t]]) #进入xxlayers的__call__函数中
                attn_outputs.append(out)  # A list of [1x Ni x F]
            input_list = list(attn_outputs)
        for t in range(0, self.num_time_steps):
            zero_padding = tf.zeros([1, tf.shape(attn_outputs[-1])[1] - tf.shape(attn_outputs[t])[1], attn_layer_config[-1]]) # 1*（最后一层Ni-Nt）*128
            attn_outputs[t] = tf.concat([attn_outputs[t], zero_padding], axis=1) #添加 将所有attn_outputs[t] 都扩展为 1*最后一层Ni*128

        structural_outputs = tf.transpose(tf.concat(attn_outputs, axis=0), [1, 0, 2])  # [N, T, F]
        structural_outputs = tf.reshape(structural_outputs,[-1, self.num_time_steps, attn_layer_config[-1]])  # [N, T, F]

        temporal_inputs = structural_outputs
        # outputs = structural_outputs
        for temporal_layer in self.temporal_attention_layers:
            outputs = temporal_layer(temporal_inputs)  # [N, T, F]
            temporal_inputs = outputs
            self.attn_wts_all.append(temporal_layer.attn_wts_all)
        return outputs

    def _loss(self):
        self.graph_loss = tf.constant(0.0)
        num_time_steps_train = self.num_time_steps_train
        for t in range(self.num_time_steps_train - num_time_steps_train, self.num_time_steps_train):
            output_embeds_t = tf.nn.embedding_lookup(tf.transpose(self.final_output_embeddings, [1, 0, 2]), t)#选取一个张量里面索引对应的元素
            inputs1 = tf.nn.embedding_lookup(output_embeds_t, self.placeholders['node_1'][t])
            inputs2 = tf.nn.embedding_lookup(output_embeds_t, self.placeholders['node_2'][t])
            pos_score = tf.reduce_sum(inputs1 * inputs2, axis=1)

            neg_samples = tf.nn.embedding_lookup(output_embeds_t, self.proximity_neg_samples[t])
            neg_score = (-1.0) * tf.matmul(inputs1, tf.transpose(neg_samples))
            pos_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pos_score), logits=pos_score)#计算交叉熵
            neg_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(neg_score), logits=neg_score)
            self.graph_loss += tf.reduce_mean(pos_ent) + FLAGS.neg_weight * tf.reduce_mean(neg_ent)

        self.reg_loss = tf.constant(0.0)
        if len([v for v in tf.trainable_variables() if "struct_attn" in v.name and "bias" not in v.name]) > 0:
            self.reg_loss += tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
                                       if "struct_attn" in v.name and "bias" not in v.name]) * FLAGS.weight_decay
        self.loss = self.graph_loss + self.reg_loss
        # self.loss = self.graph_loss + self.reg_loss + make_online()

    def init_optimizer(self):
        trainable_params = tf.trainable_variables()
        actual_loss = self.loss
        gradients = tf.gradients(actual_loss, trainable_params)
        # Clip gradients by a given maximum_gradient_norm
        clip_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        # if FLAGS.optimizer == 'Adam':
		# 	self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
		# elif FLAGS.optimizer == 'SGD':
		# 	self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
		# elif FLAGS.optimizer == 'Adade':
		# 	self.optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate)
		# elif FLAGS.optimizer == 'RSMP':
		# 	self.optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate)
		# elif FLAGS.optimizer == 'Momentum':
		# 	self.optimizer == tf.compat.v1.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate)
        # Set the model optimization op.
        self.opt_op = self.optimizer.apply_gradients(zip(clip_gradients, trainable_params))

# def make_online(self):
#         embedding = K.variable(np.random.uniform(0, 1, (self.dataset.nsize, self.flowargs['embdim'])))
#         prevemb = K.placeholder(ndim=2, dtype='float32')  # (nsize, d)
#         data = K.placeholder(ndim=2, dtype='int32')  # (batchsize, 5), [k, from_pos, to_pos, from_neg, to_neg]
#         weight = K.placeholder(ndim=1, dtype='float32')  # (batchsize, )

#         if K._BACKEND == 'theano':
#             dist_pos = embedding[data[:, 1]] - embedding[data[:, 2]]
#             dist_pos = K.sum(dist_pos * dist_pos, axis=-1)
#             dist_neg = embedding[data[:, 3]] - embedding[data[:, 4]]
#             dist_neg = K.sum(dist_neg * dist_neg, axis=-1)
#         else:
#             dist_pos = K.gather(embedding, K.squeeze(K.slice(data, [0, 1], [-1, 1]), axis=1)) - \
#                        K.gather(embedding, K.squeeze(K.slice(data, [0, 2], [-1, 1]), axis=1))
#             dist_pos = K.sum(dist_pos * dist_pos, axis=-1)
#             dist_neg = K.gather(embedding, K.squeeze(K.slice(data, [0, 3], [-1, 1]), axis=1)) - \
#                        K.gather(embedding, K.squeeze(K.slice(data, [0, 4], [-1, 1]), axis=1))
#             dist_neg = K.sum(dist_neg * dist_neg, axis=-1)

#         margin = 1
#         lprox = K.maximum(margin + dist_pos - dist_neg, 0) * weight

#         # (1, )
#         lprox = K.mean(lprox)

#         # lsmooth
#         lsmooth = embedding - prevemb  # (nsize, d)
#         lsmooth = K.sum(K.square(lsmooth), axis=-1)  # (nsize)
#         lsmooth = K.mean(lsmooth)

#         # loss = lprox + self.flowargs['beta'][0] * lsmooth
#         loss = self.flowargs['beta'][0] * lsmooth

#         return loss