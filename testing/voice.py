import tensorflow as tf

MODEL = '../models/sttw_pretrained/model.ckpt-205919.meta'
print(f'Model location: {MODEL}')

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                log_device_placement=True)) as sess:
    # load the computation graph
    loader = tf.train.import_meta_graph(MODEL)
    sess.run(tf.global_variables_initializer())
    loader = loader.restore(sess, MODEL)
    train_op = tf.get_collection()
    print('--------------TRAIN OPS--------------')
    print(len(train_op))
    sess.close()



