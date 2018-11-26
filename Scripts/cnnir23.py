#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
import os
import sys
import time
import numpy as np
import tensorflow as tf
import inception_resnet_v2
import Accessory as ac

slim = tf.contrib.slim


class INCEPTION():
    """
    Use the Inception-ResV2 architecture

    """

    DEFAULTS = {
        "batch_size": 128,
        "dropout": 0.8,
        "learning_rate": 1E-3
    }

    RESTORE_KEY = "cnn_to_restore"

    def __init__(self, input_dim, d_hyperparams={},
                 save_graph_def=True, meta_graph=None,
                 log_dir="./log", meta_dir="./meta"):

        self.input_dim = input_dim
        self.__dict__.update(INCEPTION.DEFAULTS, **d_hyperparams)
        self.sesh = tf.Session()

        if meta_graph:  # load saved graph
            model_name = os.path.basename(meta_graph)
            meta_graph = os.path.abspath(meta_graph)
            tf.train.import_meta_graph(meta_dir + '/' + model_name +'.meta').restore(
                self.sesh, meta_dir + '/' + model_name)
            handles = self.sesh.graph.get_collection(INCEPTION.RESTORE_KEY)


        else:  # build graph from scratch
            self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")
            handles = self._buildGraph()
            for handle in handles:
                tf.add_to_collection(INCEPTION.RESTORE_KEY, handle)
            self.sesh.run(tf.global_variables_initializer())

        # unpack handles for tensor ops to feed or fetch for lower layers
        (self.x_in, self.dropout_, self.is_train,
         self.y_in, self.logits, self.net, self.w, self.pred, self.pred_cost,
         self.global_step, self.train_op, self.merged_summary) = handles

        if save_graph_def:  # tensorboard
            try:
                os.mkdir(log_dir + '/training')
                os.mkdir(log_dir + '/validation')

            except(FileExistsError):
                pass

            self.train_logger = tf.summary.FileWriter(log_dir + '/training', self.sesh.graph)
            self.valid_logger = tf.summary.FileWriter(log_dir + '/validation', self.sesh.graph)

    @property
    def step(self):
        return self.global_step.eval(session=self.sesh)

    def _buildGraph(self):
        x_in = tf.placeholder(tf.float32, name="x")
        x_in_reshape = tf.reshape(x_in, [-1, self.input_dim[1], self.input_dim[2], 3])

        dropout = tf.placeholder_with_default(1., shape=[], name="dropout")

        y_in = tf.placeholder(dtype=tf.int8, name="y")

        onehot_labels = tf.one_hot(indices=tf.cast(y_in, tf.int32), depth=2)

        is_train = tf.placeholder_with_default(True, shape=[], name="is_train")

        logits, nett, ww = inception_resnet_v2.inception_resnet_v2(x_in_reshape,
                                                                   num_classes=2,
                                                                   is_training=is_train,
                                                                   dropout_keep_prob=dropout,
                                                                   reuse=None,
                                                                   create_aux_logits=True,
                                                                   scope='InceptionRes2')

        pred = tf.nn.softmax(logits, name="prediction")

        global_step = tf.Variable(0, trainable=False)

        pred_cost = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

        tf.summary.scalar("InceptionRes2_cost", pred_cost)

        train_op = tf.contrib.layers.optimize_loss(
            loss=pred_cost,
            learning_rate=self.learning_rate,
            global_step=global_step,
            optimizer="Adam")

        merged_summary = tf.summary.merge_all()

        return (x_in, dropout, is_train,
                y_in, logits, nett, ww, pred, pred_cost,
                global_step, train_op, merged_summary)

    def inference(self, X, dirr, train_status=False, Not_Realtest=True):
        now = datetime.now().isoformat()[11:]
        print("------- Testing begin: {} -------\n".format(now), flush=True)
        rd = 0
        pdx = []
        yl = []
        if Not_Realtest:
            itr, file, ph = X.data()
            next_element = itr.get_next()
            with tf.Session() as sessa:
                sessa.run(itr.initializer, feed_dict={ph: file})
                while True:
                    try:
                        x, y = sessa.run(next_element)
                        feed_dict = {self.x_in: x, self.is_train: train_status}
                        fetches = [self.pred, self.net, self.w]
                        pred, net, w = self.sesh.run(fetches, feed_dict)
                        ac.CAM(net, w, pred, x, y, dirr, 'Test', rd)
                        if rd == 0:
                            pdx = pred
                            yl = y
                        else:
                            pdx = np.concatenate((pdx, pred), axis=0)
                            yl = np.concatenate((yl, y), axis=None)
                        rd += 1
                    except tf.errors.OutOfRangeError:
                        ac.metrics(pdx, yl, dirr, 'Test')
                        break
        else:
            itr, img, ph = X.data(Not_Realtest=False)
            next_element = itr.get_next()
            with tf.Session() as sessa:
                sessa.run(itr.initializer, feed_dict={ph: img})
                while True:
                    try:
                        x = sessa.run(next_element)
                        feed_dict = {self.x_in: x, self.is_train: train_status}
                        fetches = [self.pred, self.net, self.w]
                        pred, net, w = self.sesh.run(fetches, feed_dict)
                        # ac.CAM_R(net, w, pred, x, dirr, 'Test', rd)
                        if rd == 0:
                            pdx = pred
                        else:
                            pdx = np.concatenate((pdx, pred), axis=0)
                        rd += 1
                    except tf.errors.OutOfRangeError:
                        ac.realout(pdx, dirr, 'Test')
                        break

        now = datetime.now().isoformat()[11:]
        print("------- Testing end: {} -------\n".format(now), flush=True)

    def get_global_step(self, X):
        itr, file, ph = X.data()
        next_element = itr.get_next()
        with tf.Session() as sessa:
            sessa.run(itr.initializer, feed_dict={ph: file})
            x, y = sessa.run(next_element)

            feed_dict = {self.x_in: x, self.y_in: y}

            fetches = [self.global_step]

            i = self.sesh.run(fetches, feed_dict)

        return i

    def train(self, X, ct, bs, dirr, max_iter=np.inf, cross_validate=True, verbose=True, save=True, outdir="./out"):
        start_time = time.time()
        if save:
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

        try:
            err_train = 0
            now = datetime.now().isoformat()[11:]
            print("------- Training begin: {} -------\n".format(now), flush=True)
            itr, file, ph = X.data()
            next_element = itr.get_next()
            with tf.Session() as sessa:
                sessa.run(itr.initializer, feed_dict={ph: file})
                while True:
                    try:
                        x, y = sessa.run(next_element)

                        feed_dict = {self.x_in: x, self.y_in: y,
                                     self.dropout_: self.dropout}

                        fetches = [self.merged_summary, self.logits, self.pred,
                                   self.pred_cost, self.global_step, self.train_op]

                        summary, logits, pred, cost, i, _ = self.sesh.run(fetches, feed_dict)

                        self.train_logger.add_summary(summary, i)
                        err_train += cost

                        if i % 1000 == 0 and verbose:
                            print("round {} --> cost: ".format(i), cost, flush=True)

                            if cross_validate:
                                xv, yv = sessa.run(next_element)

                                feed_dict = {self.x_in: xv, self.y_in: yv}
                                fetches = [self.pred_cost, self.merged_summary]
                                valid_cost, valid_summary = self.sesh.run(fetches, feed_dict)

                                self.valid_logger.add_summary(valid_summary, i)

                                print("round {} --> CV cost: ".format(i), valid_cost, flush=True)

                        if i == max_iter-int(i/1000)-2 and verbose:  # and i >= 10000:

                            if cross_validate:
                                now = datetime.now().isoformat()[11:]
                                print("------- Validation begin: {} -------\n".format(now), flush=True)
                                xv, yv = sessa.run(next_element)

                                feed_dict = {self.x_in: xv, self.y_in: yv}
                                fetches = [self.pred_cost, self.merged_summary, self.pred, self.net, self.w]
                                valid_cost, valid_summary, pred, net, w = self.sesh.run(fetches, feed_dict)

                                self.valid_logger.add_summary(valid_summary, i)

                                print("round {} --> Last CV cost: ".format(i), valid_cost, flush=True)
                                ac.CAM(net, w, pred, xv, yv, dirr, 'Validation')
                                ac.metrics(pred, yv, dirr, 'Validation')
                                now = datetime.now().isoformat()[11:]
                                print("------- Validation end: {} -------\n".format(now), flush=True)

                        # if i%50000 == 0 and save:
                        #     interfile=os.path.join(os.path.abspath(outdir), "{}_cnn_{}".format(
                        #             self.datetime, "_".join(map(str, self.input_dim))))
                        #     saver.save(self.sesh, interfile, global_step=self.step)

                    except tf.errors.OutOfRangeError:
                        print("final avg cost (@ step {} = epoch {}): {}".format(
                            i+1, np.around(i / ct * bs), err_train / i), flush=True)

                        now = datetime.now().isoformat()[11:]
                        print("------- Training end: {} -------\n".format(now), flush=True)

                        if save:
                            outfile = os.path.join(os.path.abspath(outdir),
                                                   "inceptionres2_{}".format("_".join(['dropout', str(self.dropout)])))
                            saver.save(self.sesh, outfile, global_step=None)
                        try:
                            self.train_logger.flush()
                            self.train_logger.close()
                            self.valid_logger.flush()
                            self.valid_logger.close()

                        except(AttributeError):  # not logging
                            print('Not logging', flush=True)

                        break
            print("--- %s seconds ---" % (time.time() - start_time))

        except(KeyboardInterrupt):

            print("final avg cost (@ step {} = epoch {}): {}".format(
                i, np.around(i / ct * bs), err_train / i), flush=True)

            now = datetime.now().isoformat()[11:]
            print("------- Training end: {} -------\n".format(now), flush=True)

            if save:
                outfile = os.path.join(os.path.abspath(outdir), "inceptionres2_{}".format("_".join(['dropout', str(self.dropout)])))
                saver.save(self.sesh, outfile, global_step=None)
            try:
                self.train_logger.flush()
                self.train_logger.close()
                self.valid_logger.flush()
                self.valid_logger.close()

            except(AttributeError):  # not logging
                print('Not logging', flush=True)

            sys.exit(0)


