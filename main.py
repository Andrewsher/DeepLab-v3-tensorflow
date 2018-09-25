import os
import tensorflow as tf
import numpy as np
import cv2
import csv

import config as cfg
import deeplab
import data

class Solver(object):

    def __init__(self, net, data):
        self.net = net
        self.data = data
        self.weights_file = cfg.weights_file
        self.learning_rate = cfg.learning_rate
        self.batch_size = cfg.batch_size
        self.output_dir = cfg.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=5)
        self.ckpt_file = os.path.join(self.output_dir, 'deeplabv3+')
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        self.global_step = tf.train.create_global_step()
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.net.loss, global_step=self.global_step)
        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        if self.weights_file is not None:
            print('Restoring weights from: ' + self.weights_file)
            self.saver.restore(self.sess, self.weights_file)

        self.writer.add_graph(self.sess.graph)

    def train(self):
        f = open('dice_record.csv', 'w')
        csv_writer = csv.writer(f)
        csv_writer.writerow(['iteration', 'train dice', 'val dice', 'val dice without 0', 'loss'])
        step = 0

        for iteration in range(10000):
            images, labels = self.data.get_train_data(batch_size=self.batch_size)
            feed_dict = {self.net.images: images, self.net.labels: labels, self.net.keep_prob: np.float32(0.6)}
            summary_str, loss, _, train_dice = self.sess.run(
                [self.summary_op, self.net.loss, self.train_op, self.net.dice_coef],
                feed_dict=feed_dict)
            self.writer.add_summary(summary_str, step)
            step += 1
            print('iteration = ', iteration, ' loss = ', loss, 'train dice = ', train_dice)
            if iteration % 100 == 0:
                val_dice, val_dice_without_0 = self.val_per_epoch()
                print('val dice = ', val_dice, 'val dice without 0 = ', val_dice_without_0)
                csv_writer.writerow([str(iteration), str(train_dice), str(val_dice), str(val_dice_without_0), str(loss)])
        self.saver.save(self.sess, self.ckpt_file, global_step=self.global_step)
        f.close()

    def val_per_epoch(self):
        dice_coef_set = []
        dice_coef_without_0 = []
        for i in range(self.data.val_set_num):
            images, labels = self.data.get_val_data()
            feed_dict = {self.net.images: images, self.net.labels: labels, self.net.keep_prob: np.float32(1)}
            dice_coef = self.sess.run([self.net.dice_coef], feed_dict=feed_dict)
            dice_coef_set.append(dice_coef)
            if np.sum(labels) > 1:
                dice_coef_without_0.append(dice_coef)
        return np.average(np.array(dice_coef_set)), np.average(np.array(dice_coef_without_0))

    def evaluate(self):
        # dice_coef_set = []
        # dice_set_without_0 = []
        for i in range(len(self.data.val_idxs)):
            images, labels = self.data.get_val_data()
            feed_dict = {self.net.images: images, self.net.labels: labels, self.net.keep_prob: np.float32(1)}
            dice_coef, predicts = self.sess.run([self.net.dice_coef, self.net.predits], feed_dict=feed_dict)
            print('dice coefficient = ', dice_coef)
            # dice_coef_set.append(dice_coef)
            cv2.imshow('predicts', np.squeeze(predicts))
            # cv2.waitKey(0)
            cv2.imshow('label', np.squeeze(labels))
            cv2.waitKey(0)
        # for i in range(self.data.val_set_num):
        #     images, labels = self.data.get_val_data()
        #     feed_dict = {self.net.images: images, self.net.labels: labels, self.net.keep_prob: np.float32(1)}
        #     dice_coef = self.sess.run([self.net.dice_coef], feed_dict=feed_dict)
        #     print('dice coefficient = ', dice_coef)
        #     dice_coef_set.append(dice_coef)
        #     if np.sum(labels) >= 1:
        #         dice_set_without_0.append(dice_coef)
        #
        # print('average dice coefficient = ', np.average(np.array(dice_coef_set)))
        # print('average dice without 0 = ', np.average(np.array(dice_set_without_0)))


    def save_cfg(self):

        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)



def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    net = deeplab.DeepLab()
    dataset = data.data()
    solver = Solver(net, dataset)

    # print('Start training ...')
    # solver.train()
    # print('Training done.')
    # solver.save_cfg()

    print('Start evaluation ...')
    solver.evaluate()
    print('Evaluation done.')


if __name__ == '__main__':

    main()