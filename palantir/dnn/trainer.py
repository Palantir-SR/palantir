import time
import os
import shutil

import tensorflow as tf
import numpy as np

from palantir.dnn.utility import resolve_bilinear

class Trainer:
    def __init__(self, model, loss, learning_rate, checkpoint_dir, log_dir, max_to_keep=1):
        self.loss = loss
        self.learning_rate = learning_rate
        self.checkpoint = tf.train.Checkpoint(model=model)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint, directory=checkpoint_dir, max_to_keep=1)
        self.checkpoint_dir = checkpoint_dir
        self.writer = tf.contrib.summary.create_file_writer(log_dir)

        os.makedirs(log_dir, exist_ok=True)

    def evaluate(self, dataset):
        sr_psnrs = []
        bilinear_psnrs = []

        progbar = tf.keras.utils.Progbar(dataset.num_images)
        for idx, imgs in enumerate(dataset):
            lr_img = imgs[0]
            hr_img = imgs[1]

            lr_img = tf.cast(lr_img, tf.float32)
            sr_img = self.checkpoint.model(lr_img)
            sr_img = tf.clip_by_value(sr_img, 0, 255)
            sr_img = tf.round(sr_img)
            sr_img = tf.cast(sr_img, tf.uint8)
            sr_psnr = tf.image.psnr(sr_img, hr_img, max_val=255)[0]
            if tf.math.is_inf(sr_psnr):
                sr_psnr = 100
            sr_psnrs.append(sr_psnr)

            height = tf.shape(hr_img)[1]
            width = tf.shape(hr_img)[2]
            bilinear_img = resolve_bilinear(lr_img, height, width)
            bilinear_psnr = tf.image.psnr(bilinear_img, hr_img, max_val=255)[0]
            if tf.math.is_inf(bilinear_psnr):
                bilinear_psnr = 100
            bilinear_psnrs.append(bilinear_psnr)

            progbar.update(idx+1)

            """
            bilinear_img = tf.squeeze(bilinear_img)
            hr_img = tf.squeeze(hr_img)
            bilinear_img = tf.image.encode_png(tf.cast(bilinear_img, tf.uint8))
            hr_img = tf.image.encode_png(tf.cast(hr_img, tf.uint8))
            tf.io.write_file('test_lr_{}.png'.format(idx), bilinear_img)
            tf.io.write_file('test_hr_{}.png'.format(idx), hr_img)
            """

        return tf.reduce_mean(sr_psnrs), tf.reduce_mean(bilinear_psnrs)

    def restore(self, checkpoint_dir):
        ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)
        self.checkpoint.restore(ckpt_path)

    def train(self, train_dataset, test_dataset, num_epochs, num_steps_per_epoch, time_log_file, select_best=False, saturation_detection=False):
        loss_mean = tf.keras.metrics.Mean()



        if select_best:
            avg_sr_psnr_all_epochs = []

        progbar = tf.keras.utils.Progbar(num_steps_per_epoch)
        current_time = time.time()
        for curr_step, imgs in enumerate(train_dataset.take(num_epochs * num_steps_per_epoch)):
            lr_img = imgs[0]
            hr_img = imgs[1]
            loss = self.train_step(lr_img, hr_img)
            loss_mean(loss)

            progbar.update(curr_step % num_steps_per_epoch + 1)

            if (curr_step + 1) % num_steps_per_epoch == 0:
                curr_epoch = (curr_step + 1) // num_steps_per_epoch
                loss_value = loss_mean.result()
                loss_mean.reset_states()
                avg_sr_psnr, avg_bilinear_psnr = self.evaluate(test_dataset)
                with self.writer.as_default(), tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('Loss', loss_value, step=curr_epoch * num_steps_per_epoch)
                    tf.contrib.summary.scalar('PSNR_SR', avg_sr_psnr, step=curr_epoch * num_steps_per_epoch)
                    tf.contrib.summary.scalar('PSNR_Bilinear', avg_bilinear_psnr, step=curr_epoch * num_steps_per_epoch)
                    tf.contrib.summary.scalar('PSNR_Gain', avg_sr_psnr - avg_bilinear_psnr, step=curr_epoch * num_steps_per_epoch)
                    tf.contrib.summary.flush(self.writer)
                self.checkpoint_manager.save()

                os.makedirs(os.path.join(self.checkpoint_dir, f"epoch_{curr_epoch}"))
                for file in os.listdir(self.checkpoint_dir):
                    if os.path.isfile(os.path.join(self.checkpoint_dir, file)):
                        shutil.copy(os.path.join(self.checkpoint_dir, file), os.path.join(self.checkpoint_dir, f"epoch_{curr_epoch}"))

                progbar = tf.keras.utils.Progbar(num_steps_per_epoch)
                print('[{} epoch] PSNR (bilinear) - {:.2f}dB, PSNR (SR) - {:.2f}dB'.format(curr_epoch, avg_bilinear_psnr, avg_sr_psnr))

                time_log_file.write(f"[Epoch {curr_epoch}]: {time.time()-current_time} seconds")
                current_time = time.time()
                time_log_file.flush()
    
                if select_best:
                    avg_sr_psnr_all_epochs.append(avg_sr_psnr)
                
                if saturation_detection:
                    assert select_best
                    # detect saturation
                    DETECTION_WINDOW = 5
                    PSNR_SATURATION_THRESHOLD = 0.1
                    if len(avg_sr_psnr_all_epochs) > DETECTION_WINDOW:
                        if max(avg_sr_psnr_all_epochs) < max(avg_sr_psnr_all_epochs[:-DETECTION_WINDOW]) + PSNR_SATURATION_THRESHOLD:
                            break

        if select_best:
            for file_name in os.listdir(self.checkpoint_dir):
                if file_name.startswith('ckpt-') or file_name == "checkpoint":
                    os.remove(os.path.join(self.checkpoint_dir, file_name))
            best_epoch_idx = np.argmax(avg_sr_psnr_all_epochs)
            for file_name in os.listdir(os.path.join(self.checkpoint_dir, f"epoch_{best_epoch_idx+1}")):
                if file_name.startswith('ckpt-') or file_name == "checkpoint":
                    shutil.copy(os.path.join(self.checkpoint_dir, f"epoch_{best_epoch_idx+1}", file_name), self.checkpoint_dir)



    def train_step(self, lr_img, hr_img):
        with tf.GradientTape() as tape:
            lr_img = tf.cast(lr_img, tf.float32)
            hr_img = tf.cast(hr_img, tf.float32)

            sr_img = self.checkpoint.model(lr_img, training=True)
            loss_value = self.loss(sr_img, hr_img)

            gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

        return loss_value

#we used a training setting identical to EDSR (CVPRW'17)
class PalantirTrainer(Trainer):
    def __init__(self,
                    model,
                    checkpoint_dir,
                    log_dir,
                    learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[200000], values=[1e-4, 5e-5])):
        super().__init__(model, loss=tf.keras.losses.MeanAbsoluteError(), learning_rate=learning_rate, checkpoint_dir=checkpoint_dir, log_dir=log_dir)

    def train(self, train_dataset, test_dataset, num_epochs=300, num_steps_per_epoch=1000, time_log_file=None, select_best=False, saturation_detection=False):
        super().train(train_dataset, test_dataset, num_epochs, num_steps_per_epoch, time_log_file, select_best=select_best, saturation_detection=saturation_detection)