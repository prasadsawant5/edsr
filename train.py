import tensorflow as tf
from tensorflow.python.keras import metrics
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.experimental.set_memory_growth(gpus[1], True)
tf.autograph.set_verbosity(3)

from tensorflow.python import debug as tf_debug
from tensorflow.python.ops.gen_dataset_ops import MapDataset
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.metrics import Mean
import os
import numpy as np
from model.my_model import MyModel
from model.edsr import *
import time

LR = 1e-4
BATCH_SIZE = 8
EPOCHS = 10

SAVED_MODELS = './saved_models'
LOGS = './logs'

LOW_RES_SIZE = (180, 240)
HI_RES_SIZE = (360, 480)

DATA_HIGH_RES = './data/high_res/'
DATA_LOW_RES = './data/low_res/'

DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255

def configure_for_performance(ds: MapDataset) -> MapDataset:
    ds = ds.cache()
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def get_dataset() -> MapDataset:
    normalization = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

    hi_res_ds = image_dataset_from_directory(
        DATA_HIGH_RES, 
        label_mode=None, 
        image_size=HI_RES_SIZE, 
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    low_res_ds = image_dataset_from_directory(
        DATA_LOW_RES,
        label_mode=None,
        image_size=LOW_RES_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    dataset = tf.data.Dataset.zip((low_res_ds, hi_res_ds))
    dataset = dataset.map(lambda low_res, hi_res: (normalization(low_res), normalization(hi_res)))
    dataset = configure_for_performance(dataset)

    return dataset



if __name__ == '__main__':
    if not os.path.exists(SAVED_MODELS):
        os.mkdir(SAVED_MODELS)

    if not os.path.exists(LOGS):
        os.mkdir(LOGS)

    milli_sec = str(int(round(time.time() * 1000)))
    tensorboard_dir = os.path.join(LOGS, milli_sec)
    save_dir = os.path.join(SAVED_MODELS, milli_sec)

    os.mkdir(tensorboard_dir)
    os.mkdir(save_dir)

    dataset = get_dataset()

    loss = MeanAbsoluteError()
    optimizer = Adam(LR, epsilon=1e-8)
    train_acc_metric = Mean()
    train_writer = tf.summary.create_file_writer(tensorboard_dir)

    x = tf.keras.Input(shape=(LOW_RES_SIZE[0], LOW_RES_SIZE[1], 3), name='X')

    # tf.debugging.experimental.enable_dump_debug_info(tensorboard_dir)
    model = MyModel().build(x)

    model.summary()

    tf.debugging.experimental.enable_dump_debug_info(
        tensorboard_dir,
        tensor_debug_mode="FULL_HEALTH", 
        circular_buffer_size=-1)

    # tf.debugging.enable_check_numerics()

    @tf.function
    def train_step(x, y) -> tuple:
        with tf.GradientTape(persistent=True) as tape:
            logits = model(x, training=True)

            loss_value = loss(y, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        train_acc_metric.update_state(y, logits)
        train_acc = train_acc_metric.result()

        return loss_value, train_acc

    
    total_steps = len(os.listdir(os.path.join(DATA_LOW_RES, 'images'))) // BATCH_SIZE

    for epoch in range(EPOCHS):
        step = 0
        final_loss = None
        final_acc = None
        for batch_low, batch_high in dataset.take(total_steps):
            tf.summary.trace_on(graph=True, profiler=True)
            loss_value, acc = train_step(batch_low, batch_high)

            with train_writer.as_default():
                tf.summary.trace_export(name='my_func_trace', step=step, profiler_outdir=tensorboard_dir)

            print(
                '{:03d}/{:03d}   {:03d}/{:03d}, Loss: {:6f}, Accuracy: {:6f}'
                .format(epoch + 1, EPOCHS, step + 1, total_steps, loss_value, acc)
            )

            step += 1

            final_loss, final_acc = loss_value, acc

        with train_writer.as_default():
            tf.summary.scalar('loss', final_loss, step=epoch)
            tf.summary.scalar('accuracy', final_acc, step=epoch)

        train_acc_metric.reset_states()

    # model.save(save_dir)