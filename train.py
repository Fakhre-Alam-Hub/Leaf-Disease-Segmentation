import os
import tensorflow as tf
from unet import unet_model
import prepare_dataframe as pdf
from segmentation_models.losses import bce_jaccard_loss, bce_dice_loss
from segmentation_models.metrics import iou_score
import preprocess_pipeline

def compile_model():
    model = unet_model()

    model.compile(optimizer='adam',
                loss=bce_jaccard_loss,
                metrics=[iou_score])

    return model

def model_callbacks():
    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=5, monitor='val_iou_score',
        mode='max', restore_best_weights=True
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='models/model_disease_check.h5', monitor='val_iou_score',
        verbose=0, save_best_only=True, mode='max'
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_iou_score', factor=0.1, patience=5, verbose=0,
        mode='max', min_delta=0.0001, min_lr=0.00002,
    )

    tb_callback = tf.keras.callbacks.TensorBoard('models/logs_check/logs')

    return [early_stop, reduce_lr, checkpoint, tb_callback]


def fit_model(model, train_batches, test_batches, 
              STEPS_PER_EPOCH, VALIDATION_STEPS, EPOCHS):

    # callbacks
    callbacks = model_callbacks()

    # fit model
    model_history = model.fit(train_batches, epochs=EPOCHS,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_data=test_batches,
                            validation_steps=VALIDATION_STEPS,
                            callbacks=callbacks)


if __name__ == "__main__":
    BASE_DIR = os.getcwd()

    img_folder = os.path.join(BASE_DIR, 'aug_data/images')
    mask_folder = os.path.join(BASE_DIR, 'aug_data/masks')

    df = pdf.get_df(img_folder, mask_folder)
    train_df, test_df = pdf.split_df(df)

    train_images = preprocess_pipeline.create_dataset(train_df)
    test_images = preprocess_pipeline.create_dataset(test_df)

    TRAIN_LENGTH = len(train_df)
    TEST_LENGTH = len(test_df)

    print(TRAIN_LENGTH, TEST_LENGTH)
    BATCH_SIZE = 16
    BUFFER_SIZE = 500
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
    VALIDATION_STEPS = TEST_LENGTH // BATCH_SIZE

    train_batches = (train_images
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .repeat()
        .prefetch(buffer_size=tf.data.AUTOTUNE))

    test_batches = test_images.batch(BATCH_SIZE)

    model = compile_model()

    fit_model(model, train_batches, test_batches,
              STEPS_PER_EPOCH, VALIDATION_STEPS, EPOCHS=3)

    print('Done training.')
