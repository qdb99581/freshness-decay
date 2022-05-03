from keras.callbacks import ModelCheckpoint, EarlyStopping

import utils
import trainer

if __name__ == "__main__":
    opt = utils.Config()

    # These data are for training
    x_train_data, y_train_data = utils.import_data(
        data_root_path=opt.data_root_path,
        selected_bands=opt.selected_bands,
        train_for_regression=True,
        derivative=opt.derivative,
        mushroom_class=opt.mushroom_class,
        normalize=None,
        shuffle=True,
    )

    x_train_data, y_train_data = x_train_data[:80], y_train_data[:80]

    # Callbacks
    checkpoint = ModelCheckpoint(
        filepath=opt.save_path,
        monitor='val_mean_squared_error',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        save_freq='epoch', # Save model for every epoch
        mode='auto'
    )

    early_stop = EarlyStopping(
        monitor='val_mean_squared_error',
        patience=500,
        restore_best_weights=True,
        verbose=0
    )

    callbacks = [checkpoint, early_stop]

    layout = 'original'

    model = trainer.build_tf_model(
        neurons_layout=layout,
        activation=opt.activation,
        selected_bands=opt.selected_bands,
        learning_rate=opt.learning_rate,
        objective='regression',
    )

    history = model.fit(
        x_train_data, y_train_data,
        batch_size=opt.batch_size,
        validation_split=opt.valid_ratio,
        epochs=opt.n_epochs,
        verbose='auto',
        callbacks=callbacks,
    )
