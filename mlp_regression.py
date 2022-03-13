from keras.callbacks import ModelCheckpoint, EarlyStopping

import utils
import trainer

if __name__ == "__main__":
    opt = utils.Config()

    # These data are for training
    x_train_data, y_train_data = utils.import_data(
        data_root_path=opt.data_root_path,
        selected_bands=opt.selected_bands,
        train_for_regression=opt.regression,
        derivative=opt.derivative,
        mushroom_class=opt.mushroom_class,
        normalize="zscore",
        shuffle=True,
    )

    x_train_data, y_train_data = x_train_data[:80], y_train_data[:80]

    # These data are for evaluation and plotting freshness curve
    x_all_data, y_all_data = utils.import_data(
        data_root_path=opt.data_root_path,
        selected_bands=opt.selected_bands,
        train_for_regression=False,
        derivative=opt.derivative,
        mushroom_class=opt.mushroom_class,
        normalize="zscore",
        shuffle=False,
    )

    # Callbacks
    checkpoint = ModelCheckpoint(
        opt.save_path,
        monitor='val_mean_squared_error',
        verbose=2,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        initial_value_threshold=0.0
    )

    early_stop = EarlyStopping(
        monitor='val_mean_squared_error',
        patience=500,
        restore_best_weights=True,
        verbose=0
    )

    callbacks = [checkpoint, early_stop]

    layout = [512]

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

    preds = model.predict(x_all_data)
    trainer.eval_mlp_regression(preds, opt.mushroom_class, layout)
