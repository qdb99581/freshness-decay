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
        normalize='zscore',
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
        normalize='zscore',
        shuffle=False,
    )

    early_stop = EarlyStopping(
        monitor='val_mean_squared_error',
        patience=500,
        restore_best_weights=True,
        verbose=0
    )

    callbacks = [early_stop]

    model_id = "MLP32"
    layout = opt.mlp_layout[model_id]

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
    trainer.eval_mlp_regression(preds, opt.mushroom_class, model_id)
