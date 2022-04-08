import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Custom Modules
import utils
import trainer


if __name__ == "__main__":
    opt = utils.Config()

    x_data, y_data = utils.import_data(
        data_root_path=opt.data_root_path,
        selected_bands=opt.selected_bands,
        derivative=opt.derivative,
        mushroom_class=opt.mushroom_class,
        normalize="zscore",
        shuffle=True,
    )

    # Callbacks
    checkpoint = ModelCheckpoint(
        opt.save_path,
        monitor='val_accuracy',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        initial_value_threshold=0.0
    )

    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=500,
        restore_best_weights=True,
        verbose=0
    )

    callbacks = [checkpoint, early_stop]

    mlp_layout = [512, 512, 32]
    mlp_acc_hist, svm_acc_hist = [], []
    kf = KFold(n_splits=opt.n_KFold)

    k = 0
    for train_idx, test_idx in kf.split(x_data):
        x_train, x_test = x_data[train_idx], x_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]

        # MLP model
        mlp_model = trainer.build_tf_model(
            neurons_layout=mlp_layout,
            activation=opt.activation,
            selected_bands=opt.selected_bands,
            learning_rate=opt.learning_rate,
        )
        
        print(f"Start training with the {k+1}/{opt.n_KFold} fold.")
        # Train MLP model
        mlp_history = mlp_model.fit(
            x_train, y_train,
            batch_size=opt.batch_size,
            validation_split=opt.valid_ratio,
            epochs=opt.n_epochs,
            verbose=1,
            callbacks=callbacks
        )

        # Evaluate MLP model
        mlp_scores = mlp_model.evaluate(x_test, y_test)
        mlp_acc_hist.append(mlp_scores[1])

        # Construct SVM model
        scaler = StandardScaler() # Normalization
        svm = SVC(
            C=100,
            kernel="linear",
            gamma=1e-4
        )
        # svm_model = Pipeline(steps=[("scaler", scaler), ("svm", svm)])
        svm_model = Pipeline(steps=[("svm", svm)])

        # Train SVM model
        svm_model.fit(x_train, y_train)

        # Evaluate SVM model
        svm_scores = svm_model.score(x_test, y_test)
        svm_acc_hist.append(svm_scores)

        k += 1

    # Compute two models avg accuracy and std
    mlp_mean_acc = round(np.mean(mlp_acc_hist) * 100, 2)
    mlp_std_acc = round(np.std(mlp_acc_hist) * 100, 2)

    svm_mean_acc = round(np.mean(svm_acc_hist) * 100, 2)
    svm_std_acc = round(np.std(svm_acc_hist) * 100, 2)

    # Generate report
    mlp_report = f"Accuracy of MLP: {mlp_mean_acc:2.2f}% ± {mlp_std_acc:2.2f}% for {mlp_layout}."
    svm_report = f"Accuracy of SVM: {svm_mean_acc:2.2f}% ± {svm_std_acc:2.2f}% for C={100}, gamma={1e-4}."

    print()
    print("=" * 40)
    print(mlp_report)
    print(svm_report)
    print("=" * 40)

    