from tqdm import tqdm

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold
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

    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=500,
        restore_best_weights=True,
        verbose=0
    )

    callbacks = [early_stop]

    mlp_layouts = opt.mlp_layout
    # mlp_layout_acc_dict has the same keys as mlp_layouts, but the value is a list,
    # as the list contains different accuracy from K-Fold.
    # e.g., {'MLP21': [20.0, 34.0, 15.0]}
    mlp_layout_acc_dict = {}

    svm_layouts = opt.svm_layouts
    # svm_layout_acc_dict has the same keys as svm_layouts, but the value is a list,
    # as the list contains different accuracy from K-Fold.
    # e.g., {'lSVM1': [20.0, 34.0, 15.0]}
    svm_params_acc_dict = {}

    kf = KFold(n_splits=opt.n_KFold)
    k = 0
    for train_idx, test_idx in kf.split(x_data):
        tqdm.write(f"Start training with the {k+1}/{opt.n_KFold} fold...")
        x_train, x_test = x_data[train_idx], x_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]

        # Start training on different MLP layouts
        for mlp_layout_id, layout in tqdm(mlp_layouts.items()):
            # MLP model
            mlp_model = trainer.build_tf_model(
                neurons_layout=layout,
                activation=opt.activation,
                selected_bands=opt.selected_bands,
                learning_rate=opt.learning_rate,
            )

            tqdm.write("=" * 15)
            tqdm.write(
                f"Start training MLP layout: {mlp_layout_id}: {layout}...")
            tqdm.write(f"Current training progress: {k+1}/{opt.n_KFold} fold.")
            # Train MLP model
            mlp_history = mlp_model.fit(
                x_train, y_train,
                batch_size=opt.batch_size,
                validation_split=opt.valid_ratio,
                epochs=opt.n_epochs,
                verbose=1,
                callbacks=callbacks
            )

            # Evaluate and record scores for MLP model
            cur_mlp_scores = mlp_model.evaluate(x_test, y_test)
            cur_mlp_acc = cur_mlp_scores[1]

            # Initialize if there's no current ID as key
            if mlp_layout_id not in mlp_layout_acc_dict:
                mlp_layout_acc_dict[mlp_layout_id] = []

            mlp_layout_acc_dict[mlp_layout_id].append(cur_mlp_acc)

        tqdm.write(f"Start training on SVM...")
        for svm_layout_id, params_list in tqdm(svm_layouts.items()):
            # Determine current kernel to construct SVM model
            if len(params_list) == 1:
                svm_model = SVC(
                    C=params_list[0],
                    kernel='linear'
                )
            else:
                svm_model = SVC(
                    C=params_list[0],
                    gamma=params_list[1],
                    kernel='rbf'
                )

            # Train SVM model
            svm_model.fit(x_train, y_train)

            # Evaluate SVM model
            cur_svm_score = svm_model.score(x_test, y_test)
            if svm_layout_id not in svm_params_acc_dict:
                svm_params_acc_dict[svm_layout_id] = []

            svm_params_acc_dict[svm_layout_id].append(cur_svm_score)

        k += 1

    print("="*30)
    print(f"Finish training, generating reports...")

    mlp_scores = utils.compute_scores(mlp_layout_acc_dict)
    svm_scores = utils.compute_scores(svm_params_acc_dict)

    results_dir = "./mlp_svm_results/"
    if opt.derivative:
        data_mode = "derivative"
    else:
        data_mode = "reflectance"

    mlp_score_df = pd.DataFrame(
        list(mlp_scores.items()),
        columns=["Model ID", "Scores"]).to_csv(results_dir + f"/mlp_results_{opt.mushroom_class}_{data_mode}.csv")
    svm_scores_df = pd.DataFrame(
        list(svm_scores.items()),
        columns=["Model ID", "Scores"]).to_csv(results_dir + f"/svm_results_{opt.mushroom_class}_{data_mode}.csv")

    print(f"Reports are saved at {results_dir}")
