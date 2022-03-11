# Prepare for SVR
from sklearn.svm import SVR
import numpy as np

# Custom Modules
import utils

if __name__ == "__main__":
    opt = utils.Config()

    # Prepare data and train_test_split
    x_train_data, y_train_data = utils.import_data(
        data_root_path=opt.data_root_path,
        selected_bands=opt.selected_bands,
        regression=opt.regression,
        derivative=opt.derivative,
        mushroom_class=opt.mushroom_class,
        normalize="zscore",
        shuffle=True,
    )

    # Warning from sklearn documentation:
    # If training data is not C-contihuous, the error may occur.
    print(
        f"Check if the input data is C-contiguous: {x_train_data.flags['C_CONTIGUOUS']}")

    # Define SVR
    model = SVR(
        kernel='linear',
        C=0.1
    )

    # Training
    model.fit(x_train_data, y_train_data)

    # Import all of the data
    x_data, y_data = utils.import_data(
        data_root_path=opt.data_root_path,
        selected_bands=opt.selected_bands,
        regression=False,
        derivative=opt.derivative,
        mushroom_class=opt.mushroom_class,
        normalize="zscore",
        shuffle=False,
    )

    # Evaluating
    preds = model.predict(x_data)

    n_data_per_class = 15
    mean_regression_scores = []
    for i in range(0, len(preds), n_data_per_class):
        mean_of_class = np.mean(preds[i:i+n_data_per_class])
        mean_regression_scores.append(mean_of_class)

    print(mean_regression_scores)
