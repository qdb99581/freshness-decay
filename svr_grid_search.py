# Prepare for SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
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

    # Define scaler and SVR
    scaler = StandardScaler()
    svr = SVR()

    pipe = Pipeline(steps=[("scaler", scaler), ("svr", svr)])

    # Set parameters for Grid seach cross-validation
    tuned_params = [
        {
            "svr__kernel": ["rbf"],
            "svr__gamma": [1e-4, 1e-3, 'scale', 'auto'],
            "svr__C": [1e-2, 0.1, 1, 10, 100]
        },

        {
            "svr__kernel": ["linear"],
            "svr__C": [1e-2, 0.1, 1, 10, 100]
        },
    ]

    # Grid Search
    model = GridSearchCV(pipe, tuned_params)
    model.fit(x_train_data, y_train_data)

    print("Best parameters set: ")
    print()
    print(model.best_params_)
    print()

    means = model.cv_results_["mean_test_score"]
    stds = model.cv_results_["std_test_score"]

    for mean, std, params in zip(means, stds, model.cv_results_["params"]):
        print(f"Mean R2 Scores: {mean:0.4f} ± {std*2:0.4f} for {params}")
