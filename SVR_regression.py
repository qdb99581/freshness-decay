# Prepare for SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVR

# Custom Modules
import utils

if __name__ == "__main__":
    opt = utils.Config()

    # Prepare data and train_test_split
    x_data, y_data = utils.import_data(
        data_root_path=opt.data_root_path,
        selected_bands=opt.selected_bands,
        regression=opt.regression,
        derivative=opt.derivative,
        mushroom_class=opt.mushroom_class,
        normalize="zscore",
        shuffle=True,
    )

    x_train, y_train, x_test, y_test = utils.train_test_split(
        x_data, y_data, train_ratio=opt.train_ratio)

    # Warning from sklearn documentation:
    # If training data is not C-contihuous, the error may occur.
    print(
        f"Check if the input data is C-contiguous: {x_train.flags['C_CONTIGUOUS']}")

    # Define scaler and SVR
    scaler = StandardScaler()
    svr = SVR()

    # Scale the trainig data by z-score (Normalization)
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
    model = GridSearchCV(pipe, tuned_params, cv=5)
    model.fit(x_train, y_train)

    print("Best parameters set: ")
    print(model.best_params_)
    print()

    means = model.cv_results_["mean_test_score"]
    stds = model.cv_results_["std_test_score"]

    for mean, std, params in zip(means, stds, model.cv_results_["params"]):
        print(f"Mean Accuracy: {mean:0.4f} ± {std*2:0.4f} for {params}")

    y_true, y_pred = y_test, model.predict(x_test)

    # print(classification_report(y_true, y_pred))
