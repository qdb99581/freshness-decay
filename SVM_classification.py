# Prepare for SVM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

# Custom Modules
import utils

if __name__ == "__main__":
    opt = utils.Config()

    # Prepare data and train_test_split
    x_data, y_data = utils.import_data(
        data_root_path=opt.data_root_path,
        selected_bands=opt.selected_bands,
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

    # Define scaler and SVM
    scaler = StandardScaler()
    svm = SVC()

    # Scale the trainig data by z-score (Normalization)
    pipe = Pipeline(steps=[("scaler", scaler), ("svm", svm)])

    # Set parameters for Grid seach cross-validation
    tuned_params = [
        {
            "svm__kernel": ["rbf"],
            "svm__gamma": [1e-4, 1e-3, 'scale', 'auto'],
            "svm__C": [1e-2, 0.1, 1, 10, 100]
        },

        {
            "svm__kernel": ["linear"],
            "svm__C": [1e-2, 0.1, 1, 10, 100]
        },
    ]

    # Grid Search
    model = GridSearchCV(pipe, tuned_params)
    model.fit(x_train, y_train)

    print("Best parameters set: ")
    print()
    print(model.best_params_)
    print()

    means = model.cv_results_["mean_test_score"]
    stds = model.cv_results_["std_test_score"]

    for mean, std, params in zip(means, stds, model.cv_results_["params"]):
        print(f"Mean Accuracy: {mean*100:2.2f}% ± {std*200:2.2f}% for {params}")

    y_true, y_pred = y_test, model.predict(x_test)

    print(classification_report(y_true, y_pred))
