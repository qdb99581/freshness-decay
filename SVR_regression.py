# Prepare for SVR
from sklearn.svm import SVR

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

    for pred in preds:

        pass
