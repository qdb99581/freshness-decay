# Custom Modules
from sklearn.svm import SVR
import utils
import trainer

if __name__ == "__main__":
    opt = utils.Config()

    # Prepare data and train_test_split
    x_train_data, y_train_data = utils.import_data(
        data_root_path=opt.data_root_path,
        selected_bands=opt.selected_bands,
        train_for_regression=opt.regression,
        derivative=opt.derivative,
        mushroom_class=opt.mushroom_class,
        normalize="zscore",
        shuffle=True,
    )

    # Warning from sklearn documentation:
    # If training data is not C-contihuous, the error may occur.
    print(
        f"Check if the input data is C-contiguous: {x_train_data.flags['C_CONTIGUOUS']}")

    # Import evaluation data
    x_all_data, y_all_data = utils.import_data(
        data_root_path=opt.data_root_path,
        selected_bands=opt.selected_bands,
        train_for_regression=False,
        derivative=opt.derivative,
        mushroom_class=opt.mushroom_class,
        normalize="zscore",
        shuffle=False,
    )

    model_id = "lSVM3"
    params = opt.svm_layout[model_id]

    if len(params) == 1:
        model = SVR(
            C=params[0],
            kernel='linear'
        )
    else:
        model = SVR(
            C=params[0],
            gamma=params[1],
            kernel='rbf'
        )
    
    model.fit(x_train_data, y_train_data)
    preds = model.predict(x_all_data)
    trainer.eval_regression(opt.mushroom_class, preds, model_id)
    