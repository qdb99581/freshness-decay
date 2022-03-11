# Custom Modules
import utils
import trainer

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

    # Import evaluation data
    x_all_data, y_all_data = utils.import_data(
        data_root_path=opt.data_root_path,
        selected_bands=opt.selected_bands,
        regression=False,
        derivative=opt.derivative,
        mushroom_class=opt.mushroom_class,
        normalize="zscore",
        shuffle=False,
    )

    # Grid search by training on day 0 and day 28
    rbf_param_set = tuned_params[0]
    linear_param_set = tuned_params[1]
    # RBF
    for gamma in rbf_param_set["svr__gamma"]:
        for C in rbf_param_set["svr__C"]:
            model = trainer.build_SVR(kernel='rbf', gamma=gamma, C=C)
            model.fit(x_train_data, y_train_data)
            preds = model.predict(x_all_data)
            trainer.eval_regression(
                opt.mushroom_class, preds, gamma, C)

    # Linear
    for C in linear_param_set["svr__C"]:
        model = trainer.build_SVR(kernel='linear', C=C)
        model.fit(x_train_data, y_train_data)
        preds = model.predict(x_all_data)
        trainer.eval_regression(opt.mushroom_class, preds, C)
