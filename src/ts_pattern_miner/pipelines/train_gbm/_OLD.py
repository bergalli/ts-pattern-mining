import lightgbm as lgb


def init_model(model_name, hyperparams):
    if model_name == "gbrt":
        model_instance = lgb.DaskLGBMRegressor(**hyperparams)
    elif model_name == "lstm":
        model_instance = False
    else:
        raise NotImplementedError

    return model_instance


def train_model(model_instance, nb_folds, dask_client, X_train_folds, y_train_folds, X_valid_folds, y_valid_folds):
    for fold_id in range(nb_folds):
        import optuna
        from optuna.integration.lightgbm import LightGBMTuner, LightGBMTunerCV
        import lightgbm as lgb
        import joblib

        X_train = X_train_folds[fold_id]
        y_train = y_train_folds[fold_id]
        X_valid = X_valid_folds[fold_id]
        y_valid = y_valid_folds[fold_id]
        # train_pool = Pool(data=X_train, label=y_train, feature_names=feature_names)
        # validation_pool = Pool(data=X_valid, label=y_valid, feature_names=feature_names)
        dtrain = lgb.Dataset(X_train, y_train[:, 0])
        dvalid = lgb.Dataset(X_valid, y_valid[:, 0])
        LightGBMTuner(train_set=dtrain, valid_sets=dvalid)
        LightGBMTunerCV
        with joblib.parallel_backend("dask", scatter=[X_train, y_train]):
            model_instance.fit(X_train, y_train[:, 0])

        study = optuna.create_study(study_name='distributed-ray', storage='sqlite:///optuna_db/ray.db',
                                    direction="maximize")
        with joblib.parallel_backend("ray", n_jobs=-1):
            def objective(trial):
                x = trial.suggest_uniform('x', -10, 10)
                return (x - 2) ** 2

            study.optimize(objective, n_trials=100)

        optuna.create_study(study_name='distributed-dask', storage='sqlite:///optuna_db/dask.db')
        njobs = 6
        futures = []
        for i in range(njobs):
            def spawn_study(_model_instance, _X_train, _y_train):
                def objective(trial):
                    _model_instance.fit(_X_train, _y_train)
                    return max(_model_instance.predict(_X_train))

                study = optuna.load_study(study_name='distributed-example', storage='sqlite:///optuna_db/dask.db')
                study.optimize(objective, n_trials=100)

            future = dask_client.submit(spawn_study, **dict(_model_instance=model_instance,
                                                            _X_train=X_train,
                                                            _y_train=y_train))
            futures.append(future)
        pass
    return model_instance


def eval_model(model_trained, targets_names, nb_folds, X_test_folds, y_test_folds):
    return 0
