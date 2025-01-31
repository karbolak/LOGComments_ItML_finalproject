from ml.hyperparameter_tuning import tune_hyperparameters
from ml.hyperparameter_choice import HPARAMS
from ml.trainer import train_final_model, evaluate_model
from ml.data_processing import preprocess_datasets
from ml.data_manipulation import save_predictions, select_and_load_datasets


def run_pipeline(directory="datasets"):
    """Runs an automated pipeline for Logistic Regression."""
    # Step 1: Select and load datasets
    train_dataset, test_dataset, target_column = select_and_load_datasets(
        directory)

    # Step 2: Preprocess datasets
    X_train, X_test, y_train, y_test = preprocess_datasets(train_dataset,
                                                           test_dataset,
                                                           target_column)

    # Step 3: Import hyperparameter grid
    HPARAM_GRID = HPARAMS

    # Step 4: Perform hyperparameter tuning + cross-validation
    best_hparams = tune_hyperparameters(X_train, y_train, HPARAM_GRID)

    # Step 5: Train final model
    final_model = train_final_model(X_train, y_train, best_hparams)

    # Step 6: Evaluate the model
    predictions = evaluate_model(final_model, X_test, y_test)

    # Step 7: Save predictions
    save_predictions(test_dataset, predictions, target_column, directory)


if __name__ == "__main__":
    run_pipeline()
