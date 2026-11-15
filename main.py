
import pandas as pd
from pathlib import Path

from src.Feature.feature import feature
from src.spliting.spliting import DataSpliting
from src.preprocessing.preprocessing import Preprocessing
from src.Feature.featureSelection import FeatureSelection
from src.preprocessing.dataValidation import DataValidation
from src.model.modelSelection import ModelSelection
from src.model.modelTrening import ModelTraining
from src.model.hyperparameterTuning import HyperparameterTuning

from src.model.finalEvaluation import FinalModelEvaluation


if __name__ == "__main__":

    # 1. Load Data
    df = pd.read_csv(
        Path(__file__).parent / "data" / "heart_cleaned.csv",
        sep=";"
    )
    # 2. Feature Engineering
    df = (
        feature(df)
        .createAgeYear()
        .createBMI()
        .createPulsePressure()
        .ageBinning()
        .df
    )

    print(df.columns)

    # 3. Data Splitting
    X = df.drop(columns="cardio")
    y = df["cardio"]

    X_train, X_test, y_train, y_test = (
        DataSpliting(X, y).spliting()
    )

    # 4. Preprocessing
    preprocessing = Preprocessing(X_train, X_test)

    preprocessing.buildPipeline()

    X_train_processed, X_test_processed = (
        preprocessing.getProcessedData()
    )

    # 5. Feature Selection
    selection = FeatureSelection(
        X_train_processed,
        X_test_processed,
        preprocessing.feature_names
    ).selectFeatures(y_train, k=10)

    X_train, X_test = selection.getSelectedFeatures()

    # 6. Data Validation
    DataValidation(
        X_train, X_test, y_train, y_test
    ).validateAll()

    # 7. Results
    print("\nSelected Features:")
    print(selection.getSelectedFeatureNames())

    print("\nFeature Scores:")
    print(selection.getFeatureScores())

    print("\nTraining Shape:", X_train.shape)
    print("Testing Shape:", X_test.shape)

    selection = ModelSelection(
    X_train,
    y_train
    )

    results = selection.compareModels()

    print(results)

    best_model_name = results.iloc[0]["Model"]

    best_model = selection.models[best_model_name]

    # 1. Model Training
    training = ModelTraining(
        model=best_model,
        X_train=X_train,
        y_train=y_train
    )

    trained_model = training.train()


    # 2. Hyperparameter Tuning
    tuning = HyperparameterTuning(
        X_train=X_train,
        y_train=y_train
    )

    grid_search = tuning.tune()


    # 3. Best Model
    best_model = grid_search.best_estimator_

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest CV Score:")
    print(grid_search.best_score_)


    # 4. Final Evaluation
    evaluation = FinalModelEvaluation(
        model=best_model,
        X_test=X_test,
        y_test=y_test
    )

    results = evaluation.evaluate()
