
import pandas as pd
from pathlib import Path

from src.Feature.feature import feature
from src.spliting.spliting import DataSpliting
from src.preprocessing.preprocessing import Preprocessing
from src.Feature.featureSelection import FeatureSelection
from src.preprocessing.dataValidation import DataValidation
from src.model.modelSelection import ModelSelection
from src.model.hyperparameterTuning import HyperparameterTuning

from src.model.finalEvaluation import FinalModelEvaluation


if __name__ == "__main__":
    df = pd.read_csv(Path(__file__).parent / "data" / "cardio_cleaned.csv")

    df = (feature(df).createAgeYear().createBMI().createPulsePressure().ageBinning().dropAge().getFeatures())
    print(df.head(10))
    print(df.columns)

    X = df.drop(columns=["cardio", "id"])
    y = df["cardio"]

    X_train, X_test, y_train, y_test = (DataSpliting(X, y).spliting())

    preprocessing = Preprocessing(X_train, X_test)
    preprocessing.buildPipeline()
    X_train_processed, X_test_processed = (preprocessing.getProcessedData())

    feature_selection = FeatureSelection(X_train_processed,X_test_processed,preprocessing.feature_names).selectFeatures(y_train, k=10)
    X_train, X_test = feature_selection.getSelectedFeatures()


    DataValidation(X_train, X_test, y_train, y_test).validateAll()

    # 7. Results
    print("\nSelected Features:")
    print(feature_selection.getSelectedFeatureNames())
    print("\nFeature Scores:")
    print(feature_selection.getFeatureScores())
    print("\nTraining Shape:", X_train.shape)
    print("Testing Shape:", X_test.shape)
    
    selection = ModelSelection(X_train,y_train)
    results = selection.compareModels()
    print(results)
    best_model_name = results.iloc[0]["Model"]
    best_model = selection.models[best_model_name]


    tuning = HyperparameterTuning(X_train=X_train,y_train=y_train)

    best_model = tuning.tune()
    evaluation = FinalModelEvaluation(model=best_model,X_test=X_test,y_test=y_test)
    results = evaluation.evaluate()

    y_pred = best_model.predict(X_test)

    hasil = pd.DataFrame({"Aktual": y_test.values,"Prediksi": y_pred})
    hasil["Benar"] = hasil["Aktual"] == hasil["Prediksi"]
    print(hasil.head(20))