
import numpy as np
import pandas as pd


class DataValidation:

    def __init__(self, X_train, X_test, y_train=None, y_test=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def validateAll(self):
        assert len(self.X_train) > 0, "Training data kosong"
        assert len(self.X_test) > 0, "Testing data kosong"
        assert len(self.X_train) == len(self.y_train) if self.y_train is not None else True
        assert len(self.X_test) == len(self.y_test) if self.y_test is not None else True
        if isinstance(self.X_train, pd.DataFrame):
            assert self.X_train.columns.equals(self.X_test.columns), "Kolom train dan test berbeda"
        assert not pd.DataFrame(self.X_train).isnull().any().any(), ("Training memiliki missing value")
        assert not pd.DataFrame(self.X_test).isnull().any().any(), ("Testing memiliki missing value")
        assert np.isfinite(np.asarray(self.X_train, dtype=float)).all(), "Training memiliki nilai infinite"
        assert np.isfinite(np.asarray(self.X_test, dtype=float)).all(), "Testing memiliki nilai infinite"
        if self.y_train is not None:
            assert pd.Series(self.y_train).nunique() >= 2, ("Target training hanya memiliki satu kelas")

        print("Data Validation: PASSED")

        return True
