
import numpy as np
import pandas as pd


class feature:

    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def createAgeYear(self):

        self.df['age_year'] = (
            self.df['age'] / 365.25
        ).astype(int)

        return self

    def createBMI(self):

        height_meter = self.df['height'] / 100

        self.df['BMI'] = np.where(
            height_meter > 0,
            self.df['weight'] / height_meter ** 2,
            np.nan
        )

        return self

    def createPulsePressure(self):

        self.df['pulse_pressure'] = (
            self.df['ap_hi'] - self.df['ap_lo']
        )

        return self

    def ageBinning(self):

        self.df['age_group'] = pd.cut(
            self.df['age_year'],
            bins=[0, 30, 45, 60, 120],
            labels=[
                'Muda',
                'Dewasa',
                'Paruh baya',
                'Lansia'
            ]
        )

        return self

    def getFeatures(self):

        return self.df
