import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DataVisualization:
    def __init__(self, dataframe):
        self.df = dataframe

    def histogram(self):
        numeric = self.df.select_dtypes(include="number")
        for col in numeric.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(data=self.df,x=col,bins=30,kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()

    def boxplot(self):
        numerical_columns = self.df.select_dtypes(include="number").columns
        for column in numerical_columns:
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=self.df, x=column)
            plt.title(f"Boxplot of {column}")
            plt.xlabel(column)
            plt.tight_layout()
            plt.show()

    def countplot(self):
            categorical_columns = self.df.select_dtypes(
            include=["object", "category", "bool"]).columns
            for column in categorical_columns:
                plt.figure(figsize=(8, 5))
                sns.countplot(data=self.df, x=column)
                plt.title(f"Countplot of {column}")
                plt.xlabel(column)
                plt.ylabel("Count")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

    def correlationHeatmap(self):
            correlation = self.df.corr(numeric_only=True)
            plt.figure(figsize=(12, 8))
            sns.heatmap(correlation,annot=True,cmap="coolwarm",fmt=".2f",linewidths=0.5,vmin=-1,vmax=1)
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.show()

    def targetDistribution(self, target="cardio"):
            if target not in self.df.columns:
                raise ValueError(f"Target '{target}' tidak ditemukan")
            plt.figure(figsize=(8, 5))
            sns.countplot(data=self.df,x=target)
            plt.title(f"Target Distribution: {target}")
            plt.xlabel(target)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.show()

