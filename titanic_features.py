"""Transformaciones compartidas del proyecto Titanic.

Este módulo lo usan el notebook 5_pipeline.ipynb y app.py, para que el
modelo guardado aplique exactamente la misma limpieza e ingeniería de
variables que se hizo en los notebooks 1 y 3.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TITLE_MAP = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Rare", "Countess": "Rare", "the Countess": "Rare", "Dona": "Rare",
    "Capt": "Rare", "Col": "Rare", "Don": "Rare", "Dr": "Rare",
    "Major": "Rare", "Rev": "Rare", "Sir": "Rare", "Jonkheer": "Rare",
}

NUM_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare",
                "FamilySize", "IsAlone", "HasCabin", "FarePerPerson", "IsChild"]
CAT_FEATURES = ["Sex", "Embarked", "Title", "Deck", "AgeGroup"]


def extract_title(name: str) -> str:
    """'Braund, Mr. Owen Harris' -> 'Mr' (agrupando títulos poco comunes)."""
    if not isinstance(name, str) or "," not in name:
        return "Rare"
    raw = name.split(",", 1)[1].split(".", 1)[0].strip()
    return TITLE_MAP.get(raw, "Rare")


def extract_deck(cabin) -> str:
    """'C85' -> 'C'; sin cabina -> 'U' (unknown). T se agrupa con U por ser 1 caso."""
    if not isinstance(cabin, str) or cabin.strip() == "":
        return "U"
    deck = cabin.strip()[0]
    return "U" if deck == "T" else deck


class TitanicFeatures(BaseEstimator, TransformerMixin):
    """Limpieza + ingeniería de variables a partir de las columnas originales.

    Aprende en fit() las medianas de edad por título y la mediana de tarifa
    usando SOLO los datos de entrenamiento (evita fuga de información).

    Mejoras opcionales (cuaderno 9):
    - split_rare: los títulos raros de MUJER (Lady, Countess, Dr mujer) pasan a
      "Mrs", así "Rare" queda solo para hombres (Dr, Rev, Col, Major...).
    - fare_zero: las tarifas en 0 (boletos especiales) se marcan con FareZero=1
      y se imputan con la mediana de tarifa de su clase.
    """

    def __init__(self, split_rare=False, fare_zero=False):
        self.split_rare = split_rare
        self.fare_zero = fare_zero

    def _titles(self, df):
        t = df["Name"].map(extract_title)
        if getattr(self, "split_rare", False):
            t = t.where(~((t == "Rare") & (df["Sex"] == "female")), "Mrs")
        return t

    def fit(self, X, y=None):
        df = X.copy()
        df["Title"] = self._titles(df)
        self.age_by_title_ = df.groupby("Title")["Age"].median().to_dict()
        self.age_global_ = float(df["Age"].median())
        self.fare_median_ = float(df["Fare"].median())
        pagada = df[df["Fare"] > 0]
        self.fare_by_class_ = pagada.groupby("Pclass")["Fare"].median().to_dict()
        mode = df["Embarked"].mode()
        self.embarked_mode_ = mode.iloc[0] if len(mode) else "S"
        return self

    def transform(self, X):
        df = X.copy()
        df["Title"] = self._titles(df)
        df["Age"] = df.apply(
            lambda r: r["Age"] if pd.notna(r["Age"])
            else self.age_by_title_.get(r["Title"], self.age_global_), axis=1)
        extra = []
        if getattr(self, "fare_zero", False):
            df["FareZero"] = (df["Fare"] == 0).astype(int)
            imput = df["Pclass"].map(self.fare_by_class_).fillna(self.fare_median_)
            df["Fare"] = df["Fare"].where(df["Fare"] != 0, imput)
            extra.append("FareZero")
        df["Fare"] = df["Fare"].fillna(self.fare_median_)
        df["Embarked"] = df["Embarked"].fillna(self.embarked_mode_)
        df["Deck"] = df["Cabin"].map(extract_deck)
        df["HasCabin"] = (df["Deck"] != "U").astype(int)
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
        df["IsChild"] = (df["Age"] < 13).astype(int)
        df["AgeGroup"] = pd.cut(df["Age"], bins=[-1, 12, 18, 35, 60, 120],
                                labels=["Niño", "Adolescente", "Joven", "Adulto", "Mayor"]).astype(str)
        return df[NUM_FEATURES + extra + CAT_FEATURES]

    def get_num_features(self):
        return NUM_FEATURES + (["FareZero"] if getattr(self, "fare_zero", False) else [])
