# Aprendizaje supervisado — Titanic

Predicción de la supervivencia de los pasajeros del Titanic comparando 11 algoritmos de clasificación.

| Cuaderno | Contenido |
|---|---|
| `1_clean.ipynb` | Limpieza: faltantes (edad por título, puerto por moda), cabina → cubierta, atípicos. Genera `data/titanic_clean.csv` |
| `2_EDA.ipynb` | Análisis exploratorio: sexo, clase, edad, tarifa, familia, título, correlaciones |
| `3_feature_engineering.ipynb` | Variables nuevas (`FamilySize`, `IsAlone`, `FarePerPerson`, `IsChild`, …), one-hot, relevancia. Genera `data/titanic_features.csv` |
| `4_machine_learning.ipynb` | 11 modelos con validación cruzada (5 pliegues) y prueba: exactitud, precisión, sensibilidad, especificidad, AUC-ROC. Selección e interpretación |
| `5_pipeline.ipynb` | Pipeline completo (datos crudos → predicción), GridSearchCV y guardado en `models/titanic_pipeline.joblib` |
| `6_validacion_robusta.ipynb` | Validación cruzada repetida 10×5, intervalos de confianza y prueba t corregida (Nadeau-Bengio) para justificar el mejor modelo |
| `7_umbral_decision.ipynb` | Ajuste del umbral de decisión (sensibilidad vs especificidad) con predicciones fuera de pliegue; umbral elegido 0.32 |
| `8_interpretacion_shap.ipynb` | Interpretación con SHAP: importancia global, dirección de efectos, interacciones clase × sexo y explicación de aciertos y errores individuales |
| `app.py` | App web Flask y API JSON para predecir |

## Resultados

| Modelo | AUC-ROC (CV) | Exactitud (prueba) | AUC-ROC (prueba) |
|---|---|---|---|
| **Gradient Boosting** | **0.895** | 0.810 | 0.851 |
| CatBoost | 0.894 | 0.793 | 0.846 |
| XGBoost | 0.894 | 0.804 | 0.836 |
| Random Forest | 0.890 | 0.799 | 0.842 |
| LightGBM | 0.890 | 0.788 | 0.817 |
| AdaBoost | 0.877 | 0.804 | 0.839 |
| SVC | 0.865 | 0.799 | 0.851 |
| Regresión Logística | 0.863 | 0.827 | 0.867 |
| Decision Tree | 0.854 | 0.810 | 0.817 |
| KNeighbors | 0.847 | 0.771 | 0.806 |
| GaussianNB | 0.816 | 0.737 | 0.781 |

Mejor modelo: **Gradient Boosting** (mayor AUC en validación cruzada y el más estable). Variables más influyentes: título (sobre todo "Mr"), sexo, clase, tarifa y edad.

## Cómo correrlo

```bash
pip install -r requirements.txt
jupyter lab          # ejecutar los cuadernos en orden 1 → 5
python app.py        # http://127.0.0.1:5000
```
