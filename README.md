# Predicción de abandono escolar en educación superior

Este proyecto implementa una solución analítica para predecir la situación académica de estudiantes de educación superior (`Dropout`, `Enrolled`, `Graduate`) a partir de información académica, socioeconómica y de contexto.

## Contenido del repositorio

- `app.py`: aplicación en Streamlit con dashboard interactivo y simulador de predicción.
- `data.csv`: dataset utilizado para el análisis (versión descargada de Kaggle).
- `Proyecto_Final_Dropout_MartinMarquezUrbina.ipynb`: notebook de análisis con limpieza, EDA y modelo de Machine Learning.
- `modelo_dropout.pkl`: modelo entrenado (Random Forest) guardado con `joblib`.
- `requirements.txt`: dependencias necesarias para ejecutar la app.

## Cómo ejecutar la app localmente

1. Clonar o descargar este repositorio.
2. Instalar dependencias y ejecutar la app:

   ```bash
   pip install -r requirements.txt
   streamlit run app.py
```
## Autor

Martín Márquez Urbina

