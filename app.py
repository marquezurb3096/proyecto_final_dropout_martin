import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(
    page_title="Abandono escolar – Dashboard",
    layout="wide"
)

sns.set(style="whitegrid")

# =========================
# Funciones auxiliares
# =========================
@st.cache_data
def load_data():
    # Asegúrate que el nombre coincida con tu archivo
    df = pd.read_csv("data.csv", sep=";")
    return df

@st.cache_resource
def load_model():
    clf = joblib.load("modelo_dropout.pkl")
    return clf

df = load_data()
clf = load_model()

X = df.drop("Target", axis=1)
y = df["Target"]

# Columnas numéricas y categóricas
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# Valores por defecto para el simulador
num_defaults = X[numeric_features].median()

# Puede pasar que no detecte columnas categóricas; evitamos que truene
if len(categorical_features) > 0:
    cat_defaults = X[categorical_features].mode().iloc[0]
else:
    import pandas as pd
    cat_defaults = pd.Series(dtype=object)


# =========================
# Navegación
# =========================
st.sidebar.title("Navegación")
pagina = st.sidebar.radio(
    "Ir a:",
    ["Inicio", "Exploración de datos", "Simulador de predicción"]
)

# =========================
# Página: INICIO
# =========================
if pagina == "Inicio":
    st.title("Predicción de abandono escolar en educación superior")

    st.write(
        """
        Este dashboard utiliza un modelo de **clasificación** para estimar si un estudiante
        tiene mayor probabilidad de **abandonar**, **seguir inscrito** o **graduarse**.
        Los datos provienen de una institución de educación superior e incluyen información
        académica, socioeconómica y de desempeño en los primeros semestres.
        """
    )

    col1, col2, col3 = st.columns(3)

    total_estudiantes = len(df)
    tasa_dropout = (df["Target"] == "Dropout").mean() * 100
    tasa_graduate = (df["Target"] == "Graduate").mean() * 100

    col1.metric("Total de estudiantes", total_estudiantes)
    col2.metric("% que abandonan", f"{tasa_dropout:.1f}%")
    col3.metric("% que se gradúan", f"{tasa_graduate:.1f}%")

    st.subheader("Distribución general de la situación académica")

    fig, ax = plt.subplots()
    df["Target"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Distribución de la situación académica (Target)")
    ax.set_xlabel("Target")
    ax.set_ylabel("Número de estudiantes")
    st.pyplot(fig)

    st.markdown(
        """
        **Resumen del modelo (Random Forest):**
        - Problema: clasificación multiclase (`Dropout`, `Enrolled`, `Graduate`).
        - Métrica principal: *accuracy* ≈ **0.77** en el conjunto de prueba.
        - Las variables con mayor importancia son unidades curriculares aprobadas y calificación
          en los dos primeros semestres, así como la edad de ingreso y el cumplimiento de pagos.
        """
    )

# =========================
# Página: EXPLORACIÓN
# =========================
elif pagina == "Exploración de datos":
    st.title("Exploración de datos")

    st.write("Puedes filtrar por **curso** para ver cómo cambia la distribución.")

    cursos = ["Todos"] + sorted(df["Course"].unique().tolist())
    curso_sel = st.selectbox("Curso", cursos)

    if curso_sel == "Todos":
        df_filtro = df.copy()
    else:
        df_filtro = df[df["Course"] == curso_sel]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribución de Target")
        fig1, ax1 = plt.subplots()
        df_filtro["Target"].value_counts().plot(kind="bar", ax=ax1)
        ax1.set_title("Situación académica")
        ax1.set_xlabel("Target")
        ax1.set_ylabel("Número de estudiantes")
        st.pyplot(fig1)

    with col2:
        st.subheader("Situación académica por género")
        fig2, ax2 = plt.subplots()
        sns.countplot(data=df_filtro, x="Gender", hue="Target", ax=ax2)
        ax2.set_title("Target por género")
        ax2.set_xlabel("Género")
        ax2.set_ylabel("Número de estudiantes")
        st.pyplot(fig2)

    st.subheader("Tasa de abandono por curso (Top 15)")

    dropout_rate = (
        df.groupby("Course")["Target"]
        .apply(lambda x: (x == "Dropout").mean())
        .sort_values(ascending=False)
    )

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    dropout_rate.head(15).plot(kind="bar", ax=ax3)
    ax3.set_title("Proporción de estudiantes que abandonan por curso")
    ax3.set_ylabel("Proporción de abandono")
    ax3.set_xlabel("Curso")
    st.pyplot(fig3)

# =========================
# Página: SIMULADOR
# =========================
elif pagina == "Simulador de predicción":
    st.title("Simulador de riesgo de abandono")

    st.write(
        """
        Completa la información del estudiante para estimar la probabilidad de que
        **abandone**, continúe **inscrito** o se **gradúe**.  
        Algunas variables ya vienen prellenadas con valores típicos del conjunto de datos.
        """
    )

    # ---------- Entradas del usuario ----------
    st.subheader("Datos académicos principales")

    input_data = {}

    vars_academicas = [
        "Age at enrollment",
        "Admission grade",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Tuition fees up to date",
    ]

    for col in vars_academicas:
        if col not in X.columns:
            continue
        if col in numeric_features:
            min_val = float(X[col].min())
            max_val = float(X[col].max())
            default = float(num_defaults[col])
            input_data[col] = st.slider(
                col,
                min_value=min_val,
                max_value=max_val,
                value=default
            )
        else:
            opciones = sorted(df[col].unique().tolist())
            default = cat_defaults[col]
            idx = opciones.index(default) if default in opciones else 0
            input_data[col] = st.selectbox(col, opciones, index=idx)

    st.subheader("Datos socioeconómicos y de contexto")

    vars_contexto = [
        "Gender",
        "Course",
        "Marital status",
        "Application mode",
        "Daytime/evening attendance",
        "Scholarship holder",
        "Debtor",
        "Mother's qualification",
        "Father's qualification",
    ]

    for col in vars_contexto:
        if col not in X.columns:
            continue
        opciones = sorted(df[col].unique().tolist())
        default = cat_defaults[col] if col in categorical_features else opciones[0]
        idx = opciones.index(default) if default in opciones else 0
        input_data[col] = st.selectbox(col, opciones, index=idx)

    # Rellenar el resto de columnas con valores por defecto
    for col in X.columns:
        if col not in input_data:
            if col in numeric_features:
                input_data[col] = num_defaults[col]
            else:
                input_data[col] = cat_defaults[col]

    if st.button("Calcular predicción"):
        entrada = pd.DataFrame([input_data])
        pred = clf.predict(entrada)[0]
        proba = clf.predict_proba(entrada)[0]
        proba_dict = {clase: round(p * 100, 1) for clase, p in zip(clf.classes_, proba)}

        st.subheader("Resultado del modelo")
        st.write(f"**Predicción:** {pred}")
        st.write("**Probabilidades:**")
        st.write(proba_dict)

        st.info(
            "Este resultado es una estimación basada en datos históricos. "
            "No sustituye el análisis individual del caso, pero puede ayudar a priorizar "
            "estudiantes en riesgo para ofrecerles apoyo oportuno."
        )
