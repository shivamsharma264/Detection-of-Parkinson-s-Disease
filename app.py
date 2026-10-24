"""Educational Streamlit demo for the UCI Parkinson's voice dataset."""

from pathlib import Path

import pandas as pd
import streamlit as st
import joblib


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "submission" / "parkinsons_models.joblib"

st.set_page_config(page_title="Parkinson's Voice Demo", page_icon="🎙️")
st.title("Parkinson's Disease Detection — Educational Demo")
st.error(
    "NOT A MEDICAL DIAGNOSTIC TOOL. This demo must not be used for diagnosis, "
    "treatment, or any medical decision. Consult a qualified healthcare "
    "professional for health concerns."
)
st.write(
    "This demonstration uses a machine-learning model trained on a small public "
    "voice-measurement dataset. Its output is a model prediction, not a clinical "
    "assessment."
)

if not MODEL_PATH.is_file():
    st.info(
        "No trained model is available yet. Download the real dataset as described "
        "in `data/README.md`, then run `python train_model.py` from the project "
        "folder."
    )
    st.stop()

with MODEL_PATH.open("rb") as model_file:
    artifact = joblib.load(model_file)

feature_columns = artifact["feature_columns"]
models = artifact["models"]
selected_model_name = st.selectbox("Choose a trained model", sorted(models))
st.caption(
    "Enter voice-measurement values in the same units and format as the UCI "
    "dataset. The default values are placeholders, not patient data."
)

values: dict[str, float] = {}
left_column, right_column = st.columns(2)
for index, feature_name in enumerate(feature_columns):
    column = left_column if index % 2 == 0 else right_column
    with column:
        values[feature_name] = st.number_input(
            feature_name,
            value=0.0,
            format="%.6f",
            help="Placeholder default; replace with a compatible measured value.",
        )

if st.button("Generate educational model prediction", type="primary"):
    input_frame = pd.DataFrame([values], columns=feature_columns)
    prediction = int(models[selected_model_name].predict(input_frame)[0])
    if prediction == 1:
        st.warning(
            "The model predicted class 1 for these inputs. This is not a diagnosis "
            "and must not guide medical decisions."
        )
    else:
        st.info(
            "The model predicted class 0 for these inputs. This does not rule out "
            "Parkinson's disease and must not guide medical decisions."
        )