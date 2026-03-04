"""
Streamlit app for Wine Cultivar Classification
---------------------------------------------

This Streamlit application allows users to explore the UCI Wine dataset
and predict the cultivar (class) of a wine sample based on its
chemical properties.  It trains a random forest classifier on the
dataset at runtime and provides sliders in the sidebar for users to
adjust feature values.  The predicted cultivar is displayed along
with a correlation heatmap to visualise relationships among features.

To run the app locally, install the dependencies listed in
``requirements.txt`` and execute ``streamlit run app.py``.
"""

import streamlit as st
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier


def main():
    st.set_page_config(page_title="Wine Cultivar Classification", layout="wide")
    st.title("🍷 Wine Cultivar Classification")

    # Load Wine dataset
    data = load_wine()
    df = pd.concat([
        pd.DataFrame(data.data, columns=data.feature_names),
        pd.Series(data.target, name="class"),
    ], axis=1)

    st.write("This app uses a Random Forest classifier to predict the cultivar of a wine based on its chemical analysis.")

    # Display preview of data
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Train random forest on entire dataset (for demonstration purposes)
    X = df.drop(columns=["class"])
    y = df["class"]
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    # Sidebar for user inputs
    st.sidebar.header("Input Wine Features")
    input_features = {}
    for feature in data.feature_names:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())
        input_features[feature] = st.sidebar.slider(
            label=feature.replace("_", " ").title(),
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            step=(max_val - min_val) / 100.0,
            format="%.3f",
        )

    # Prediction
    input_df = pd.DataFrame([input_features])
    prediction = model.predict(input_df)[0]
    cultivar_map = {0: "Cultivar 0", 1: "Cultivar 1", 2: "Cultivar 2"}
    st.sidebar.markdown("---")
    st.sidebar.subheader("Predicted Cultivar")
    st.sidebar.write(f"**{cultivar_map[prediction]}**")

    # Display correlation heatmap if available
    st.subheader("Feature Correlation Heatmap")
    heatmap_path = "figures/correlation_heatmap.png"
    try:
        st.image(heatmap_path, caption="Correlation Heatmap", use_column_width=True)
    except Exception:
        st.warning("Heatmap image not found. Run the training script to generate figures.")


if __name__ == "__main__":
    main()