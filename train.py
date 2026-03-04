"""
Wine Cultivar Classification Project
===================================

This script implements an end‑to‑end machine learning pipeline for the **Wine**
dataset from the UCI Machine Learning Repository.  The Wine dataset is the
result of a chemical analysis of wines grown in the same region of Italy but
derived from three different cultivars.  It contains **13 continuous
features** (e.g., ``alcohol``, ``malic acid``, ``color intensity`` and
``proline``) and a **categorical target** indicating which of the three
cultivars the wine belongs to.  There are **178 instances** in the dataset
and no missing values.  The dataset is provided under the CC BY 4.0
license.

The goal of this project is to train machine learning models that can predict
the cultivar of a wine based on its chemical properties.  The pipeline
includes data loading, exploratory data analysis (EDA), model training, and
evaluation.  We also visualise correlations between features and examine
feature importances for tree‑based models.

Steps
-----
1. **Load the Data**:  Use scikit‑learn's built‑in function to load the
   Wine dataset into a pandas DataFrame.  The 13 numeric features and
   categorical target (labels 0–2) are combined into a single table.
2. **Exploratory Data Analysis (EDA)**:  Display basic statistics, check
   class distribution, and plot a correlation heatmap to understand
   relationships between variables.
3. **Data Splitting and Preprocessing**:  Split the data into training and
   test sets.  Standardize features for models that benefit from
   normalization (e.g., logistic regression).
4. **Model Training**:  Fit several classifiers – logistic regression,
   random forest, and gradient boosting – using scikit‑learn.  Evaluate
   performance using accuracy and weighted F1 score on the test set.
5. **Feature Importance**:  Compute and plot feature importances from the
   random forest and gradient boosting models to understand which
   variables most influence the predictions.

The script saves generated figures into the ``figures/`` directory and
classification reports into the ``reports/`` directory, and prints
evaluation metrics to the console.  This end‑to‑end workflow can be
highlighted on a resume to demonstrate proficiency in data preprocessing,
machine learning, and result interpretation.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
# Use a non‑interactive backend for matplotlib to avoid remote logging
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.datasets import load_wine
from PIL import Image, ImageDraw, ImageFont


def load_data() -> pd.DataFrame:
    """Load the Wine dataset using scikit‑learn.

    Returns
    -------
    pd.DataFrame
        DataFrame containing features and target label.
    """
    dataset = load_wine()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="class")
    df = pd.concat([X, y], axis=1)
    return df


def engineer_target(df: pd.DataFrame) -> pd.DataFrame:
    """Pass‑through for datasets that already contain a categorical target."""
    return df.copy()


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Plot and save a correlation heatmap of the features."""
    corr_matrix = df.drop(columns=[col for col in ["class", "quality", "quality_label"] if col in df.columns]).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def train_models(X_train, X_test, y_train, y_test) -> dict:
    """Train multiple classifiers and evaluate them on the test set.

    Returns a dictionary containing fitted models and their evaluation metrics.
    """
    results = {}
    # Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    log_reg = LogisticRegression(max_iter=500)
    log_reg.fit(X_train_scaled, y_train)
    y_pred_lr = log_reg.predict(X_test_scaled)
    results["logistic_regression"] = {
        "model": log_reg,
        "scaler": scaler,
        "accuracy": accuracy_score(y_test, y_pred_lr),
        "f1": f1_score(y_test, y_pred_lr, average="weighted"),
        "report": classification_report(y_test, y_pred_lr, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred_lr),
    }
    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results["random_forest"] = {
        "model": rf,
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "f1": f1_score(y_test, y_pred_rf, average="weighted"),
        "report": classification_report(y_test, y_pred_rf, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred_rf),
        "feature_importances": rf.feature_importances_,
    }
    # Gradient Boosting
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    results["gradient_boosting"] = {
        "model": gb,
        "accuracy": accuracy_score(y_test, y_pred_gb),
        "f1": f1_score(y_test, y_pred_gb, average="weighted"),
        "report": classification_report(y_test, y_pred_gb, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred_gb),
        "feature_importances": gb.feature_importances_,
    }
    return results


def draw_bar_chart_pil(features: list, importances: list, output_path: Path, title: str) -> None:
    """Draw a simple horizontal bar chart using Pillow and save as PNG."""
    width = 800
    bar_height = 30
    bar_spacing = 15
    margin_top = 80
    margin_left = 200
    margin_right = 80
    margin_bottom = 40
    num_bars = len(features)
    height = margin_top + num_bars * (bar_height + bar_spacing) + margin_bottom
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        font_labels = ImageFont.truetype("DejaVuSans.ttf", 16)
    except IOError:
        font_title = ImageFont.load_default()
        font_labels = ImageFont.load_default()
    bbox = font_title.getbbox(title)
    title_width, title_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - title_width) / 2, 20), title, fill="black", font=font_title)
    max_imp = max(importances)
    for idx, (feat, imp) in enumerate(zip(features, importances)):
        y_top = margin_top + idx * (bar_height + bar_spacing)
        bar_length = (width - margin_left - margin_right) * (imp / max_imp)
        x0 = margin_left
        y0 = y_top
        x1 = margin_left + bar_length
        y1 = y_top + bar_height
        bar_color = (76, 114, 176)
        draw.rectangle([x0, y0, x1, y1], fill=bar_color)
        feat_bbox = font_labels.getbbox(feat)
        feat_height = feat_bbox[3] - feat_bbox[1]
        draw.text((10, y_top + (bar_height - feat_height) / 2), feat, fill="black", font=font_labels)
        imp_text = f"{imp:.4f}"
        imp_bbox = font_labels.getbbox(imp_text)
        text_width, text_height = imp_bbox[2] - imp_bbox[0], imp_bbox[3] - imp_bbox[1]
        draw.text((x1 + 5, y_top + (bar_height - text_height) / 2), imp_text, fill="black", font=font_labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main():
    df = load_data()
    print(f"Loaded dataset shape: {df.shape}")
    print(df.head())
    df = engineer_target(df)
    print("Class distribution:")
    print(df["class"].value_counts())
    heatmap_path = Path("figures/correlation_heatmap.png")
    plot_correlation_heatmap(df, heatmap_path)
    print(f"Correlation heatmap saved to {heatmap_path}")
    X = df.drop(columns=["class"])
    y = df["class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    results = train_models(X_train, X_test, y_train, y_test)
    for name, metrics in results.items():
        print(f"\n{name.upper()} MODEL PERFORMANCE:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Weighted F1 Score: {metrics['f1']:.4f}")
    feature_names = list(X.columns)
    rf_importances = results["random_forest"]["feature_importances"]
    gb_importances = results["gradient_boosting"]["feature_importances"]
    rf_top = sorted(zip(feature_names, rf_importances), key=lambda x: x[1], reverse=True)
    gb_top = sorted(zip(feature_names, gb_importances), key=lambda x: x[1], reverse=True)
    print("\nTop features (Random Forest):")
    for feat, imp in rf_top[:5]:
        print(f" {feat}: {imp:.4f}")
    print("\nTop features (Gradient Boosting):")
    for feat, imp in gb_top[:5]:
        print(f" {feat}: {imp:.4f}")
    rf_plot_path = Path("figures/rf_feature_importance.png")
    draw_bar_chart_pil([feat for feat, _ in rf_top], [imp for _, imp in rf_top], rf_plot_path, title="Random Forest Feature Importances")
    print(f"Random Forest feature importance plot saved to {rf_plot_path}")
    gb_plot_path = Path("figures/gb_feature_importance.png")
    draw_bar_chart_pil([feat for feat, _ in gb_top], [imp for _, imp in gb_top], gb_plot_path, title="Gradient Boosting Feature Importances")
    print(f"Gradient Boosting feature importance plot saved to {gb_plot_path}")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    for name, metrics in results.items():
        report_path = reports_dir / f"{name}_report.csv"
        report_dict = metrics["report"]
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(report_path)
        print(f"Classification report for {name} saved to {report_path}")


if __name__ == "__main__":
    main()