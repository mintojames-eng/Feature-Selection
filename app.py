import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer
)

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    StratifiedKFold
)

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import (
    VarianceThreshold,
    SequentialFeatureSelector
)

from sklearn.decomposition import FactorAnalysis

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Feature Selection ML Lab",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Feature Selection and Reduction")
st.subheader("Advanced Machine Learning – Lab Exercise 3")

st.write(
    """
    This application demonstrates different feature selection and
    feature reduction techniques for Linear Regression and
    Logistic Regression.
    """
)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("Model Selection")

model_type = st.sidebar.selectbox(
    "Choose Model",
    [
        "Linear Regression",
        "Logistic Regression"
    ]
)

method = st.sidebar.selectbox(
    "Choose Feature Selection Method",
    [
        "Original Features",
        "Low Variance Filter",
        "High Correlation Filter",
        "Factor Analysis",
        "Forward Selection",
        "Backward Elimination"
    ]
)

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_regression_data():

    data = fetch_california_housing(
        as_frame=True
    )

    df = data.frame.copy()

    X = df.drop(
        "MedHouseVal",
        axis=1
    )

    y = df["MedHouseVal"]

    return X, y


@st.cache_data
def load_classification_data():

    data = load_breast_cancer()

    X = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    y = pd.Series(
        data.target,
        name="target"
    )

    return X, y, data.target_names


# ==========================================================
# LINEAR REGRESSION
# ==========================================================

if model_type == "Linear Regression":

    X, y = load_regression_data()

    st.header("Linear Regression")

    st.info(
        "Dataset: California Housing"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Instances",
        X.shape[0]
    )

    col2.metric(
        "Original Features",
        X.shape[1]
    )

    col3.metric(
        "Missing Values",
        int(X.isnull().sum().sum())
    )

    with st.expander("View Dataset"):

        st.dataframe(
            X.head(10)
        )

    # ------------------------------------------------------
    # TRAIN TEST SPLIT
    # ------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # ------------------------------------------------------
    # SCALING
    # ------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X.columns
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X.columns
    )

    # ------------------------------------------------------
    # FEATURE SELECTION
    # ------------------------------------------------------

    if method == "Original Features":

        Xtr = X_train_scaled
        Xte = X_test_scaled

        selected_features = list(
            X.columns
        )

    elif method == "Low Variance Filter":

        selector = VarianceThreshold(
            threshold=0.01
        )

        selector.fit(
            X_train_scaled
        )

        selected_features = list(
            X.columns[
                selector.get_support()
            ]
        )

        Xtr = X_train_scaled[
            selected_features
        ]

        Xte = X_test_scaled[
            selected_features
        ]

    elif method == "High Correlation Filter":

        corr_matrix = X_train_scaled.corr().abs()

        upper = corr_matrix.where(
            np.triu(
                np.ones(corr_matrix.shape),
                k=1
            ).astype(bool)
        )

        remove_features = [
            column
            for column in upper.columns
            if any(
                upper[column] > 0.80
            )
        ]

        selected_features = [
            column
            for column in X.columns
            if column not in remove_features
        ]

        Xtr = X_train_scaled[
            selected_features
        ]

        Xte = X_test_scaled[
            selected_features
        ]

    elif method == "Factor Analysis":

        fa = FactorAnalysis(
            n_components=4,
            random_state=42
        )

        Xtr_array = fa.fit_transform(
            X_train_scaled
        )

        Xte_array = fa.transform(
            X_test_scaled
        )

        selected_features = [
            "Factor 1",
            "Factor 2",
            "Factor 3",
            "Factor 4"
        ]

        Xtr = pd.DataFrame(
            Xtr_array,
            columns=selected_features
        )

        Xte = pd.DataFrame(
            Xte_array,
            columns=selected_features
        )

    else:

        model = LinearRegression()

        direction = (
            "forward"
            if method == "Forward Selection"
            else "backward"
        )

        selector = SequentialFeatureSelector(
            model,
            n_features_to_select=4,
            direction=direction,
            scoring="r2",
            cv=5,
            n_jobs=-1
        )

        selector.fit(
            X_train_scaled,
            y_train
        )

        selected_features = list(
            X.columns[
                selector.get_support()
            ]
        )

        Xtr = X_train_scaled[
            selected_features
        ]

        Xte = X_test_scaled[
            selected_features
        ]

    # ------------------------------------------------------
    # DISPLAY SELECTED FEATURES
    # ------------------------------------------------------

    st.subheader("Selected Features")

    st.write(
        f"**Method:** {method}"
    )

    st.write(
        f"**Number of selected features:** "
        f"{len(selected_features)}"
    )

    st.write(selected_features)

    # ------------------------------------------------------
    # MODEL
    # ------------------------------------------------------

    model = LinearRegression()

    model.fit(
        Xtr,
        y_train
    )

    predictions = model.predict(
        Xte
    )

    # ------------------------------------------------------
    # METRICS
    # ------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    st.subheader("Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "MAE",
        f"{mae:.4f}"
    )

    c2.metric(
        "MSE",
        f"{mse:.4f}"
    )

    c3.metric(
        "RMSE",
        f"{rmse:.4f}"
    )

    c4.metric(
        "R² Score",
        f"{r2:.4f}"
    )

    # ------------------------------------------------------
    # CROSS VALIDATION
    # ------------------------------------------------------

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_scores = cross_val_score(
        LinearRegression(),
        Xtr,
        y_train,
        cv=kfold,
        scoring="r2"
    )

    st.subheader("5-Fold Cross Validation")

    c1, c2 = st.columns(2)

    c1.metric(
        "Mean CV R²",
        f"{cv_scores.mean():.4f}"
    )

    c2.metric(
        "Std CV R²",
        f"{cv_scores.std():.4f}"
    )

    # ------------------------------------------------------
    # ACTUAL VS PREDICTED
    # ------------------------------------------------------

    st.subheader("Actual vs Predicted")

    fig, ax = plt.subplots()

    ax.scatter(
        y_test,
        predictions
    )

    ax.set_xlabel(
        "Actual Values"
    )

    ax.set_ylabel(
        "Predicted Values"
    )

    ax.set_title(
        "Actual vs Predicted"
    )

    st.pyplot(fig)


# ==========================================================
# LOGISTIC REGRESSION
# ==========================================================

else:

    X, y, target_names = load_classification_data()

    st.header("Logistic Regression")

    st.info(
        "Dataset: Breast Cancer Wisconsin"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Instances",
        X.shape[0]
    )

    col2.metric(
        "Original Features",
        X.shape[1]
    )

    col3.metric(
        "Missing Values",
        int(X.isnull().sum().sum())
    )

    with st.expander("View Dataset"):

        st.dataframe(
            X.head(10)
        )

    # ------------------------------------------------------
    # TRAIN TEST SPLIT
    # ------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ------------------------------------------------------
    # SCALING
    # ------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X.columns
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X.columns
    )

    # ------------------------------------------------------
    # FEATURE SELECTION
    # ------------------------------------------------------

    if method == "Original Features":

        Xtr = X_train_scaled
        Xte = X_test_scaled

        selected_features = list(
            X.columns
        )

    elif method == "Low Variance Filter":

        selector = VarianceThreshold(
            threshold=0.01
        )

        selector.fit(
            X_train_scaled
        )

        selected_features = list(
            X.columns[
                selector.get_support()
            ]
        )

        Xtr = X_train_scaled[
            selected_features
        ]

        Xte = X_test_scaled[
            selected_features
        ]

    elif method == "High Correlation Filter":

        corr_matrix = X_train_scaled.corr().abs()

        upper = corr_matrix.where(
            np.triu(
                np.ones(corr_matrix.shape),
                k=1
            ).astype(bool)
        )

        remove_features = [
            column
            for column in upper.columns
            if any(
                upper[column] > 0.80
            )
        ]

        selected_features = [
            column
            for column in X.columns
            if column not in remove_features
        ]

        Xtr = X_train_scaled[
            selected_features
        ]

        Xte = X_test_scaled[
            selected_features
        ]

    elif method == "Factor Analysis":

        fa = FactorAnalysis(
            n_components=10,
            random_state=42
        )

        Xtr_array = fa.fit_transform(
            X_train_scaled
        )

        Xte_array = fa.transform(
            X_test_scaled
        )

        selected_features = [
            f"Factor {i+1}"
            for i in range(10)
        ]

        Xtr = pd.DataFrame(
            Xtr_array,
            columns=selected_features
        )

        Xte = pd.DataFrame(
            Xte_array,
            columns=selected_features
        )

    else:

        model = LogisticRegression(
            max_iter=5000
        )

        direction = (
            "forward"
            if method == "Forward Selection"
            else "backward"
        )

        selector = SequentialFeatureSelector(
            model,
            n_features_to_select=10,
            direction=direction,
            scoring="accuracy",
            cv=5,
            n_jobs=-1
        )

        selector.fit(
            X_train_scaled,
            y_train
        )

        selected_features = list(
            X.columns[
                selector.get_support()
            ]
        )

        Xtr = X_train_scaled[
            selected_features
        ]

        Xte = X_test_scaled[
            selected_features
        ]

    # ------------------------------------------------------
    # SELECTED FEATURES
    # ------------------------------------------------------

    st.subheader("Selected Features")

    st.write(
        f"**Method:** {method}"
    )

    st.write(
        f"**Number of selected features:** "
        f"{len(selected_features)}"
    )

    st.write(selected_features)

    # ------------------------------------------------------
    # MODEL
    # ------------------------------------------------------

    model = LogisticRegression(
        max_iter=5000
    )

    model.fit(
        Xtr,
        y_train
    )

    predictions = model.predict(
        Xte
    )

    # ------------------------------------------------------
    # METRICS
    # ------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    st.subheader("Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

    c2.metric(
        "Precision",
        f"{precision:.4f}"
    )

    c3.metric(
        "Recall",
        f"{recall:.4f}"
    )

    c4.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

    # ------------------------------------------------------
    # CROSS VALIDATION
    # ------------------------------------------------------

    skfold = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_scores = cross_val_score(
        LogisticRegression(
            max_iter=5000
        ),
        Xtr,
        y_train,
        cv=skfold,
        scoring="accuracy"
    )

    st.subheader(
        "5-Fold Cross Validation"
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Mean CV Accuracy",
        f"{cv_scores.mean():.4f}"
    )

    c2.metric(
        "Std CV Accuracy",
        f"{cv_scores.std():.4f}"
    )

    # ------------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------------

    st.subheader(
        "Confusion Matrix"
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    fig, ax = plt.subplots()

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names
    )

    disp.plot(
        ax=ax
    )

    st.pyplot(fig)


# ==========================================================
# SELF LEARNING COMPONENT
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.header(
    "⭐ Self Learning"
)

run_self_learning = st.sidebar.checkbox(
    "Run L1 Regularization"
)

if run_self_learning:

    st.header(
        "⭐ Self Learning: L1 Regularization"
    )

    st.write(
        """
        L1 regularization can automatically perform feature selection
        by forcing some model coefficients to zero.
        """
    )

    if model_type == "Logistic Regression":

        X, y, target_names = load_classification_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        C_values = [
            0.01,
            0.1,
            1,
            10
        ]

        results = []

        for C in C_values:

            model = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=C,
                max_iter=5000
            )

            model.fit(
                X_train_scaled,
                y_train
            )

            predictions = model.predict(
                X_test_scaled
            )

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            selected = np.sum(
                model.coef_[0] != 0
            )

            results.append({
                "C": C,
                "Accuracy": accuracy,
                "Selected Features": selected
            })

        results_df = pd.DataFrame(
            results
        )

        st.dataframe(
            results_df
        )

        fig, ax = plt.subplots()

        ax.plot(
            results_df["C"],
            results_df["Accuracy"],
            marker="o"
        )

        ax.set_xscale(
            "log"
        )

        ax.set_xlabel(
            "C"
        )

        ax.set_ylabel(
            "Accuracy"
        )

        ax.set_title(
            "L1 Regularization vs Accuracy"
        )

        st.pyplot(fig)

    else:

        st.info(
            """
            L1 regularization is demonstrated here using
            Logistic Regression because it naturally performs
            coefficient-based feature selection.
            """
        )

# ==========================================================
# FOOTER
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.write(
    "MAI511-2 Advanced Machine Learning"
)

st.sidebar.write(
    "Lab Exercise 3"
)
