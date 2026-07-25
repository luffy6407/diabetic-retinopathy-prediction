import os
import joblib
import pandas as pd
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetic Retinopathy Prediction",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* Main page */
.block-container {
    max-width: 1350px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

/* Hero */
.hero {
    padding: 2.2rem 2.4rem;
    border-radius: 22px;
    background: linear-gradient(
        135deg,
        rgba(20, 110, 190, 0.12),
        rgba(25, 160, 140, 0.08)
    );
    border: 1px solid rgba(128,128,128,0.20);
    margin-bottom: 1.5rem;
}

.hero h1 {
    margin: 0;
    font-size: 2.55rem;
    font-weight: 750;
}

.hero p {
    margin-top: 0.7rem;
    margin-bottom: 0;
    font-size: 1.05rem;
    opacity: 0.78;
}

/* Page headings */
.page-title {
    font-size: 2.35rem;
    font-weight: 750;
    margin-bottom: 0.2rem;
}

.page-subtitle {
    font-size: 1.03rem;
    opacity: 0.72;
    margin-bottom: 1.6rem;
}

/* Metric cards */
[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 16px;
    padding: 1rem 1.1rem;
    background: rgba(128,128,128,0.025);
}

/* Buttons */
.stButton > button {
    border-radius: 12px;
    min-height: 3rem;
    font-weight: 650;
}

/* Positive prediction */
.positive-result {
    padding: 1.6rem;
    border-radius: 18px;
    border-left: 6px solid #d9534f;
    background: rgba(217,83,79,0.08);
    margin: 1rem 0;
}

.positive-result h2 {
    margin-top: 0.25rem;
    margin-bottom: 0.4rem;
}

/* Negative prediction */
.negative-result {
    padding: 1.6rem;
    border-radius: 18px;
    border-left: 6px solid #2e9d68;
    background: rgba(46,157,104,0.08);
    margin: 1rem 0;
}

.negative-result h2 {
    margin-top: 0.25rem;
    margin-bottom: 0.4rem;
}

/* Result label */
.result-label {
    font-size: 0.80rem;
    font-weight: 700;
    opacity: 0.65;
    letter-spacing: 0.08rem;
}

/* Insight cards */
.insight-card {
    padding: 1.3rem;
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 16px;
    margin-bottom: 1rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,0.18);
}

/* Tables */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "svm_retinopathy_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "scaler.pkl"
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset.csv"
)

EXPECTED_COLUMNS = [
    "ID",
    "age",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
    "prognosis"
]

MODEL_FEATURES = [
    "age",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol"
]

# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

@st.cache_resource
def load_model_artifacts():

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


try:

    model, scaler = load_model_artifacts()

except Exception as error:

    st.error(
        "Unable to load the trained SVM model or StandardScaler. "
        "Make sure svm_retinopathy_model.pkl and scaler.pkl are "
        "in the same folder as app.py."
    )

    st.exception(error)
    st.stop()

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    # Dataset uses semicolon separator
    data = pd.read_csv(
        DATASET_PATH,
        sep=";"
    )

    data.columns = data.columns.str.strip()

    return data


dataset_available = True
dataset_error = None

try:

    df = load_dataset()

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        dataset_available = False

        dataset_error = (
            "Dataset is missing expected columns: "
            + ", ".join(missing_columns)
        )

except Exception as error:

    dataset_available = False
    dataset_error = str(error)
    df = None

# ============================================================
# MODEL PERFORMANCE DATA
# ============================================================

model_results = pd.DataFrame({

    "Model": [
        "SVM",
        "Logistic Regression",
        "XGBoost",
        "KNN",
        "Random Forest",
        "Decision Tree"
    ],

    "Accuracy": [
        0.7708,
        0.7675,
        0.7417,
        0.7300,
        0.7275,
        0.6800
    ],

    "Precision": [
        0.7714,
        0.7864,
        0.7480,
        0.7382,
        0.7324,
        0.6998
    ],

    "Recall": [
        0.7877,
        0.7520,
        0.7504,
        0.7358,
        0.7407,
        0.6613
    ],

    "F1 Score": [
        0.7795,
        0.7688,
        0.7492,
        0.7370,
        0.7365,
        0.6800
    ],

    "ROC-AUC": [
        0.8348,
        0.8415,
        0.8088,
        0.7888,
        0.8134,
        0.6805
    ]
})

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 👁️ DR Predictor")

st.sidebar.caption(
    "Diabetic Retinopathy Prediction System"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Prediction",
        "Dataset & EDA",
        "Model Performance",
        "Project Insights",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Final Model")

st.sidebar.success(
    "Support Vector Machine (SVM)"
)

st.sidebar.markdown("""
**Problem Type:** Binary Classification

**Accuracy:** 77.08%

**Recall:** 78.77%

**F1 Score:** 77.95%

**ROC-AUC:** 83.48%
""")

st.sidebar.markdown("---")

st.sidebar.caption(
    "Educational machine learning project. "
    "Not intended for clinical diagnosis."
)

# ============================================================
# PREDICTION PAGE
# ============================================================

if page == "Prediction":

    st.markdown(
        '<div class="hero">'
        '<h1>👁️ Diabetic Retinopathy Prediction</h1>'
        '<p>Machine learning–based classification using selected patient clinical measurements and a trained Support Vector Machine.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # PERFORMANCE SUMMARY
    # --------------------------------------------------------

    m1, m2, m3, m4 = st.columns(4)
    # --------------------------------------------------------
    # PERFORMANCE SUMMARY
    # --------------------------------------------------------

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Final Model",
        "SVM"
    )

    m2.metric(
        "Accuracy",
        "77.08%"
    )

    m3.metric(
        "Recall",
        "78.77%"
    )

    m4.metric(
        "ROC-AUC",
        "83.48%"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # INPUT SECTION
    # --------------------------------------------------------

    st.subheader(
        "Patient Clinical Information"
    )

    st.caption(
        "Enter the four clinical measurements used by the trained model."
    )

    col1, col2 = st.columns(
        2,
        gap="large"
    )

    with col1:

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=50,
            step=1,
            help="Enter the patient's age in years."
        )

        systolic_bp = st.number_input(
            "Systolic Blood Pressure (mmHg)",
            min_value=50,
            max_value=250,
            value=120,
            step=1,
            help="Enter systolic blood pressure."
        )

    with col2:

        diastolic_bp = st.number_input(
            "Diastolic Blood Pressure (mmHg)",
            min_value=30,
            max_value=150,
            value=80,
            step=1,
            help="Enter diastolic blood pressure."
        )

        cholesterol = st.number_input(
            "Cholesterol",
            min_value=50.0,
            max_value=500.0,
            value=200.0,
            step=1.0,
            help=(
                "Enter cholesterol using the same measurement "
                "convention as the training dataset."
            )
        )

    # --------------------------------------------------------
    # BASIC INPUT VALIDATION
    # --------------------------------------------------------

    valid_input = True

    if systolic_bp <= diastolic_bp:

        st.warning(
            "Please verify the blood pressure values. "
            "Systolic blood pressure should normally be greater "
            "than diastolic blood pressure."
        )

        valid_input = False

    st.markdown("")

    predict_button = st.button(
        "Analyze Patient",
        type="primary",
        use_container_width=True,
        disabled=not valid_input
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if predict_button:

        patient_data = pd.DataFrame(
            [[
                age,
                systolic_bp,
                diastolic_bp,
                cholesterol
            ]],
            columns=MODEL_FEATURES
        )

        try:

            # Apply the exact scaler used during training
            patient_scaled = scaler.transform(
                patient_data
            )

            prediction = model.predict(
                patient_scaled
            )[0]

            # ------------------------------------------------
            # SAFER CLASS HANDLING
            # ------------------------------------------------

            classes = list(model.classes_)

            class_strings = [
                str(value).strip().lower()
                for value in classes
            ]

            # Determine how classes were encoded
            if 0 in classes and 1 in classes:

                no_ret_class = 0
                ret_class = 1

            elif (
                "no_retinopathy" in class_strings
                and "retinopathy" in class_strings
            ):

                no_ret_class = classes[
                    class_strings.index(
                        "no_retinopathy"
                    )
                ]

                ret_class = classes[
                    class_strings.index(
                        "retinopathy"
                    )
                ]

            else:

                st.error(
                    "The saved model contains unexpected class labels: "
                    f"{classes}"
                )

                st.stop()

            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = model.predict_proba(
                    patient_scaled
                )[0]

                probability_map = dict(
                    zip(
                        classes,
                        probabilities
                    )
                )

                no_ret_prob = (
                    probability_map[
                        no_ret_class
                    ] * 100
                )

                ret_prob = (
                    probability_map[
                        ret_class
                    ] * 100
                )

            else:

                no_ret_prob = None
                ret_prob = None

            is_retinopathy = (
                prediction == ret_class
            )

            st.markdown("---")

            st.markdown(
                "## Prediction Result"
            )

            # ------------------------------------------------
            # RESULT CARD
            # ------------------------------------------------

            if is_retinopathy:

                probability_text = (
                    f"{ret_prob:.2f}%"
                    if ret_prob is not None
                    else "Not available"
                )

                st.error("### Retinopathy")

                st.write(
                    "The trained SVM classified the entered "
                    "measurements into the **retinopathy** class."
                )

                st.metric(
                    "Retinopathy Probability",
                    probability_text
                )

            else:

                probability_text = (
                    f"{no_ret_prob:.2f}%"
                    if no_ret_prob is not None
                    else "Not available"
                )

                st.success("### No Retinopathy")

                st.write(
                    "The trained SVM classified the entered "
                    "measurements into the **no-retinopathy** class."
                )

                st.metric(
                    "No Retinopathy Probability",
                    probability_text
                )

            # ------------------------------------------------
            # PROBABILITY DISPLAY
            # ------------------------------------------------

            if (
                ret_prob is not None
                and no_ret_prob is not None
            ):

                st.subheader(
                    "Model Class Probabilities"
                )

                p1, p2 = st.columns(2)

                p1.metric(
                    "No Retinopathy",
                    f"{no_ret_prob:.2f}%"
                )

                p2.metric(
                    "Retinopathy",
                    f"{ret_prob:.2f}%"
                )

                st.progress(
                    max(
                        0,
                        min(
                            100,
                            int(round(ret_prob))
                        )
                    )
                )

                st.caption(
                    "The percentage above represents the SVM model's "
                    "estimated class probability. It is not a clinical "
                    "risk score."
                )

            # ------------------------------------------------
            # INPUT SUMMARY
            # ------------------------------------------------

            with st.expander(
                "View Patient Input Summary"
            ):

                input_summary = pd.DataFrame({

                    "Clinical Feature": [
                        "Age",
                        "Systolic Blood Pressure",
                        "Diastolic Blood Pressure",
                        "Cholesterol"
                    ],

                    "Entered Value": [
                        age,
                        systolic_bp,
                        diastolic_bp,
                        cholesterol
                    ],

                    "Unit": [
                        "Years",
                        "mmHg",
                        "mmHg",
                        "Dataset unit"
                    ]
                })

                st.dataframe(
                    input_summary,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as error:

            st.error(
                "The prediction could not be generated."
            )

            st.exception(error)

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.markdown("---")

    st.warning(
        """
        **Educational Use Only:** This application demonstrates a
        machine learning classification model. The prediction is not
        a medical diagnosis, screening result, or treatment
        recommendation. Clinical assessment should be performed by
        qualified healthcare professionals.
        """
    )

# ============================================================
# DATASET & EDA PAGE
# ============================================================

elif page == "Dataset & EDA":

    st.markdown(
        '<div class="page-title">📊 Dataset & Exploratory Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Interactive overview of the dataset used for model development.'
        '</div>',
        unsafe_allow_html=True
    )

    if not dataset_available:

        st.error(
            "The dataset could not be loaded."
        )

        st.code(
            dataset_error
        )

        st.info(
            "Make sure dataset.csv is in the same folder as app.py."
        )

    else:

        # ----------------------------------------------------
        # DATASET OVERVIEW
        # ----------------------------------------------------

        total_rows = len(df)
        total_columns = df.shape[1]
        missing_values = int(
            df.isnull().sum().sum()
        )
        duplicates = int(
            df.duplicated().sum()
        )

        d1, d2, d3, d4 = st.columns(4)

        d1.metric(
            "Patient Records",
            f"{total_rows:,}"
        )

        d2.metric(
            "Dataset Columns",
            total_columns
        )

        d3.metric(
            "Missing Values",
            missing_values
        )

        d4.metric(
            "Duplicate Rows",
            duplicates
        )

        st.markdown("---")

        # ----------------------------------------------------
        # DATA PREVIEW
        # ----------------------------------------------------

        st.subheader(
            "Dataset Preview"
        )

        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # CLASS DISTRIBUTION
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "Target Class Distribution"
        )

        class_counts = (
            df["prognosis"]
            .astype(str)
            .str.strip()
            .value_counts()
        )

        class_df = (
            class_counts
            .rename_axis("Prognosis")
            .reset_index(name="Patients")
        )

        c1, c2 = st.columns(
            [1, 1.5],
            gap="large"
        )

        with c1:

            st.dataframe(
                class_df,
                use_container_width=True,
                hide_index=True
            )

            ret_count = int(
                class_counts.get(
                    "retinopathy",
                    0
                )
            )

            no_ret_count = int(
                class_counts.get(
                    "no_retinopathy",
                    0
                )
            )

            total_target = (
                ret_count
                + no_ret_count
            )

            if total_target > 0:

                ret_pct = (
                    ret_count
                    / total_target
                    * 100
                )

                no_ret_pct = (
                    no_ret_count
                    / total_target
                    * 100
                )

                st.metric(
                    "Retinopathy",
                    f"{ret_count:,}",
                    f"{ret_pct:.2f}% of dataset"
                )

                st.metric(
                    "No Retinopathy",
                    f"{no_ret_count:,}",
                    f"{no_ret_pct:.2f}% of dataset"
                )

        with c2:

            st.bar_chart(
                class_df.set_index(
                    "Prognosis"
                )
            )

        st.info(
            "The target classes are relatively balanced, which reduces "
            "the risk of model performance being dominated by one class."
        )

        # ----------------------------------------------------
        # DESCRIPTIVE STATISTICS
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "Descriptive Statistics"
        )

        numeric_features = [
            "age",
            "systolic_bp",
            "diastolic_bp",
            "cholesterol"
        ]

        statistics = (
            df[numeric_features]
            .describe()
            .T
            .round(2)
        )

        st.dataframe(
            statistics,
            use_container_width=True
        )

        # ----------------------------------------------------
        # FEATURE DISTRIBUTIONS
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "Feature Distributions"
        )

        selected_feature = st.selectbox(
            "Select a clinical feature",
            numeric_features,
            format_func=lambda x: {
                "age": "Age",
                "systolic_bp": "Systolic Blood Pressure",
                "diastolic_bp": "Diastolic Blood Pressure",
                "cholesterol": "Cholesterol"
            }[x]
        )

        distribution_data = (
            df[[selected_feature]]
            .dropna()
            .reset_index(drop=True)
        )

        st.bar_chart(
            distribution_data[
                selected_feature
            ].value_counts()
            .sort_index()
        )

        # ----------------------------------------------------
        # GROUP COMPARISON
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "Average Clinical Measurements by Prognosis"
        )

        group_means = (
            df.groupby(
                "prognosis"
            )[numeric_features]
            .mean()
            .round(2)
        )

        st.dataframe(
            group_means,
            use_container_width=True
        )

        st.bar_chart(
            group_means
        )

        # ----------------------------------------------------
        # CORRELATION
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "Feature Correlation"
        )

        correlation = (
            df[numeric_features]
            .corr()
            .round(3)
        )

        st.dataframe(
            correlation.style.background_gradient(
                cmap="Blues",
                vmin=-1,
                vmax=1
            ),
            use_container_width=True
        )

        st.caption(
            "Correlation values range from -1 to +1. Values closer "
            "to ±1 indicate stronger linear relationships."
        )

        # ----------------------------------------------------
        # DATA QUALITY
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "Data Quality Summary"
        )

        q1, q2, q3 = st.columns(3)

        q1.metric(
            "Missing Values",
            missing_values
        )

        q2.metric(
            "Duplicate Rows",
            duplicates
        )

        if "ID" in df.columns:

            unique_ids = int(
                df["ID"].nunique()
            )

            q3.metric(
                "Unique Patient IDs",
                f"{unique_ids:,}"
            )

        if (
            missing_values == 0
            and duplicates == 0
        ):

            st.success(
                "No missing values or duplicate rows were detected "
                "in the loaded dataset."
            )

# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="page-title">📈 Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Evaluation and comparison of all classification models.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # FINAL MODEL METRICS
    # --------------------------------------------------------

    st.subheader(
        "Final SVM Performance"
    )

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric(
        "Accuracy",
        "77.08%"
    )

    m2.metric(
        "Precision",
        "77.14%"
    )

    m3.metric(
        "Recall",
        "78.77%"
    )

    m4.metric(
        "F1 Score",
        "77.95%"
    )

    m5.metric(
        "ROC-AUC",
        "83.48%"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.subheader(
        "Model Comparison"
    )

    formatted_results = (
        model_results.copy()
    )

    st.dataframe(
        formatted_results.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "ROC-AUC": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Performance Visualization"
    )

    chart_df = (
        model_results
        .set_index("Model")
    )

    st.bar_chart(
        chart_df
    )

    st.info(
        """
        SVM achieved the highest accuracy, recall and F1-score.
        Logistic Regression achieved the highest precision and
        ROC-AUC. SVM was selected because it provided the strongest
        overall balance for this project's classification objective.
        """
    )

    # --------------------------------------------------------
    # METRIC EXPLANATION
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Evaluation Metrics"
    )

    e1, e2 = st.columns(2)

    with e1:

        st.markdown("""
        **Accuracy**

        Percentage of all test observations classified correctly.

        **Precision**

        Among cases predicted as retinopathy, the proportion that
        were actually retinopathy.
        """)

    with e2:

        st.markdown("""
        **Recall**

        Among actual retinopathy cases, the proportion successfully
        identified by the model.

        **F1 Score**

        Harmonic mean of precision and recall.

        **ROC-AUC**

        Measures the model's ability to distinguish between the two
        classes across classification thresholds.
        """)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "SVM Confusion Matrix"
    )

    confusion_df = pd.DataFrame(
        [
            [439, 144],
            [131, 486]
        ],
        index=[
            "Actual No Retinopathy",
            "Actual Retinopathy"
        ],
        columns=[
            "Predicted No Retinopathy",
            "Predicted Retinopathy"
        ]
    )

    st.dataframe(
        confusion_df.style.background_gradient(
            cmap="Blues"
        ),
        use_container_width=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "True Negatives",
        "439"
    )

    c2.metric(
        "False Positives",
        "144"
    )

    c3.metric(
        "False Negatives",
        "131"
    )

    c4.metric(
        "True Positives",
        "486"
    )

    st.markdown("""
    **Interpretation**

    - **439** no-retinopathy cases were correctly classified.
    - **486** retinopathy cases were correctly classified.
    - **144** no-retinopathy cases were incorrectly classified as retinopathy.
    - **131** retinopathy cases were missed by the model.
    """)

    # --------------------------------------------------------
    # GENERALIZATION CHECK
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Training vs Testing Performance"
    )

    g1, g2, g3 = st.columns(3)

    g1.metric(
        "Training Accuracy",
        "75.06%"
    )

    g2.metric(
        "Testing Accuracy",
        "77.08%"
    )

    g3.metric(
        "Accuracy Difference",
        "-2.02 pp"
    )

    st.info(
        """
        Training and testing accuracy are reasonably close.
        Testing accuracy is approximately 2.02 percentage points
        higher than training accuracy. This comparison does not
        indicate overfitting.
        """
    )

    # --------------------------------------------------------
    # FINAL MODEL
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Why SVM Was Selected"
    )

    st.success(
        """
        **Support Vector Machine (SVM)** was selected as the final model.

        It achieved the highest **accuracy (77.08%)**,
        **recall (78.77%)**, and **F1-score (77.95%)** among the
        evaluated models.

        Logistic Regression produced slightly higher precision and
        ROC-AUC, but SVM provided stronger recall and overall
        classification balance.
        """
    )

# ============================================================
# PROJECT INSIGHTS PAGE
# ============================================================

elif page == "Project Insights":

    st.markdown(
        '<div class="page-title">💡 Project Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Key findings from preprocessing, modelling and evaluation.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # PROJECT SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "Project Summary"
    )

    if dataset_available:

        i1, i2, i3, i4 = st.columns(4)

        i1.metric(
            "Records",
            f"{len(df):,}"
        )

        i2.metric(
            "Model Features",
            "4"
        )

        i3.metric(
            "Models Evaluated",
            "6"
        )

        i4.metric(
            "Final Model",
            "SVM"
        )

    # --------------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Data Quality Findings"
    )

    st.markdown("""
    - The dataset contained no missing values.
    - Duplicate records were checked before modelling.
    - Patient ID was excluded from the predictive features because
      it is an identifier rather than a clinical predictor.
    - The prognosis target was encoded into binary classes.
    - Numerical clinical features were examined for unusual values
      and outliers during EDA.
    """)

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Preprocessing Strategy"
    )

    preprocessing_df = pd.DataFrame({

        "Step": [
            "Identifier Removal",
            "Target Encoding",
            "Train-Test Split",
            "Feature Standardization"
        ],

        "Purpose": [
            "Remove non-predictive patient ID",
            "Convert prognosis into binary classes",
            "Create independent training and testing datasets",
            "Place numerical features on comparable scales"
        ]
    })

    st.dataframe(
        preprocessing_df,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        """
        StandardScaler was fitted using the training features only
        and then applied to the testing data. This avoids using
        information from the test set during preprocessing.
        """
    )

    # --------------------------------------------------------
    # MODEL INSIGHTS
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Model Evaluation Insights"
    )

    st.markdown("""
    - Six classification algorithms were evaluated.
    - SVM achieved the highest overall accuracy.
    - SVM also achieved the highest recall and F1-score.
    - Logistic Regression achieved the highest precision and ROC-AUC.
    - Decision Tree produced the weakest overall performance among
      the evaluated models.
    - The final SVM showed no clear evidence of overfitting based on
      the training-versus-testing accuracy comparison.
    """)

    # --------------------------------------------------------
    # FINAL CONCLUSION
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Final Conclusion"
    )

    st.success(
        """
        The project successfully developed a binary classification
        pipeline for predicting the diabetic retinopathy class from
        four clinical measurements.

        Support Vector Machine provided the strongest overall
        combination of accuracy, recall and F1-score and was therefore
        selected for deployment.

        The trained model and preprocessing scaler were saved and
        integrated into this Streamlit application for real-time
        prediction.
        """
    )

# ============================================================
# ABOUT PROJECT PAGE
# ============================================================

elif page == "About Project":

    st.markdown(
        '<div class="page-title">ℹ️ About the Project</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Diabetic Retinopathy Prediction using Machine Learning'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # OBJECTIVE
    # --------------------------------------------------------

    st.subheader(
        "Project Objective"
    )

    st.markdown(
        """
        The objective of this project is to build, evaluate and deploy
        a machine learning classification system that predicts the
        diabetic retinopathy class using selected patient clinical
        measurements.
        """
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Prediction Features"
    )

    feature_information = pd.DataFrame({

        "Feature": [
            "Age",
            "Systolic Blood Pressure",
            "Diastolic Blood Pressure",
            "Cholesterol"
        ],

        "Model Column": [
            "age",
            "systolic_bp",
            "diastolic_bp",
            "cholesterol"
        ]
    })

    st.dataframe(
        feature_information,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Machine Learning Workflow"
    )

    workflow = pd.DataFrame({

        "Stage": [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7"
        ],

        "Process": [
            "Data Understanding",
            "Exploratory Data Analysis",
            "Data Preprocessing",
            "Feature Standardization",
            "Model Building",
            "Model Evaluation & Selection",
            "Streamlit Deployment"
        ]
    })

    st.dataframe(
        workflow,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Models Evaluated"
    )

    model_col1, model_col2 = st.columns(2)

    with model_col1:

        st.markdown("""
        - Logistic Regression
        - Decision Tree
        - Random Forest
        """)

    with model_col2:

        st.markdown("""
        - K-Nearest Neighbors (KNN)
        - Support Vector Machine (SVM)
        - XGBoost
        """)

    # --------------------------------------------------------
    # FINAL MODEL
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Final Model"
    )

    st.success(
        """
        **Support Vector Machine (SVM)**

        Testing Accuracy: **77.08%**

        Precision: **77.14%**

        Recall: **78.77%**

        F1 Score: **77.95%**

        ROC-AUC: **83.48%**
        """
    )

    # --------------------------------------------------------
    # TECH STACK
    # --------------------------------------------------------

    st.subheader(
        "Technology Stack"
    )

    st.markdown(
        """
        **Python • Pandas • Scikit-learn • XGBoost •
        Streamlit • Joblib • Jupyter Notebook**
        """
    )

    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "Project Limitations"
    )

    st.markdown("""
    - Predictions depend on the patterns represented in the training dataset.
    - The model uses only four clinical features.
    - Model probabilities should not be interpreted as clinical risk estimates.
    - External clinical validation has not been performed.
    - The application is intended for educational machine learning
      demonstration rather than real-world medical decision-making.
    """)

    st.warning(
        """
        **Important:** This application is an educational machine
        learning project. It is not a medical device and must not be
        used to diagnose diabetic retinopathy or make treatment
        decisions.
        """
    )