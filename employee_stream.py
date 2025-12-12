import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# Base directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Dashboard",
    layout="wide"
)

st.title("🧑‍💼 Employee Attrition Interactive Dashboard")
st.write("Select an employee from the dropdown to view all feature & target details and model prediction.")

# ---------------------------------------------
# 1. LOAD DATA & MODEL
# ---------------------------------------------
@st.cache_data
def load_data():
    raw_path = os.path.join(BASE_DIR, "Employee-Attrition - Employee-Attrition.csv")
    pre_path = os.path.join(BASE_DIR, "employee_attrition_preprocessed.csv")

    raw_df = pd.read_csv(raw_path)
    pre_df = pd.read_csv(pre_path)

    return raw_df, pre_df



@st.cache_resource
def load_model_and_preprocessor():
    try:
        model_path = os.path.join(BASE_DIR, "best_knn_model.joblib")
        preproc_path = os.path.join(BASE_DIR, "knn_preprocessor.joblib")

        model = joblib.load(model_path)
        preprocessor = joblib.load(preproc_path)
    except Exception as e:
        st.error(f"Error loading model or preprocessor: {e}")
        model, preprocessor = None, None
    return model, preprocessor




raw_df, pre_df = load_data()
model, preprocessor = load_model_and_preprocessor()

# ---------------------------------------------
# 2. IDENTIFY EMPLOYEE ID COLUMN
# ---------------------------------------------
possible_id_cols = ["EmployeeNumber", "SerialNumber", "employee_number", "ID", "Id"]
id_col = None

for c in possible_id_cols:
    if c in raw_df.columns:
        id_col = c
        break

# If not found, create a serial ID
if id_col is None:
    id_col = "EmployeeID"
    raw_df[id_col] = raw_df.index + 1
    if id_col not in pre_df.columns:
        pre_df[id_col] = pre_df.index + 1

# Ensure preprocessed df also has same ID column
if id_col not in pre_df.columns and id_col in raw_df.columns:
    pre_df[id_col] = raw_df[id_col].values

# ---------------------------------------------
# 3. SIDEBAR - EMPLOYEE DROPDOWN
# ---------------------------------------------
st.sidebar.header("🔎 Employee Selection")

employee_ids = sorted(raw_df[id_col].unique().tolist())

selected_id = st.sidebar.selectbox(
    f"Select {id_col}",
    options=employee_ids
)

# ---------------------------------------------
# 4. OVERALL DATASET METRICS
# ---------------------------------------------
st.subheader("📊 Overall Dataset Summary")

col1, col2, col3, col4 = st.columns(4)

total_employees = raw_df.shape[0]
col1.metric("Total Employees", total_employees)

if "Attrition" in raw_df.columns:
    attr_col = raw_df["Attrition"]
    if attr_col.dtype == "object":
        attr_rate = (attr_col == "Yes").mean()
    else:
        attr_rate = (attr_col == 1).mean()
    col2.metric("Attrition Rate", f"{attr_rate * 100:.1f} %")
else:
    col2.metric("Attrition Rate", "N/A")

if "MonthlyIncome" in raw_df.columns:
    col3.metric("Avg Monthly Income", f"{raw_df['MonthlyIncome'].mean():.0f}")
else:
    col3.metric("Avg Monthly Income", "N/A")

if "YearsAtCompany" in raw_df.columns:
    col4.metric("Avg Years at Company", f"{raw_df['YearsAtCompany'].mean():.1f}")
else:
    col4.metric("Avg Years at Company", "N/A")

# ---------------------------------------------
# 5. EMPLOYEE DETAIL VIEW (ALL FEATURES + TARGET)
# ---------------------------------------------
st.markdown("---")
st.subheader(f"🧾 Full Details for {id_col}: `{selected_id}`")

emp_row_raw = raw_df[raw_df[id_col] == selected_id]

if emp_row_raw.empty:
    st.error("No data found for selected employee in original dataset.")
else:
    # Show all columns (features + Attrition) transposed
    st.markdown("### 🗂 Original Employee Data (Features + Attrition)")
    st.dataframe(emp_row_raw.T, use_container_width=True)

# ---------------------------------------------
# 6. MODEL PREDICTION FOR SELECTED EMPLOYEE
# ---------------------------------------------
st.markdown("---")
st.subheader("🤖 Attrition Prediction from KNN Model")

if (model is not None) and (preprocessor is not None):
    # Get matching record from preprocessed df
    emp_row_pre = pre_df[pre_df[id_col] == selected_id]

    if emp_row_pre.empty:
        st.warning("No matching record for this employee in the preprocessed dataset. Cannot predict.")
    else:
        # Separate X and y (if Attrition present)
        target_col = "Attrition" if "Attrition" in emp_row_pre.columns else None

        if target_col:
            X_emp = emp_row_pre.drop(columns=[target_col])
            true_attr_raw = emp_row_pre[target_col].iloc[0]
        else:
            X_emp = emp_row_pre.copy()
            true_attr_raw = None

        try:
            # Transform features
            X_emp_trans = preprocessor.transform(X_emp)

            # Predict probability and class
            proba = model.predict_proba(X_emp_trans)[0, 1]
            pred_class = 1 if proba >= 0.5 else 0

            pred_label = "Yes" if pred_class == 1 else "No"

            if true_attr_raw is None:
                true_label = "Unknown"
            else:
                if isinstance(true_attr_raw, str):
                    true_label = true_attr_raw
                else:
                    true_label = "Yes" if true_attr_raw == 1 else "No"

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Attrition", pred_label)
            c2.metric("Attrition Probability", f"{proba * 100:.1f} %")
            c3.metric("Actual Attrition (from data)", true_label)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.info("Model or preprocessor not loaded. Make sure 'best_knn_model.joblib' "
            "and 'knn_preprocessor.joblib' are in this folder.")

# ---------------------------------------------
# 7. SIMPLE INTERACTIVE VISUALS
# ---------------------------------------------
st.markdown("---")
st.subheader("📈 Visual Overview")

col_left, col_right = st.columns(2)

# Attrition by Department
if "Department" in raw_df.columns and "Attrition" in raw_df.columns:
    with col_left:
        st.markdown("#### Attrition Count by Department")
        attr_counts = raw_df.groupby(["Department", "Attrition"]).size().reset_index(name="Count")
        fig_dept = px.bar(
            attr_counts,
            x="Department",
            y="Count",
            color="Attrition",
            barmode="group"
        )
        st.plotly_chart(fig_dept, use_container_width=True)

# Income distribution, highlight selected employee
if "MonthlyIncome" in raw_df.columns:
    with col_right:
        st.markdown("#### Monthly Income Distribution")
        fig_income = px.histogram(raw_df, x="MonthlyIncome", nbins=30, marginal="box")
        st.plotly_chart(fig_income, use_container_width=True)

        if not emp_row_raw.empty:
            sel_income = emp_row_raw["MonthlyIncome"].iloc[0]
            st.markdown(f"**Selected Employee's Monthly Income:** `{sel_income}`")

st.markdown("✅ Use the dropdown in the sidebar to switch between employees.")
