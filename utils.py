import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor, XGBClassifier
import shap
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    r2_score, mean_absolute_error, accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# 1. LOAD & PREPARE DATA – only the given dataset
# ------------------------------------------------------------
def load_and_prepare():
    """Load Excel, clean column names, engineer business metrics."""
    df = pd.read_excel("Sample data.xlsx")  # relative path – place file in project root
    df.columns = df.columns.str.strip()     # remove leading/trailing spaces

    # Clean zeros (replace 0 with NaN to avoid division issues)
    df["Units Sold"] = df["Units Sold"].replace(0, np.nan)
    df["Sale Price"] = df["Sale Price"].replace(0, np.nan)
    df["Gross Sales"] = df["Gross Sales"].replace(0, np.nan)
    df["Manufacturing Price"] = df["Manufacturing Price"].replace(0, np.nan)

    # Unit economics
    df["Revenue_per_Unit"] = df["Sales"] / df["Units Sold"]
    df["Cost_per_Unit"] = df["COGS"] / df["Units Sold"]
    df["Profit_per_Unit"] = df["Profit"] / df["Units Sold"]

    # Margins & efficiency
    df["Gross_Margin"] = np.where(df["Sales"] > 0, df["Profit"] / df["Sales"], 0)
    df["Discount_Rate"] = np.where(df["Gross Sales"] > 0, df["Discounts"] / df["Gross Sales"], 0)
    df["Manufacturing_Efficiency"] = np.where(
        df["Manufacturing Price"] > 0,
        df["Sale Price"] / df["Manufacturing Price"],
        0
    )

    # Date features
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Month_Name"] = df["Date"].dt.strftime("%B")

    # Flags
    df["Is_Loss"] = (df["Profit"] < 0).astype(int)
    df["High_Discount"] = (df["Discount_Rate"] > df["Discount_Rate"].median()).astype(int)

    return df

# Load once – this DataFrame will be used throughout the app
df = load_and_prepare()

# ------------------------------------------------------------
# 2. KPI METRICS – high‑level numbers for dashboards
# ------------------------------------------------------------
def get_kpi_metrics(data=df):
    """Return a dictionary of key performance indicators."""
    return {
        'total_sales': float(data['Sales'].sum()),
        'total_profit': float(data['Profit'].sum()),
        'total_units': float(data['Units Sold'].sum()),
        'avg_profit_margin': float((data['Profit'].sum() / data['Sales'].sum()) * 100),
        'avg_discount_rate': float(data['Discount_Rate'].mean() * 100),
        'total_products': int(data['Product'].nunique()),
        'profitable_products': int(data[data['Profit'] > 0]['Product'].nunique()),
        'loss_making_products': int(data[data['Profit'] < 0]['Product'].nunique())
    }

# ------------------------------------------------------------
# 3. STRATEGIC INSIGHTS – data‑driven bullet points for executives
# ------------------------------------------------------------
def get_strategic_insights(data=df):
    """Generate actionable insights from the dataset (no hardcoding)."""
    insights = []

    # Best segment by total profit
    seg_profit = data.groupby('Segment')['Profit'].sum().sort_values(ascending=False)
    insights.append(f"🏆 Top Segment: {seg_profit.index[0]} (${seg_profit.iloc[0]:,.0f})")

    # Optimal discount band (by average profit)
    disc_avg = data.groupby('Discount Band')['Profit'].mean().dropna().sort_values(ascending=False)
    if not disc_avg.empty:
        insights.append(f"🎯 Best Discount Band: {disc_avg.index[0]} (avg profit ${disc_avg.iloc[0]:,.0f})")

    # Top country by profit
    country_profit = data.groupby('Country')['Profit'].sum().sort_values(ascending=False)
    insights.append(f"🌍 Top Country: {country_profit.index[0]} (${country_profit.iloc[0]:,.0f})")

    # Most and least profitable products
    prod_profit = data.groupby('Product')['Profit'].sum().sort_values(ascending=False)
    insights.append(f"📦 Best Product: {prod_profit.index[0]} (${prod_profit.iloc[0]:,.0f})")
    insights.append(f"⚠️ Needs Review: {prod_profit.index[-1]} (${prod_profit.iloc[-1]:,.0f})")

    # Strongest quarter
    q_profit = data.groupby('Quarter')['Profit'].sum().sort_values(ascending=False)
    insights.append(f"📅 Strongest Quarter: Q{q_profit.index[0]}")

    return insights

# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------
numeric_features = ['Units Sold', 'Manufacturing Price',
                    'Discount_Rate', 'Manufacturing_Efficiency']
categorical_features = ['Segment']

target_reg = 'Profit'
target_clf = 'Is_Loss'

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

X = df[numeric_features + categorical_features].copy()
X['Discount_Rate'] = X['Discount_Rate'].fillna(0)
X['Manufacturing_Efficiency'] = X['Manufacturing_Efficiency'].fillna(1)

y_reg = df[target_reg]
y_clf = df[target_clf]

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf,
    test_size=0.25,
    random_state=42,
    stratify=y_clf
)

# ------------------------------------------------------------
# XGBOOST REGRESSOR
# ------------------------------------------------------------
reg_pipeline = Pipeline([
    ('prep', preprocessor),
    ('xgb', XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8
    ))
])

reg_pipeline.fit(X_train, y_reg_train)

y_reg_pred = reg_pipeline.predict(X_test)

reg_r2 = r2_score(y_reg_test, y_reg_pred)
reg_mae = mean_absolute_error(y_reg_test, y_reg_pred)

# ------------------------------------------------------------
# XGBOOST CLASSIFIER (IMBALANCE FIXED)
# ------------------------------------------------------------
scale_pos_weight = (len(y_clf_train) - sum(y_clf_train)) / sum(y_clf_train)

clf_pipeline = Pipeline([
    ('prep', preprocessor),
    ('xgb', XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    ))
])

clf_pipeline.fit(X_train, y_clf_train)

# ------------------------------------------------------------
# EVALUATION (REAL METRICS)
# ------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

clf_acc = cross_val_score(clf_pipeline, X, y_clf, cv=cv, scoring='accuracy').mean()
clf_f1 = cross_val_score(clf_pipeline, X, y_clf, cv=cv, scoring='f1').mean()
clf_precision = cross_val_score(clf_pipeline, X, y_clf, cv=cv, scoring='precision').mean()
clf_recall = cross_val_score(clf_pipeline, X, y_clf, cv=cv, scoring='recall').mean()
clf_auc = cross_val_score(clf_pipeline, X, y_clf, cv=cv, scoring='roc_auc').mean()  # 🔥 NEW

# Test set predictions
y_clf_pred = clf_pipeline.predict(X_test)
y_clf_prob = clf_pipeline.predict_proba(X_test)[:, 1]

clf_cm = confusion_matrix(y_clf_test, y_clf_pred)

# ------------------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------------------
xgb_model = reg_pipeline.named_steps['xgb']
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(cat_feature_names)

importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

# ------------------------------------------------------------
# RESIDUALS
# ------------------------------------------------------------
residual_df = pd.DataFrame({
    'Actual': y_reg_test,
    'Predicted': y_reg_pred,
    'Residual': y_reg_test - y_reg_pred
}).sample(min(200, len(y_reg_test)), random_state=42)

# ------------------------------------------------------------
# PREDICTION FUNCTIONS (UNCHANGED)
# ------------------------------------------------------------
def predict_profit(units, manufacturing_price, discount_rate,
                   manufacturing_efficiency, segment):

    input_df = pd.DataFrame([{
        'Units Sold': units,
        'Manufacturing Price': manufacturing_price,
        'Discount_Rate': discount_rate / 100,
        'Manufacturing_Efficiency': manufacturing_efficiency,
        'Segment': segment
    }])

    input_df = input_df[numeric_features + categorical_features]
    return reg_pipeline.predict(input_df)[0]


def predict_loss_prob(units, manufacturing_price, discount_rate,
                      manufacturing_efficiency, segment):

    input_df = pd.DataFrame([{
        'Units Sold': units,
        'Manufacturing Price': manufacturing_price,
        'Discount_Rate': discount_rate / 100,
        'Manufacturing_Efficiency': manufacturing_efficiency,
        'Segment': segment
    }])

    input_df = input_df[numeric_features + categorical_features]
    return clf_pipeline.predict_proba(input_df)[0][1]


# ------------------------------------------------------------
# 6. PRE‑COMPUTED DATAFRAMES – for fast, consistent charts (Product Analysis)
# ------------------------------------------------------------
# Profit by Product & Year (grouped bar chart)
profit_product_year = df.groupby(["Year", "Product"], as_index=False)["Profit"].sum()

# Monthly profit by Year (line/bar chart)
profit_month_year = df.groupby(["Year", "Month_Name"], as_index=False)["Profit"].sum()
month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
profit_month_year["Month_Name"] = pd.Categorical(
    profit_month_year["Month_Name"],
    categories=month_order,
    ordered=True
)
profit_month_year = profit_month_year.sort_values("Month_Name")

# Product efficiency metrics (profit per unit, gross margin, manufacturing efficiency)
product_efficiency = df.groupby("Product", as_index=False).agg(
    Avg_Profit_per_Unit=("Profit_per_Unit", "mean"),
    Avg_Gross_Margin=("Gross_Margin", "mean"),
    Avg_Manufacturing_Efficiency=("Manufacturing_Efficiency", "mean")
)

# Product contribution (pie chart)
product_contribution = df.groupby("Product", as_index=False)["Profit"].sum()

# Profit by country (choropleth + scatter)
profit_by_country = (
    df.groupby("Country", as_index=False)
    .agg(
        Total_Profit=("Profit", "sum"),
        Total_Sales=("Sales", "sum"),
        Avg_Discount=("Discount_Rate", "mean")
    )
    .assign(
        Profit_Percentage=lambda x: (x["Total_Profit"] / x["Total_Sales"] * 100).round(2)
    )
)

# Discount Rate vs Profit table (AG Grid)
discount_profit_table = df[["Year", "Country", "Product", "Discount_Rate", "Profit"]].copy()
discount_profit_table["Discount_Rate"] = (discount_profit_table["Discount_Rate"] * 100).round(2)

# K‑Means Clustering for customer/product segmentation
features_cluster = [
    "Units Sold",
    "Sales",
    "COGS",
    "Profit",
    "Discount_Rate",
    "Profit_per_Unit",
    "Manufacturing_Efficiency"
]
X_cluster = df[features_cluster].fillna(0)
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
kmeans = KMeans(n_clusters=4, random_state=42)
df_with_clusters = df.copy()
df_with_clusters["Cluster"] = kmeans.fit_predict(X_cluster_scaled)

# T‑Test: profit difference between low‑discount and high‑discount groups
median_discount = df["Discount_Rate"].median()
low_discount_profit = df[df["Discount_Rate"] <= median_discount]["Profit"]
high_discount_profit = df[df["Discount_Rate"] > median_discount]["Profit"]
t_stat, p_value = stats.ttest_ind(low_discount_profit, high_discount_profit, equal_var=False)

# ------------------------------------------------------------
# All data and functions are now ready for import into Dash pages
# ------------------------------------------------------------