## Market by Segment – Business Analytics Dashboard
A comprehensive, end‑to‑end business intelligence solution extracting key insights from the sales dataset tp optimisse profitabiity and business growth. The dashboard transforms raw transactional data into actionable insights, predictive forecasts, and strategic recommendations. Attached to the repo is also a detailed analysis and explaininer (analysis interpretation) developed on Jupyternote aid the understanding of both technical and non-technical stakeholders. It combines professional data engineering, robust machine learning, and an elegant Dash interface to empower executives, analysts, and operations teams with evidence‑based decision making.

### Project previews
<img width="1856" height="966" alt="Screenshot from 2026-05-04 19-56-32" src="https://github.com/user-attachments/assets/17db305e-8d04-4cd2-9ab7-ee32163f9070" />

<img width="1856" height="966" alt="Screenshot from 2026-05-04 19-55-14" src="https://github.com/user-attachments/assets/c2bb1e7b-f422-4d37-92dd-26fe4b3ff9be" />

<img width="1856" height="966" alt="Screenshot from 2026-05-04 19-56-56" src="https://github.com/user-attachments/assets/0065b575-f581-4083-8631-d39658a30274" />

<img width="1856" height="966" alt="Screenshot from 2026-05-04 19-57-14" src="https://github.com/user-attachments/assets/ca07dc95-d78d-4cbb-9ea7-20481dd0296c" />



### 📑 Table of Contents
- Project Overview

- Dataset Description

- Key Features & Methodology

- Installation & Setup

- Running the Dashboard

- Project Structure

- Results & Business Insights

    - Year‑over‑Year Profit Growth

    - Seasonality & Demand Timing

    - Discount Impact on Profitability

    - roduct Efficiency & Contribution

    - Geographic Profit Distribution

- Model Performance

- Strategic Recommendations

- Future Enhancements

- License

### Project Overview
#### Objective:
Build a fully transparent, data‑driven analytics platform that:

- Cleans and enriches raw sales data with business‑relevant metrics.

- Provides interactive visualisations to explore product, regional, and temporal performance.

- Predicts future profit and loss‑risk using state‑of‑the‑art XGBoost models.

- Quantifies the statistical impact of pricing decisions (discounts).

- Segments customers/products via K‑Means clustering.

- Delivers concise, executive‑ready strategic insights – all computed from your data, no hardcoded numbers.


### Target Audience:
Executives, Product Managers, Finance Teams, Operations Managers, and Data Analysts.

### Technologies Used:
`Python`, `pandas`, `scikit-learn`, `XGBoost`, `Dash`, `Plotly`, `Dash Bootstrap Components`, `scipy`, `statsmodels`.

#### Dataset Description
The analysis is based on the file `Sample data.xlsx`, which contains 700 rows of historical sales transactions. Each row represents a sale and includes the following original columns:

Column	                    Description
Segment	                    Customer segment (e.g. Government, Midmarket)
Country	                    Country of sale
Product	                    Product name (Paseo, VTT, Amarilla, Carretera, Montana, Velo)
Discount Band	            Categorical discount level (e.g. Low, Medium, High)
Units Sold	                Number of units sold
Manufacturing Price	        Cost to manufacture one unit
Sale Price	                List price before discount
Gross Sales	                Sale Price × Units Sold
Discounts	                Dollar amount discounted
Sales	                    Net sales after discount
COGS	                    Cost of Goods Sold (Manufacturing Price × Units Sold)
Profit	                    Sales – COGS
Date	                    Transaction date


From these, we engineered additional business metrics:

- Revenue per Unit, Cost per Unit, Profit per Unit

- Gross Margin = Profit / Sales

- Discount Rate = Discounts / Gross Sales

- Manufacturing Efficiency = Sale Price / Manufacturing Price

- Year, Month, Quarter, Month Name

- Is_Loss (binary: profit < 0)

- High_Discount (binary: discount rate > median)

All derived fields are fully traceable and computed inside the code – no external data is ever introduced.

### Key Features & Methodology
1. Automated Data Preparation
    - Zero values in critical numeric columns are replaced with NaN to avoid division errors.

    - Unit economics and margin ratios are calculated directly from the source columns.

    - Date features are extracted for time‑series analysis.

2. KPI Dashboard (Pre‑computed Metrics)
    - The function get_kpi_metrics() returns high‑level numbers used across pages:

    - Total Sales, Total Profit, Total Units Sold

    - Average Profit Margin, Average Discount Rate

    - Number of unique products, profitable vs. loss‑making products

3. Strategic Insights Generator
    - get_strategic_insights() dynamically computes:

    - Top performing Segment, Country, Product, and Quarter by total profit.

    - Best Discount Band (by average profit).

    - Worst‑performing product for immediate review.
    - All text is generated from the data – no manual updates needed.

4. Machine Learning Pipelines
    - Two XGBoost models are trained using the same preprocessor:

    - Regressor: Predicts Profit (continuous).
        Optimised hyperparameters: `n_estimators=300`, `max_depth=4`, `learning_rate=0.05`, 
        `subsample=0.8`, `colsample_bytree=0.8`.

    - Classifier: Predicts Is_Loss (binary).
        Handles class imbalance with `scale_pos_weight` calculated from training set.
        Evaluation includes cross‑validation for Accuracy, F1, Precision, Recall, and ROC AUC.

Both models share a `ColumnTransformer` that scales numeric features and one‑hot encodes Segment.

5. Clustering (K‑Means)
    Unsupervised learning to segment products/transactions based on `Units Sold`, `Sales`, `COGS`, `Profit`, `Discount_Rate`, `Profit_per_Unit`, and `Manufacturing_Efficiency`.
    Four clusters are generated and visualised in the Predictive Analytics page.

6. Statistical Testing
A Welch’s t‑test compares profit between low‑discount and high‑discount groups (split at median discount rate).
Result: t = 14.2, p < 0.001 – highly significant difference.

7. Interactive Dash Dashboard
Four fully responsive pages built with Dash and Dash Bootstrap Components:

Page	               Path	               Description
Introduction	        `/`  	           Project overview, key metrics, raw data preview (AG Grid).
Product Analysis	  `/product_analysis`  Deep dive into product profitability, trends, geography, and discount impact.
Business Insights	    /Insighs	        Executive summary: heatmaps, quarterly trends, regional bars, efficiency scatter.
Predictive Analytics	/prediction	        Profit simulator, feature importance, confusion matrix, residual plot, clusters, t‑test result.

Every chart is interactive and can be filtered by country, product, segment, year, date range, and profit range.

### Installation & Setup
#### Prerequisites
- Python 3.9 or higher

- pip (Python package installer)

##### Step 1: Clone the Repository

`git clone https://github.com/your-org/market-by-segment.git
cd market-by-segment`

##### Step 2: Install Dependencies
All required libraries are listed in requirements.txt. Install them with:


`pip install -r requirements.txt`
Note: The environment includes `dash`, `plotly`, `scikit-learn`, `xgboost`, and `others`.

#### Step 3: Place the Data File
Ensure that Sample data.xlsx is located in the project root folder.
The script loads it via a relative path – no additional configuration needed.

Running the Dashboard
Launch the application with:

`python app.py`
After a few seconds, you will see:

``` Dash is running on http://127.0.0.1:7080/ ```

Open this URL in any modern browser.

    The dashboard is fully self‑contained. All pre‑computations (KPIs, models, clusters) run once at startup – pages load instantly.

### Project Structure
```
market-by-segment/
│
├── app.py                 # Main Dash app, layout, and page registry
├── utils.py               # Data loading, feature engineering, models, pre‑computed DataFrames
├── pages/                 # Individual Dash pages
│   ├── __init__.py
│   ├── home.py            # Introduction page (path '/')
│   ├── product_analysis.py
│   ├── insights.py        # Business Insights page (path '/Insighs')
│   └── prediction.py      # Predictive Analytics page
│
├── Sample data.xlsx       # Your original dataset (must be present)
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── assets/               # (Optional) custom CSS, favicon, etc.
```

### Results & Business Insights
All insights below are dynamically generated from the provided dataset and are visible in the dashboard. They represent a small sample of the total intelligence available.

### Year‑over‑Year Profit Growth
- Every product increased profit from 2013 to 2014 – a 100% success rate, indicating systematic business expansion, not random variation.

- Paseo grew from ~$1.10M to ~$3.70M (+236%), contributing the largest absolute profit.

- Amarilla and VTT also showed strong growth (~160% and ~145% respectively).

- Carretera achieved the highest relative growth (from $0.04M to $1.79M), suggesting a successful operational turnaround.

What it means: Profit is concentrated in a few high‑performers. Resources should be allocated accordingly.

### Seasonality & Demand Timing
- Profit is highly seasonal. In 2014, December reached ~$2.03M, October ~$1.78M, and June ~$1.47M.

- November 2014 was the lowest month at ~$0.60M – less than one‑third of the peak month.

- The pattern repeats across years, making it predictable and actionable.

What it means: Align inventory, staffing, and marketing with these statistically proven peaks to maximise revenue without increasing fixed costs.

### Discount Impact on Profitability
- t‑test: `t = 14.2`, `p < 0.0011 – the difference in profit between low‑discount and high‑discount groups is highly significant.

- Products with discount rates <5% maintain high margins (e.g., Paseo at 4% discount → $262k profit).

- Discount rates >10–15% frequently lead to negative profit (e.g., VTT, Carretera, Amarilla show losses up to -$40k).

- OLS trendline confirms: each 1% increase in discount rate is associated with a measurable decrease in profit.

What it means: Aggressive discounting destroys value. Shift to targeted, low‑discount strategies for high‑margin products.

### Product Efficiency & Contribution
- VTT and Amarilla generate the highest profit per unit (~$19.15 and $18.67).
Velo lags at $13.56 per unit.

- Carretera boasts the highest gross margin (30%) and exceptional manufacturing efficiency (37.26), likely due to automation.

- Profit concentration:

    - Paseo → 28.4% of total profit

    - VTT → 18.0%

    - Amarilla → 16.7%

    - Remaining three products share the rest.

What it means: Focus investment on Paseo, VTT, and Amarilla. Investigate cost structure of Velo. Replicate Carretera’s operational model.

### Geographic Profit Distribution
- Top profit countries: France ($3.78M), Germany ($3.68M), Canada ($3.53M).

- USA has the highest sales but lower profitability (margin 11.97% vs. France 15.53%).

- Discount rates are similar across all countries (7–8%), so profit differences stem from efficiency, product mix, and cost control.

What it means: Prioritise capacity and marketing in high‑profit regions. Improve operational efficiency in the USA and Mexico.

### Model Performance
#### Profit Regression (XGBoost)
- R² (test set): 0.932

- Mean Absolute Error: ~$23,400

- Top 3 features by importance:

    1. Manufacturing_Efficiency
    2. Units Sold
    3. Discount_Rate
    
### Loss Classification (XGBoost)
- Accuracy: 94.7%

- F1 Score: 0.89

- Precision: 0.91

- Recall: 0.87

- ROC AUC: 0.976

Confusion matrix and feature importance are displayed live on the Predictive Analytics page.

These metrics confirm that the models are highly reliable and suitable for operational deployment.

### Strategic Recommendations
Based on the statistical evidence and machine learning outputs, we recommend:

1. Scale high‑impact products – Increase inventory, marketing, and production capacity for Paseo, VTT, and Amarilla.

2. Apply surgical discounting – Cap discounts at ≤5% for proven profitable products. Use deeper discounts only for inventory clearance, and monitor profit impact in real time.

3. Prepare for seasonal peaks – Ramp up operations in October, December, and June. Reduce non‑essential spending in low‑profit months.

4. Optimise manufacturing efficiency – Investigate why Amarilla and VTT have low efficiency scores (<0.6) and consider automation or process improvements.

5. Regional resource alignment – Prioritise France, Germany, and Canada for expansion. In the USA and Mexico, focus on margin improvement rather than volume growth.

6. Review underperformers – Conduct a deep dive on Velo (lowest profit/unit) and products that frequently appear in the high‑discount / loss quadrant.

7. Operationalise the models – Integrate the profit simulator into sales planning; use the classifier as a real‑time loss‑risk alert during discount approval workflows.

#### Future Enhancements
- SHAP explainability: The library is already imported – a natural extension to visualise individual prediction explanations.

- Automated report generation: Export key charts and insights as PDF/PPTX.

- What‑if scenario analysis: Simulate changes in manufacturing price or discount bands across multiple scenarios.

- Database connectivity: Replace Excel file with live database connection (SQL, cloud storage).

- User authentication: Role‑based access for different departments.




