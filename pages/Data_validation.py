
import dash
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from dash import dcc, html, callback, Output, Input
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import dash_bootstrap_components as dbc
from statsmodels.formula.api import ols
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr
import plotly.express as px
from pandas.api.types import CategoricalDtype
import dash_bootstrap_components
import dash_daq as daq
import plotly.graph_objects as go


dash.register_page(__name__, name='Data Validation')

# Load data
df = pd.read_csv('C:\\Users\\Moritus Peters\\Documents\\Datasets\\Sale Market Data\\Financials.csv')

df.columns = df.columns.str.strip()

import pandas as pd

def fill_nan_with_zero(df):
    return df.fillna(0)

# Assuming df is your DataFrame
filled_df = fill_nan_with_zero(df)


# Data cleaning function
def clean_and_convert_to_numeric(df, columns):
    for column in columns:
        df[column] = df[column].astype(str)
        df[column] = pd.to_numeric(df[column].str.replace(r'[^\d.]', '', regex=True), errors='coerce')
    return df

columns_to_convert = ['Units Sold', 'Manufacturing Price', 'Sale Price', 'Gross Sales', 'Discounts', 'Sales', 'COGS',
                      'Profit']
df = clean_and_convert_to_numeric(df, columns_to_convert)

import pandas as pd

def rename_columns_with_keywords(df, keywords, replace_char='_'):
    renamed_columns = {}
    for col in df.columns:
        for keyword in keywords:
            if keyword in col:
                new_col = col.replace(' ', replace_char)
                renamed_columns[col] = new_col
                break  # Move to the next column after renaming the current one
    return df.rename(columns=renamed_columns)

# Define keywords
keywords = ['Discount Band', 'Units Sold', 'Manufacturing Price', 'Sale Price',
            'Gross Sales', 'Discounts', 'Sales', 'COGS', 'Profit', 'Date',
            'Month_Number', 'Month Name', 'Year']

# Renaming columns with keywords and replacing spaces with underscores
df = rename_columns_with_keywords(df, keywords)





layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id='Country_Dropdown',
                        multi=True,
                        options=[{'label': x, 'value': x} for x in sorted(df['Country'].unique())],
                        style={'color': 'black'},
                        className='px-2 bg-light border, mb-4'
                    ),
                    xs=12, sm=12, md=6, lg=4, xl=4
                ),
                dash_bootstrap_components.Col([
                    html.Div([
                        html.H5('Product Checklist'),
                        dcc.Checklist(
                            id='product_checklist',
                            options=[
                                {'label': str(Product), 'value': Product} for Product in sorted(df['Product'].unique())
                            ],
                            value=sorted(df['Product'].unique()),
                            inline=True,
                            className='ml-5 ms-3'

                        )
                    ])
                ],

                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5(
                                'Calculated correlation coefficents and p-values Table ',
                                className='text-light'
                            ),
                            dcc.Graph(id='Table_Pvalues_and_corr', figure={}),
                        ])
                    ),
                    xs=6, sm=6, md=6, lg=6, xl=6

                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5(
                                'Plotting the number of predictors for each threshold',
                                className='text-light'),
                            dcc.Graph(id='Bar_Predictor_Threshold', figure={}),
                        ])
                    ),
                    xs=6, sm=6, md=6, lg=6, xl=6),
            ]
        ),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5(
                                'Assess the accuracy of the models',
                                className='text-light'),
                            dcc.Graph(id='scatterplot_accuracy_model', figure={}),
                        ])
                    ),
                    xs=12, sm=12, md=12, lg=12, xl=12, )

            ]
        ),

    ]
)



@callback(Output('Table_Pvalues_and_corr', 'figure'),
          [Input('Country_Dropdown', 'value'),
           Input('product_checklist', 'value')],
          prevent_initial_call=True)
def predictors_threshold(selected_countries, selected_products):
    if not selected_countries or not selected_products:
        return {}

    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries) & filtered_df['Product'].isin(selected_products)]

    # Define the response variable
    y_name = 'Profit'

    # Convert Date columns into dummy variables
    df_dummies = pd.get_dummies(filtered_df, columns=[col for col in filtered_df.columns if 'Date_' in col])

    # Select only numeric columns for correlation calculation
    numeric_cols = df_dummies.select_dtypes(include=np.number).columns

    # Calculate correlations between predictor variables and the response variable
    corrs = df_dummies[numeric_cols].corr()[y_name].sort_values(ascending=False)

    # Build a dictionary of correlation coefficients and p-values
    dict_cp = {}

    for col in corrs.index:
        if col != y_name:
            # Handle infinite and NaN values
            valid_indices = np.isfinite(df_dummies[col]) & np.isfinite(df_dummies[y_name])
            x_values = df_dummies[col][valid_indices]
            y_values = df_dummies[y_name][valid_indices]

            # Calculate correlation coefficient and p-value
            corr_coef, p_val = pearsonr(x_values, y_values)
            dict_cp[col] = {'Variable': col, 'Correlation_Coefficient': corr_coef, 'P_Value': p_val}

    # Convert the dictionary into a DataFrame
    df_cp = pd.DataFrame(dict_cp).T

    # Sort DataFrame by p-value
    df_cp_sorted = df_cp.sort_values('P_Value')

    # Filter out rows with p-value less than 0.1
    hu = df_cp_sorted[df_cp_sorted['P_Value'] < 0.1]

    # Create a table using Plotly Express
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df_cp_sorted.columns),
                    fill_color="dodgerblue",
                    align='left'),
        cells=dict(values=[hu[col] for col in df_cp_sorted.columns],
                   fill_color="lightgrey",
                   align='left'))
    ])

    # Set table layout
    fig.update_layout(width=800, height=400)

    # Show the table
    return fig


@callback(Output('Bar_Predictor_Threshold', 'figure'),
              [Input('Country_Dropdown', 'value'),
               Input('product_checklist', 'value')],
              prevent_initial_call=True)
def update_bar_predictor_threshold(selected_countries, selected_products):
    filtered_df = df.copy()

    # Filter by selected countries
    if selected_countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]

    # Filter by selected products
    if selected_products:
        filtered_df = filtered_df[filtered_df['Product'].isin(selected_products)]

    # Convert non-numeric data to NaN
    filtered_df = filtered_df.apply(pd.to_numeric, errors='coerce')

    # Drop columns with NaN values
    filtered_df = filtered_df.dropna(axis=1)

    # Normalize data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(filtered_df)
    X_normalize = pd.DataFrame(X_scaled, columns=filtered_df.columns)

    # Create VarianceThreshold objects for different thresholds
    thresholds = [0.03, 0.05, 0.1, 0.15]
    predictors_by_threshold = []

    for threshold in thresholds:
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X_normalize)
        predictors = X_normalize.columns[selector.get_support(indices=True)]
        predictors_by_threshold.append(len(predictors))

    # Create bar plot
    fig = go.Figure(data=[go.Bar(x=[str(threshold) for threshold in thresholds], y=predictors_by_threshold)],)
    fig.update_layout(title="Number of Predictors vs Thresholds",
                      xaxis_title="Threshold",
                      yaxis_title="Number of Predictors")

    return fig

@callback(Output('scatterplot_accuracy_model', 'figure'),
          [Input('Country_Dropdown', 'value'),
           Input('product_checklist', 'value')],
          prevent_initial_call=True)
def update_scatterplot_accuracy(selected_countries, selected_products):
    filtered_df = df.copy()

    # Filter by selected countries
    if selected_countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]

    # Filter by selected products
    if selected_products:
        filtered_df = filtered_df[filtered_df['Product'].isin(selected_products)]

    # Drop rows with missing target values
    filtered_df.dropna(subset=['Profit'], inplace=True)

    # Define dependent variable
    y_name = 'Profit'

    # Create dummy variables for categorical columns
    df_dummies = pd.get_dummies(filtered_df, columns=['Country', 'Product'], drop_first=True)

    # Independent variables
    X_names = [col for col in df_dummies.columns if col != y_name]

    # Split data into training and testing sets
    X_data = df_dummies[X_names]
    y_data = df_dummies[y_name]
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.20, shuffle=False)

    # One-hot encode categorical variables
    encoder = OneHotEncoder()
    encoder.fit(X_train[['Segment', 'Discount_Band']])
    X_train_encoded = encoder.transform(X_train[['Segment', 'Discount_Band']])
    X_test_encoded = encoder.transform(X_test[['Segment', 'Discount_Band']])

    # Instantiate the model
    lm = LinearRegression()

    # fit the model
    lm.fit(X_train_encoded, y_train)

    # Predictions
    train_pred = lm.predict(X_train_encoded)
    test_pred = lm.predict(X_test_encoded)

    # Evaluation metrics
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)

    # Creating scatter plot
    fig = go.Figure(go.Scatter(x=y_train,
                               y=train_pred,
                               mode='markers',
                               name='Training  Data'))
    # plot testing data
    fig.add_trace(go.Scatter(x=y_test,
                             y=test_pred,
                             mode='markers',
                             name='Testing Data'))

    # updating Layout
    fig.update_layout(title='Assessing Model Accuracy',
                      xaxis_title='Actual Profit',
                      yaxis_title='Predicted Profit',
                      showlegend=True)

    # print the evaluation metrics
    print(f"Training MSE: {train_mse}")
    print(f"Testing MSE: {test_mse}")
    print(f"Training R-squared : {train_r2}")
    print(f"Testing R-squared : {test_r2}")
    print(f"Training MAE : {train_mae}")
    print(f"Testing MAE: {test_mae}")

    return fig