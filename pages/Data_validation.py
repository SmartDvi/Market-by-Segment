
import dash
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from dash import dcc, html, callback, Output, Input
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
import dash_bootstrap_components as dbc
from statsmodels.formula.api import ols
from scipy.stats import pearsonr
import plotly.express as px
import dash_bootstrap_components
import dash_daq as daq
import plotly.graph_objects as go


dash.register_page(__name__, name='Data Validation')

# Load data
df = pd.read_csv('C:\\Users\\Moritus Peters\\Documents\\Datasets\\Sale Market Data\\Financials.csv')

df.columns = df.columns.str.strip()

# Data cleaning function
def clean_and_convert_to_numeric(df, columns):
    for column in columns:
        df[column] = df[column].astype(str)
        df[column] = pd.to_numeric(df[column].str.replace(r'[^\d.]', '', regex=True), errors='coerce')
    return df


columns_to_convert = ['Units Sold', 'Manufacturing Price', 'Sale Price', 'Gross Sales', 'Discounts', 'Sales', 'COGS',
                      'Profit']


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
                            id='year_checklist',
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
                            dcc.Graph(id='Table_Pvale_and_corr', figure={}),
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
