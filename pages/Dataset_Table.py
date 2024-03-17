import dash
from dash import dcc, html, callback, Output, Input, dash_table
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd

dash.register_page(__name__, name="Dataset Table📋")

# Load the dataset
df = pd.read_csv('C:\\Users\\Moritus Peters\\Documents\\Datasets\\Sale Market Data\\Financials.csv')

# Clean the data
df.columns = df.columns.str.strip()

# Define data cleaning function
def clean_and_convert_to_numeric(df, columns):
    for column in columns:
        df[column] = pd.to_numeric(df[column].str.replace(r'[^\d.]', '', regex=True), errors='coerce')
    return df

# Clean and convert columns to numeric
columns_to_convert = ['Units Sold', 'Manufacturing Price', 'Sale Price', 'Gross Sales', 'Discounts', 'Sales', 'COGS', 'Profit']

df = clean_and_convert_to_numeric(df, columns_to_convert)

# calculating the profit margin
df['Profit Margin'] = (df['Profit'] / df['Sales']) * 100

# Calculating the discount efficiency for the products
df['Discount Efficiency'] = (df['Discounts'] / df['Gross Sales']) * 100

# Drop rows with NaN values in the calculated columns
df.dropna(subset=['Profit Margin', 'Discount Efficiency'], inplace=True)

# Extract day of the week from the date
df['Day of the week'] = pd.to_datetime(df['Date']).dt.day_name()

# calculating the sales Growth for each product
df['Previous Sales'] = df.groupby('Product')['Sales'].shift(1)
df['Sales Growth'] = ((df['Sales'] - df['Previous Sales']) / df['Previous Sales'] * 100)

layout = html.Div(children=[
    html.Br(),
    dash_table.DataTable(data=df.to_dict('records'),
                         page_size=20,
                         style_cell={"background-color": "lightgrey", "border": "solid 1px white", "color": "black", "font-size": "11px", "text-align": "left"},
                         style_header={"background-color": "dodgerblue", "font-weight": "bold", "color": "white", "padding": "10px", "font-size": "18px"},
                        ),
])