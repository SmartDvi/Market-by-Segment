import dash
from dash import dcc, html, callback, Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd

dash.register_page(__name__, name='Product Analysis')

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

# Extract day of the week from the date
df['Day of the week'] = pd.to_datetime(df['Date']).dt.day_name()

# Define the Dash app


# Define layout
layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id='Country_Dropdown',
                        multi=True,
                        options=[{'label': x, 'value': x} for x in sorted(df['Country'].unique())],
                        style={'color': 'black'}
                    ),
                    xs=12, sm=12, md=6, lg=4, xl=4
                )
            ]
        ),

        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id='product_contribution_chart', figure={}),
                    xs=12, sm=12, md=6, lg=4, xl=4
                ),
                dbc.Col(
                    dcc.Graph(id='discount_band_impact_chart', figure={}),
                    xs=12, sm=12, md=6, lg=4, xl=4
                ),
                dbc.Col(
                    dcc.Graph(id='profitability_chart', figure={}),
                    xs=12, sm=12, md=6, lg=4, xl=4
                )
            ]
        )
    ]
)

# Callback to update product contribution chart
@callback(
    Output('product_contribution_chart', 'figure'),
    [Input('Country_Dropdown', 'value')]
)
def update_product_contribution_chart(selected_countries):
    if not selected_countries:
        return {}

    filtered_df = df[df['Country'].isin(selected_countries)]
    product_sales = filtered_df.groupby('Product')['Sales'].sum().reset_index()

    fig = px.bar(product_sales, x='Product', y='Sales', color='Product',
                 title='Product Contribution to Sales',
                 labels={'Sales': 'Total Sales', 'Product': 'Product'})
    fig.update_layout(xaxis_tickangle=-45)

    return fig

# Callback to update discount band impact chart
@callback(
    Output('discount_band_impact_chart', 'figure'),
    [Input('Country_Dropdown', 'value')]
)
def update_discount_band_impact_chart(selected_countries):
    if not selected_countries:
        return {}

    filtered_df = df[df['Country'].isin(selected_countries)]
    discount_sales = filtered_df.groupby('Discount Band')['Sales'].sum().reset_index()

    fig = px.pie(discount_sales, names='Discount Band', values='Sales',
                 title='Impact of Discount Bands on Sales',
                 labels={'Sales': 'Total Sales', 'Discount Band': 'Discount Band'})

    return fig

# Callback to update profitability chart
@callback(
    Output('profitability_chart', 'figure'),
    [Input('Country_Dropdown', 'value')]
)
def update_profitability_chart(selected_countries):
    if not selected_countries:
        return {}

    filtered_df = df[df['Country'].isin(selected_countries)]
    avg_profit = filtered_df.groupby('Product')[['Profit', 'Sales Growth']].mean().reset_index()
    print(avg_profit)
    fig = px.bar(avg_profit,
                 x='Product',
                 y='Profit',
                 color='Product',
                 title='Average Profitability of Sales')
    fig.add_trace(go.scatter(x=avg_profit['Product'],
                             y=avg_profit['Sales Growth'],
                             mode='lines+mackers',
                             color='red'))

    fig.update_layout(xaxis_tickangle=90)

    return fig


