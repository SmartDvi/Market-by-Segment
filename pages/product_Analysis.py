import dash
from dash import dcc, html, callback, Output, Input
import plotly.express as px
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd

dash.register_page(__name__, name='Product Analysis')

# Load the dataset
df = pd.read_csv('C:\\Users\\Moritus Peters\\Documents\\Datasets\\Sale Market Data\\Financials.csv')

# Clean the dataimport dash_bootstrap_components as dbc
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


# Drop rows with Nan values in Sales Growth Column (for the first sales data of each product)
df = df.dropna(subset=['Sales Growth'])

#Define custom color palette
color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


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
                        style={'color': 'black'},
                        className='px-2 bg-light border, mb-4'
                    ),
                    xs=12, sm=12, md=6, lg=4, xl=4
                ),
            dbc.Col(
                dcc.Checklist(
                    id='year_checklist',
                    options=[
                        {'label': str(year), 'value': year} for year in sorted(df['Year'].unique())
                    ],
                    value=sorted(df['Year'].unique()),
                    inline=True,
                    className='ml-5'
                        )
                    )
            ]
        ),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5(
                                'Product Contribution to Sales',
                                className='text-light'
                            ),
                            dcc.Graph(id='product_contribution_chart', figure={}),
                        ])
                    ),
                    xs=12, sm=12, md=6, lg=4, xl=4

                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5(
                                'Product Contribution to Sales',
                                className='text-light'),
                                    dcc.Graph(id='discount_band_impact_chart', figure={}),
                                ])
                        ),
                    xs=12, sm=12, md=6, lg=4, xl=4
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5(
                                'Profitability of Product',
                                className='text-light'),
                    dcc.Graph(id='profitability_chart', figure={}),
                        ])
                    ),
                    xs=12, sm=12, md=6, lg=4, xl=4
                )
            ]
        ),

        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5(
                                'Profit and Lost for Each product',
                                className='text-light'),
                    dcc.Graph(id='Waterfall_product_details', figure={}),
                        ])
                    ),
                     xs=4, sm=4, md=2, lg=4, xl=4,
                    className="mb-3"
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H5(
                                'Variable Coorelation on Product',
                                className='text-light'),
                    dcc.Graph(id='heatmap_correlation', figure={}),
                        ])
                    ),
                    xs=8, sm=8, md=6, lg=8, xl=8,
                    className='mb-5'

                )
            ]
        ),
    ]
)

# Callback to update product contribution chart
@callback(
    Output('product_contribution_chart', 'figure'),
    [Input('Country_Dropdown', 'value'),
     Input('year_checklist', 'value')
     ]
)
def update_product_contribution_chart(selected_countries, selected_years):
    if not selected_countries:
        return {}

    filtered_df = df[df['Country'].isin(selected_countries) & df['Year'].isin(selected_years)]
    product_sales = filtered_df.groupby('Product')['Sales'].sum().reset_index()

    fig = px.bar(product_sales, x='Product', y='Sales', color='Product',
                 title='Product Contribution to Sales',
                 labels={'Sales': 'Total Sales', 'Product': 'Product'},
                 color_discrete_sequence=color_palette)
    fig.update_layout(xaxis_tickangle=-45, )

    return fig

# Callback to update discount band impact chart
@callback(
    Output('discount_band_impact_chart', 'figure'),
    [Input('Country_Dropdown', 'value'),
     Input('year_checklist', 'value')]
)
def update_discount_band_impact_chart(selected_countries, selected_years):
    if not selected_countries:
        return {}

    filtered_df = df[df['Country'].isin(selected_countries) & df['Year'].isin(selected_years)]



    # Grouping the product and calculate average profit margin and discount efficiency
    Product_Metrics = filtered_df.groupby('Product').agg({'Profit Margin': 'mean', 'Discount Efficiency': 'mean' }).reset_index()

    # plotly the pie chat
    fig = px.pie(Product_Metrics, names='Product', values='Discount Efficiency',
                 title='Discount Efficiency by Product',
                 color_discrete_sequence=color_palette)
                 #labels={'Sales': 'Total Sales', 'Discount Band': 'Discount Band'})

   # fig.update_layout(height=500)

    return fig

# Callback to update profitability chart
@callback(
    Output('profitability_chart', 'figure'),
    [Input('Country_Dropdown', 'value'),
     Input('year_checklist', 'value')]
)
def update_profitability_chart(selected_countries, selected_years):
    if not selected_countries:
        return {}

    filtered_df = df[df['Country'].isin(selected_countries) & df['Year'].isin(selected_years)]



    # Group by product and calculate average sales growth
    avg_sales_growth = filtered_df.groupby('Product')['Sales Growth'].mean().reset_index()

    # Sort products by average sales growth
    avg_sales_growth = avg_sales_growth.sort_values(by='Sales Growth', ascending=False)

    #ploting the sales growth for each product
    fig = px.funnel(avg_sales_growth,
                    x='Product',
                    y= 'Sales Growth',
                    orientation= 'v',
                    title= ' Sales Growth Funnel by Product',
                    color_discrete_sequence=color_palette

                    )
    #fig.update_layout(height=500)
    return fig


@callback(
    Output('Waterfall_product_details', 'figure'),
    [Input('Country_Dropdown', 'value'),
     Input('year_checklist',  'value')]
)
def update_Waterfall_product_details(selected_countries, selected_years):
    if not selected_countries:
        return {}


    # filter the DataFrame base on selected coluntriess
    filtered_df = df[df['Country'].isin(selected_countries) & df['Year'].isin(selected_years)]

    # calculate the lost Aamount
    filtered_df['Lost Amount'] = filtered_df['Discounts'] - (filtered_df['Units Sold'] * (filtered_df['Sale Price'] - filtered_df['Manufacturing Price']))

    #Group by product and calculate total profit and lost
    profit_lost_df = filtered_df.groupby('Product').agg({'Profit': 'sum', 'Lost Amount': 'sum'}).reset_index()

    # Sort by profit in descending order
    profit_lost_df =profit_lost_df.sort_values(by='Profit', ascending=False)

    # creating the water fall chart
    fig = go.Figure(go.Waterfall(
        name='Profit and Lost',
        orientation='v',
        measure=['relative', 'relative'] * (len(profit_lost_df) // 2),
        x=profit_lost_df['Product'],
        y=profit_lost_df['Profit'],
        connector={'line': {'color': 'rgb(63, 63, 63)'}},
        decreasing={'marker': {'color': "#FF4136"}},
        increasing={'marker': {'color': "#2ECC40"}},
        totals={'marker': {'color': "#FF851B"}},

    )
                    )
    # updating layout

    fig.update_layout(
        title='Profit and lost waterfall chart by Product',
        showlegend=True,

    )

    return fig

# Callback to update heatmap correlation
@callback(
    Output('heatmap_correlation', 'figure'),
    [Input('Country_Dropdown', 'value'),
     Input('year_checklist', 'value')]
)
def update_heatmap_correlation(selected_countries, selected_years):
    if not selected_countries:
        return {}

    # Filter the DataFrame based on selected countries
    filtered_df = df[df['Country'].isin(selected_countries) & df['Year'].isin(selected_years)]

    # Select relevant columns
    columns_of_interest = ['Product', 'Segment', 'Discount Band', 'Discount Efficiency', 'Profit Margin', 'Sales Growth', 'Sale Price', 'Discounts', 'COGS', 'Profit', 'Sales', 'Units Sold', ' Manufacturing Price']

    # Remove leading and trailing spaces from column names
    columns_of_interest = [col.strip() for col in columns_of_interest]

    filtered_df = filtered_df[columns_of_interest]

    # pivot the DataFrame so that each row represents a product and each column represents a diferent variables
    filtered_df = filtered_df.select_dtypes(include='number')

    # Compute correlation matrix
    correlation_matrix = filtered_df.corr()

    # Transpose correlation matrix so that each product becomes a column
    #correlation_matrix = correlation_matrix.T

    # Plot heatmap
    fig = px.imshow(correlation_matrix,
                    labels=dict(color="Correlation"),
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    title='Correlation Heatmap',
                    color_continuous_scale='RdBu',
                    color_continuous_midpoint=0,


                    )



    fig.update_layout(
        title='Heatmap Correlation between Product and Important Variables',
        xaxis=dict(title='Variables'),
        yaxis=dict(title='Product',

        )

    )

    return fig