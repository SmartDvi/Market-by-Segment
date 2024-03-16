import dash
from dash import dcc, html, callback, Output, Input
import plotly.express as px
import dash_bootstrap_components
import plotly.graph_objects as go
import pandas as pd

dash.register_page(__name__, name='Sales Analysis')

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
df_cleaned = clean_and_convert_to_numeric(df, columns_to_convert)

df['Day of the week'] = pd.to_datetime(df['Date']).dt.day_name()

# Define layout
layout = html.Div(
    [
        dash_bootstrap_components.Row(
            [
                dash_bootstrap_components.Col(
                    [
                        dcc.Dropdown(
                            id='Country_Dropdown',
                            multi=True,
                            options=[{'label': x, 'value': x} for x in sorted(df['Country'].unique())],
                            style={'color': 'black'},
                            className='px-2 bg-light border, mb-4'
                        )
                    ], xs=4, sm=4, md=4, lg=4, xl=3, xxl=3

                ),
            ]
        ),
        dash_bootstrap_components.Row([
                dash_bootstrap_components.Col(
                    dash_bootstrap_components.Card(
                        dash_bootstrap_components.CardBody([
                            html.H5(
                                'Understanding the impact of different products on sales within specific countries.',
                                className='text-primary'),
                            dcc.Graph(id='scatterchart_Impact_of_sales', figure={})
                        ])
                    ),
                    xs=4, sm=4, md=4, lg=4, xl=4, xxl=4
                ),

                dash_bootstrap_components.Col(
                    dash_bootstrap_components.Card(
                        dash_bootstrap_components.CardBody([
                            html.H5('Analyzing yearly sales trends',
                                    className='text-primary'),
                            dcc.Graph(id='Linechart_sale_trend_by_year', figure={})
                        ])
                    ),

                ),

                dash_bootstrap_components.Col(
                    dash_bootstrap_components.Card(
                        dash_bootstrap_components.CardBody([
                            html.H5('Analyzing monthly sales trends',
                                    className='text-primary'),
                            dcc.Graph(id='Linechart_sale_trend_by_month', figure={})
                        ])
                    ),
                    xs=4, sm=4, md=4, lg=4, xl=4, xxl=4
                ),

                dash_bootstrap_components.Col(
                    dash_bootstrap_components.Card(
                        dash_bootstrap_components.CardBody([
                            html.H5('Analyzing day of the week sales trends',
                                    className='text-primary'),
                            dcc.Graph(id='Barchart_sales_by_day', figure={})
                        ])
                    ),
                    #xs=12, sm=12, md=4, lg=4, xl=4, xxl=4
                ),
                dash_bootstrap_components.Col(
                    dash_bootstrap_components.Card(
                        dash_bootstrap_components.CardBody([
                            html.H5(
                                'coorelation of sale.',
                                className='text-primary'),
                            dcc.Graph(id='correlation_sales', figure={})
                        ])
                    ),
                    xs=12, sm=12, md=4, lg=4, xl=4, xxl=4
                ),
        ]    )

    ]
)


# Callbacks
@callback(
    Output('Linechart_sale_trend_by_year', 'figure'),
    [Input('Country_Dropdown', 'value')]
)
def update_yearly_sales_chart(selected_countries):
    if not selected_countries:
        return {}

    filtered_df = df[df['Country'].isin(selected_countries)]
    yearly_sales = filtered_df.groupby('Year')['Sales'].sum().reset_index()
    fig = px.pie(yearly_sales, names='Year', values='Sales', title='Yearly Sales Trends')
    return fig


@callback(
    Output('Linechart_sale_trend_by_month', 'figure'),
    [Input('Country_Dropdown', 'value')]
)
def update_monthly_sales_chart(selected_countries):
    if not selected_countries:
        return {}

    filtered_df = df[df['Country'].isin(selected_countries)]
    monthly_sales = filtered_df.groupby('Month Number')['Sales'].sum().reset_index()
    fig = px.line(monthly_sales,
                 x='Month Number',
                 y='Sales',
                markers= True,

                 title='Monthly Sales Trends'
                 )
    return fig


@callback(
    Output('Barchart_sales_by_day', 'figure'),
    [Input('Country_Dropdown', 'value')]
)
def update_daily_sales_chart(selected_countries):
    if not selected_countries:
        return {}

    filtered_df = df[df['Country'].isin(selected_countries)]
    daily_sales = filtered_df.groupby('Day of the week')['Sales'].sum().reset_index()
    fig = px.bar(daily_sales, x='Day of the week', y='Sales', title='Day of the Week Sales Trends')
    return fig
@callback(
    Output('correlation_sales', 'figure'),
    [Input('Country_Dropdown', 'value')]
)
def update_heatmap(selected_countries):
    if not selected_countries:
        return {}

    filtered_df = df[df['Country'].isin(selected_countries)]
    correlation_matrix = filtered_df.corr()

    # Select columns for correlation calculation (e.g., 'Sales', 'Units Sold', 'Manufacturing Price', etc.)
    columns_for_correlation = ['Sales', 'Units Sold', 'Manufacturing Price', 'Gross Sales', 'COGS', 'Profit']

    correlation_matrix = correlation_matrix[columns_for_correlation]
    print(correlation_matrix)

    # Create heatmap
    fig = go.Heatmap(
        z=correlation_matrix.values.tolist(),
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='Viridis'
    )

    # Define layout
    fig.update_layout(
        title='Correlation Matrix',
        xaxis=dict(title='Features'),
        yaxis=dict(title='Features'))
    return fig



print(df.info())
