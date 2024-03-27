import dash
from dash import dcc, html, callback, Output, Input
import plotly.express as px
import dash_bootstrap_components
import dash_daq as daq
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

monthly_sales_growth_rate = df.groupby(['Year', 'Month Number'])['Sales'].sum().pct_change() * 100

#Define custom color palette
color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Define layout
layout = html.Div(
    [
        dash_bootstrap_components.Row(
            [
                dash_bootstrap_components.Col(
                    [
                        html.Div([
                        html.H5('List of Counties'),
                        dcc.Dropdown(
                            id='Country_Dropdown',
                            multi=True,
                            options=[{'label': x, 'value': x} for x in sorted(df['Country'].unique())],
                            style={'color': 'black'},
                            className='px-2 bg-light border, mb-4'
                        )
                        ])
                    ], xs=4, sm=4, md=4, lg=4, xl=3, xxl=3

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
        dash_bootstrap_components.Row([
                dash_bootstrap_components.Col(
                    dash_bootstrap_components.Card(
                        dash_bootstrap_components.CardBody([
                            html.H5(
                                'KPI on Profit Marget Target',
                                className='text-light'),
                            dcc.Graph(id='Busines_gauge', figure={})
                        ])
                    ),
                    xs=4, sm=4, md=4, lg=4, xl=4, xxl=4
                ),

                dash_bootstrap_components.Col(
                    dash_bootstrap_components.Card(
                        dash_bootstrap_components.CardBody([
                            html.H5('Analyzing yearly sales trends',
                                    className='text-light'),
                            dcc.Graph(id='Linechart_sale_trend_by_year', figure={})
                        ])
                    ),  xs=4, sm=4, md=4, lg=4, xl=4, xxl=4

                ),

                dash_bootstrap_components.Col(
                    dash_bootstrap_components.Card(
                        dash_bootstrap_components.CardBody([
                            html.H5('Monthly Sales Growth Rate',
                                    className='text-light'),
                            dcc.Graph(id='Linechart_Monthly_Sales_Growth_Rate', figure={})
                        ])
                    ),
                    xs=4, sm=4, md=4, lg=4, xl=4, xxl=4
                ),

            dash_bootstrap_components.Col(
                dash_bootstrap_components.Card(
                    dash_bootstrap_components.CardBody([
                    html.H5(
                'Distribution of Conversion Rates by Customer Segment',
                className='text-light'),
                        dcc.Graph(id='scatter_sales_profit', figure={})  # Updated ID and figure
                    ])
                ),
                xs=8, sm=8, md=8, lg=8, xl=8, xxl=8
            ),
                dash_bootstrap_components.Col(
                    dash_bootstrap_components.Card(
                        dash_bootstrap_components.CardBody([
                            html.H5(
                                ' Impact of Segment on Sales ',
                                className='text-light'),
                            dcc.Graph(id='bar_Conversion_Rates', figure={})
                        ])
                    ),
                    xs=4, sm=4, md=4, lg=4, xl=4, xxl=4
                ),
        ]    )

    ]
)
@callback(Output('Busines_gauge', 'figure'),
    [Input('Country_Dropdown', 'value'),
     Input('year_checklist', 'value')]
)
def update_guage(selected_countries, selected_years):
    if not selected_countries:
        return {}
    filtered_df = df[df['Country'].isin(selected_countries) & df['Product'].isin(selected_years)]
    average_profit_margin = filtered_df['Profit'].mean()

    # the indicator figure
    fig = go.Figure(go.Indicator(
        mode="number+gauge+delta",
        value=average_profit_margin,
        domain={'x': [0, 1], 'y': [0, 1]},
        delta={'reference': 280, 'position': "top"},
       # title={'text': "<b>Target Profit</b><br><span style='color: gray; font-size:0.8em'>U.S. $</span>",
              # 'font': {"size": 14}},
        title=f"(KPI) Target profit for Business"
              f" in {selected_countries} for {selected_years}",
        gauge={
            'shape': "bullet",
            'axis': {'range': [None, 300]},
            'threshold': {
                'line': {'color': "red", 'width': 2},
                'thickness': 0.75, 'value': 270},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 150], 'color': "cyan"},
                {'range': [150, 250], 'color': "royalblue"}],
            'bar': {'color': "darkblue"}}))
    fig.update_layout(height=250)

    # Return the figure as a dictionary with 'data' and 'layout' keys
    return  fig



# Callbacks
@callback(
    Output('Linechart_sale_trend_by_year', 'figure'),
    [Input('Country_Dropdown', 'value'),
     Input('year_checklist', 'value')]
)
def update_yearly_sales_chart(selected_countries, selected_years):
    if not selected_countries:
        return {}

    filtered_df = df[df['Country'].isin(selected_countries) & df['Product'].isin(selected_years)]
    yearly_sales = filtered_df.groupby('Year')['Sales'].sum().reset_index()
    fig = px.pie(yearly_sales,
                 names='Year',
                 values='Sales',
                 title=f'Yearly Sales Trends in {", ".join(selected_countries)} for {", ".join(selected_years)}',
                 color_discrete_sequence = color_palette)
    return fig


@callback(
    Output('Linechart_Monthly_Sales_Growth_Rate', 'figure'),
    [Input('Country_Dropdown', 'value'),
     Input('year_checklist', 'value')]
)
def update_monthly_sales_chart(selected_countries, selected_years):
    if not selected_countries:
        return {}

    filtered_df = df[df['Country'].isin(selected_countries) & df['Product'].isin(selected_years)]

    # fetching the monthly sales percentage
    monthly_sales = filtered_df.groupby(['Year', 'Month Number'])['Sales'].sum().reset_index()

    # calculating sales Growth Rate
    monthly_sales['Sales Growth Rate'] = monthly_sales.groupby('Year')['Sales'].pct_change() * 100
    print(monthly_sales)

    # plotly the graph
    fig = px.line(monthly_sales,
                 x='Month Number',
                 y='Sales Growth Rate',
                markers= True,
                color='Year',

                  hover_data={'Sales Growth Rate': ':.3f'},
                color_discrete_sequence = color_palette,
                title=f'Monthly Sales Growth Rate {", ".join(selected_countries)} for {", ".join(selected_years)}'
                 )
    fig.update_layout(xaxis_title='Month in Number', yaxis_title ='Sales Growth Rate (%)')

    return fig


@callback(
    Output('scatter_sales_profit', 'figure'),
    [Input('Country_Dropdown', 'value'),
     Input('year_checklist', 'value')]
)
def update_scatterplot_sales_profit(selected_countries, selected_years):
    if not selected_countries or not selected_years:
        return {}

    filtered_df = df[df['Country'].isin(selected_countries) & df['Product'].isin(selected_years)]

    # Create scatter plot
    fig = px.scatter(filtered_df, x='Sales', y='Profit', color='Country',
                     hover_name='Country',
                     title='Sales vs Profit by Country',
                     labels={'Sales': 'Sales (USD)', 'Profit': 'Profit (USD)'},
                     )

    # Customize layout
    fig.update_layout(plot_bgcolor="white",
                      xaxis_title='Sales (USD)',
                      yaxis_title='Profit (USD)',
                      margin=dict(l=50, r=50, t=80, b=50)  # Adjust margin for better layout
                      )
    return fig


@callback(
    Output('bar_Conversion_Rates', 'figure'),
    [Input('Country_Dropdown', 'value'),
     Input('year_checklist', 'value')]
)
def update_conversion_rates(selected_countries, selected_years):
    if not selected_countries:
        return {}

    filtered_df = df[df['Country'].isin(selected_countries) & df['Year'].isin(selected_years)]
    filtered_df['Conversion Rate'] = (filtered_df['Units Sold'] / filtered_df['Sales']) * 100



    filtered_df = df[df['Country'].isin(selected_countries)]
    filtered_df['Conversion Rate'] = (filtered_df['Units Sold'] / filtered_df['Sales']) * 100

    # Create a column for colors based on the Conversion Rate
    colors = ['red' if rate < 0 else 'green' for rate in filtered_df['Conversion Rate']]

    fig = px.bar(
        filtered_df,
        x='Segment',
        y='Conversion Rate',
        color='Segment',
        hover_data=df.columns,
        color_discrete_sequence = color_palette
    )

   # fig.update_traces(marker=dict(color=filtered_df['Conversion Rate'].apply(lambda x: 'red' if x < 0 else 'green')))

    fig.update_layout(
        title=f"Conversion Rates by Customer Segment for Business",
        xaxis_title="Customer Segment",
        yaxis_title="Conversion Rate (%)",
        legend_title="Segment",
        font=dict(family="Arial", size=12, color="black"),
        plot_bgcolor="white",
        hoverlabel=dict(font=dict(family="Arial", size=12)),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    return fig