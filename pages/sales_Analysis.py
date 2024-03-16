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
                        dcc.Dropdown(
                            id='Country_Dropdown',
                            multi=True,
                            options=[{'label': x, 'value': x} for x in sorted(df['Country'].unique())],
                            style={'color': 'black'},
                            className='px-2 bg-light border, mb-4'
                        )
                    ], xs=4, sm=4, md=4, lg=4, xl=3, xxl=3

                ),
                dash_bootstrap_components.Col(
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
        dash_bootstrap_components.Row([
                dash_bootstrap_components.Col(
                    dash_bootstrap_components.Card(
                        dash_bootstrap_components.CardBody([
                            html.H5(
                                'Gauge.',
                                className='text-primary'),
                            dcc.Graph(id='Busines_gauge', figure={})
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
                            html.H5('Monthly Sales Growth Rate',
                                    className='text-primary'),
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
                            className='text-primary'
                        ),
                        dcc.Graph(id='Violin_Conversion_Rates', figure={})  # Updated ID and figure
                    ])
                ),
                #xs=12, sm=12, md=4, lg=4, xl=4, xxl=4  # Adjusted column size
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
@callback(Output('Busines_gauge', 'figure'),
    [Input('Country_Dropdown', 'value'),
     Input('year_checklist', 'value')]
)
def update_guage(selected_countries, selected_years):
    if not selected_countries:
        return {}
    filtered_df = df[df['Country'].isin(selected_countries) & df['Year'].isin(selected_years)]
    average_profit_margin = filtered_df['Profit'].mean()
    # Replace 220 with the actual profit value you want to display
    # Replace 280 with the reference profit value
    fig = go.Figure(go.Indicator(
        mode="number+gauge+delta",
        value=average_profit_margin,
        domain={'x': [0, 1], 'y': [0, 1]},
        delta={'reference': 280, 'position': "top"},
        title={'text': "<b>Profit</b><br><span style='color: gray; font-size:0.8em'>U.S. $</span>",
               'font': {"size": 14}},
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

    filtered_df = df[df['Country'].isin(selected_countries) & df['Year'].isin(selected_years)]
    yearly_sales = filtered_df.groupby('Year')['Sales'].sum().reset_index()
    fig = px.pie(yearly_sales,
                 names='Year',
                 values='Sales',
                 title='Yearly Sales Trends',
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

    filtered_df = df[df['Country'].isin(selected_countries) & df['Year'].isin(selected_years)]

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
                  color_discrete_sequence = color_palette,
                 title='Monthly Sales Growth Rate Trends by Year'
                 )
    fig.update_layout(xaxis_title='Month in Number', yaxis_title ='Sales Growth Rate (%)')

    return fig


@callback(
    Output('Violin_Conversion_Rates', 'figure'),
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
        color=colors,
        hover_data=df.columns,
        color_discrete_sequence = color_palette
    )

   # fig.update_traces(marker=dict(color=filtered_df['Conversion Rate'].apply(lambda x: 'red' if x < 0 else 'green')))

    fig.update_layout(
        title=f"Conversion Rates by Customer Segment in {selected_countries}",
        xaxis_title="Customer Segment",
        yaxis_title="Conversion Rate (%)",
        legend_title="Segment",
        font=dict(family="Arial", size=12, color="black"),
        plot_bgcolor="white",
        hoverlabel=dict(font=dict(family="Arial", size=12)),
        margin=dict(l=50, r=50, t=80, b=50)
    )

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
    """
    import dash_daq as daq

    daq.Gauge(
    color={"gradient":True,"ranges":{"green":[0,6],"yellow":[6,8],"red":[8,10]}},
    value=2,
    label='Default',
    max=10,
    min=0,
)

    """
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



