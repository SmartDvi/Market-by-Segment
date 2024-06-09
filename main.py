
# calling dependencies
import pandas as pd
import dash
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Output, Input, State
import plotly.graph_objects as go
from dash import callback
import dash_bootstrap_components as dbc
import re
import pages



# loading the CsV
df = pd.read_csv('C:\\Users\\Moritus Peters\\Documents\\Datasets\\Sale Market Data\\Financials.csv')

# varifying data and checking for leading and trailing space
df.columns = df.columns.str.strip()

# fetching the information about the dataset
#df.info()

#************************************************************************************************
# data cleaning
#****************************************************************************************

def clean_and_convert_to_numeric(df, columns):
    for column in columns:
        # Ensure the column is of string type
        df[column] = df[column].astype(str)

        # Remove non-numeric characters and convert to numeric type
        df[column] = pd.to_numeric(df[column].str.replace(r'[^\d.]', '', regex=True), errors='coerce')
    return df


# Example usage:
columns_to_convert = ['Units Sold', 'Manufacturing Price', 'Sale Price', 'Gross Sales', 'Discounts', 'Sales', 'COGS',
                      'Profit']
df_cleaned = clean_and_convert_to_numeric(df, columns_to_convert)

# Verify the data types after conversion
#print(df_cleaned.dtypes)

# extracting day name from the date
df['Day of the week'] = pd.to_datetime(df['Date']).dt.day_name()

px.defaults.template = 'ggplot2'
external_css = ["https://cdn.jsdelivr.net/npm/bootstrap@5.3.1/dist/css/bootstrap.min.css"]

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.DARKLY] )



siderbar = dbc.Nav(
    [
        dbc.NavLink(
            [
                html.Div(page['name'], className='ms-2'),
            ],
            href=page['path'],
            active='exact',
        )
        for page in dash.page_registry.values()
    ],
    vertical=True,
    pills=True,
    className='bg-light',
)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div("Market Segment financial/ Business Analysis Dashboard",
                           style={'fontSize': 30, 'textAlign': 'center'}))
    ]),
    html.Hr(),

    dbc.Row(
        [
            dbc.Col(
                [
                    siderbar
                ], xs=4, sm=4, md=2, lg=2, xl=2, xxl=2
            ),
            dbc.Col(
                [
                    dash.page_container
                ], xs=8, sm=8, md=10, lg=10, xl=10, xxl=10

            )
        ]
    )

], fluid=True)



if __name__=="__main__":
    app.run(debug=True, port=7080)