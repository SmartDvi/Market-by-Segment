import dash
from dash import html

dash.register_page(__name__, path='/', name="Introduction 😃")

# Page layout
layout = html.Div(children=[
    html.Div(children=[
        html.H2("Market Segment and Country/Region Analysis"),
        html.P("This project aims to provide descriptive and predictive analysis of sales and financial data, offering valuable insights into market segments, country/regional performance, product contribution, profitability, and more. Leveraging Dash, a Python framework for building analytical web applications, and Plotly, a powerful graphing library, this project delivers interactive visualizations to facilitate data exploration and decision-making."),

        html.H3("Data Variables"),
        html.B("Segment: "), html.Span("Represents various sectors or market segments."),
        html.Br(),
        html.B("Country: "), html.Span("Denotes country or regional locations."),
        html.Br(),
        html.B("Product: "), html.Span("Signifies company products."),
        html.Br(),
        html.B("Discount Band: "), html.Span("Encompasses ranges of discount categories."),
        html.Br(),
        html.B("Unit Sold: "), html.Span("Indicates the quantity of units sold."),
        html.Br(),
        html.B("Manufacturing Price: "), html.Span("Refers to the cost of manufacturing a product."),
        html.Br(),
        html.B("Sales Price: "), html.Span("Represents the selling price of each product."),
        html.Br(),
        html.B("Gross Sales: "), html.Span("Denotes the total sales amount."),
        html.Br(),
        html.B("Discount: "), html.Span("Signifies the discount applied."),
        html.Br(),
        html.B("Sales: "), html.Span("Represents the sales amount."),
        html.Br(),
        html.B("Profit: "), html.Span("Indicates the profit generated."),
    ])
], className="bg-light p-4 m-2")
