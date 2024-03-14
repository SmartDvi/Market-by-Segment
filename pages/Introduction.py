import dash
from dash import html

dash.register_page(__name__, path='/', name="Introduction 😃")

####################### PAGE LAYOUT #############################
layout = html.Div(children=[
    html.Div(children=[
        html.H2(" `Market Segment and Country/Region`."),
        "THis .",
        html.Br(),html.Br(),
        "th",
        html.Br(), html.Br(),
        "While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.",
    ]),
    html.Div(children=[
        html.Br(),
        html.H2("Data Variables"),
        html.B("Segment: "), "the various sectors",
        html.Br(),
        html.B("Country: "), "country locations",
        html.Br(),
        html.B("Product: "), "company product",
        html.Br(),
        html.B("Discount Brand: "), "column express the range discount categories",
        html.Br(),
        html.B("Unit sold: "), "Contains the unit sold",
        html.Br(),
        html.B("Manufacturing price: "), "cost of manufacturing a product ",
        html.Br(),
        html.B("Sales price: "), "the Sales price of each product sold",
        html.Br(),
        html.B("Gross sales: "), "The sold amount",
        html.Br(),
        html.B("Discount: "), "discount ",
        html.Br(),
        html.B("Sales: "), ")",
    ])
], className="bg-light p-4 m-2")