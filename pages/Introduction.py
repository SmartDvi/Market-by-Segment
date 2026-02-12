import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash_ag_grid import AgGrid
import pandas as pd
from utils import df, get_kpi_metrics

dash.register_page(__name__, path='/', name="🏠 Introduction", order=1)

# Prepare preview data
preview_df = df[['Date', 'Segment', 'Country', 'Product', 'Discount Band',
                 'Units Sold', 'Sale Price', 'Sales', 'Profit']].copy()
preview_df['Date'] = preview_df['Date'].dt.strftime('%Y-%m-%d')
preview_df['Profit Margin %'] = (preview_df['Profit'] / preview_df['Sales'] * 100).round(1)
preview_df.rename(columns={'Sales': 'Sales'}, inplace=True)

column_defs = [
    {"headerName": col, "field": col, "filter": True, "sortable": True, "resizable": True}
    for col in preview_df.columns
]

layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Market by Segment – Analytics Dashboard", className="display-4 fw-bold mb-3",
                    style={'color': '#2D3047'}),
            html.P(
                "A comprehensive business intelligence solution built exclusively on your "
                "actual sales data. No placeholders, no synthetic numbers – every chart and "
                "insight is derived from the 700 records you provided.",
                className="lead text-muted mb-4"
            ),
        ])
    ]),

    # Key metrics row (from utils)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Total Sales", className="text-muted"),
                    html.H2(f"${get_kpi_metrics()['total_sales']:,.0f}",
                            className="text-primary fw-bold")
                ])
            ], className="shadow-sm border-0")
        ], xs=12, sm=6, md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Total Profit", className="text-muted"),
                    html.H2(f"${get_kpi_metrics()['total_profit']:,.0f}",
                            className="text-success fw-bold")
                ])
            ], className="shadow-sm border-0")
        ], xs=12, sm=6, md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Units Sold", className="text-muted"),
                    html.H2(f"{get_kpi_metrics()['total_units']:,.0f}",
                            className="text-warning fw-bold")
                ])
            ], className="shadow-sm border-0")
        ], xs=12, sm=6, md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Avg. Profit Margin", className="text-muted"),
                    html.H2(f"{get_kpi_metrics()['avg_profit_margin']:.1f}%",
                            className="text-info fw-bold")
                ])
            ], className="shadow-sm border-0")
        ], xs=12, sm=6, md=3),
    ], className="mb-5"),

    # About the project
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("📌 Project Overview", className="fw-bold")
                ], className="bg-white border-0"),
                dbc.CardBody([
                    html.P(
                        "This dashboard transforms your raw sales data into actionable "
                        "business intelligence. Every visualization, KPI, and machine learning "
                        "model is built exclusively from the columns present in your dataset: "
                        "Segment, Country, Product, Discount Band, Units Sold, Manufacturing Price, "
                        "Sale Price, Gross Sales, Discounts, Sales, COGS, Profit, and Date.",
                        className="mb-3"
                    ),
                    html.P(
                        "We've engineered additional business metrics (Gross Margin, Discount Rate, "
                        "Manufacturing Efficiency) directly from the original fields, ensuring "
                        "complete transparency and traceability.",
                        className="mb-3"
                    ),
                    html.P(
                        "The Predictive Analytics page uses XGBoost and SHAP to provide insights and forecasts based solely on the patterns"
                        "trained on your data to forecast profit and loss risk – no external or "
                        "fabricated data has been introduced.",
                        className="fw-bold"
                    )
                ])
            ], className="shadow-sm border-0")
        ])
    ], className="mb-5"),

    # Data preview
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("🔍 Raw Data Sample", className="fw-bold d-inline"),
                    html.Span(" (first 700 rows)", className="text-muted ms-2")
                ], className="bg-white border-0"),
                dbc.CardBody([
                    AgGrid(
                        rowData=preview_df.to_dict("records"),
                        columnDefs=column_defs,
                        defaultColDef={"filter": True, "sortable": True, "resizable": True},
                        dashGridOptions={"pagination": True, "paginationPageSize": 10},
                        style={"height": "450px", "width": "100%"},
                        className="ag-theme-alpine"
                    )
                ])
            ], className="shadow-sm border-0")
        ])
    ])
], fluid=True, className="py-4", style={'backgroundColor': '#F8F9FA'})