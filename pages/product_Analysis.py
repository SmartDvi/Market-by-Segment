import dash
import pandas as pd
from dash import dcc, html, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash_ag_grid import AgGrid
from utils import (
    df, get_kpi_metrics, profit_product_year, profit_month_year,
    product_efficiency, product_contribution, profit_by_country,
    discount_profit_table
)

dash.register_page(__name__, path='/product_analysis', name="📊 Product Analysis", order=2)

COLORS = {
    'primary': '#2E86AB', 'secondary': '#A23B72', 'success': '#73AB84',
    'warning': '#F18F01', 'danger': '#C73E1D', 'dark': '#2D3047'
}

def kpi_card(title, value, icon, color, suffix=""):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className=f"fas fa-{icon} fa-2x", style={'color': COLORS[color]}),
                html.Div([
                    html.H6(title, className="text-muted mb-1"),
                    html.H3(f"{value}{suffix}", className="fw-bold")
                ], className="ms-3")
            ], className="d-flex align-items-center")
        ])
    ], className="border-0 shadow-sm h-100")

layout = dbc.Container([
    # Header
    dbc.Row([dbc.Col([html.H1("Product Performance Analytics", className="fw-bold mb-2",
                              style={'color': COLORS['dark']}),
                      html.P("Analyze product profitability, sales trends, and regional distribution.",
                             className="text-muted mb-4")])]),

    # KPI Row
    dbc.Row([
        dbc.Col(kpi_card("Total Sales", f"${get_kpi_metrics()['total_sales']:,.0f}", "dollar-sign", "primary"),
                xs=12, sm=6, md=3, className="mb-4"),
        dbc.Col(kpi_card("Total Profit", f"${get_kpi_metrics()['total_profit']:,.0f}", "chart-line", "success"),
                xs=12, sm=6, md=3, className="mb-4"),
        dbc.Col(kpi_card("Units Sold", f"{get_kpi_metrics()['total_units']:,.0f}", "box", "warning"),
                xs=12, sm=6, md=3, className="mb-4"),
        dbc.Col(kpi_card("Avg Margin", f"{get_kpi_metrics()['avg_profit_margin']:.1f}", "percent", "secondary", "%"),
                xs=12, sm=6, md=3, className="mb-4"),
    ]),

    # Filters Row (simplified)
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader(html.H6("🌍 Region & Product", className="fw-bold mb-0")),
                          dbc.CardBody([
                              html.Label("Country", className="fw-bold small"),
                              dcc.Dropdown(id="product_country", options=[{"label": c, "value": c}
                                                                           for c in sorted(df["Country"].unique())],
                                           value=list(df["Country"].unique()), multi=True, className="mb-3"),
                              html.Label("Product", className="fw-bold small"),
                              dcc.Dropdown(id="product_name", options=[{"label": p, "value": p}
                                                                        for p in sorted(df["Product"].unique())],
                                           value=list(df["Product"].unique()), multi=True, className="mb-3"),
                          ])], className="shadow-sm border-0 h-100"), xs=12, md=4, className="mb-4"),
        dbc.Col(dbc.Card([dbc.CardHeader(html.H6("📊 Segment & Profit", className="fw-bold mb-0")),
                          dbc.CardBody([
                              html.Label("Segment", className="fw-bold small"),
                              dcc.Dropdown(id="product_segment", options=[{"label": s, "value": s}
                                                                           for s in sorted(df["Segment"].unique())],
                                           value=list(df["Segment"].unique()), multi=True, className="mb-3"),
                              html.Label("Profit Range ($)", className="fw-bold small"),
                              dcc.RangeSlider(id="product_profit_range", min=df["Profit"].min(), max=df["Profit"].max(),
                                              step=5000, value=[df["Profit"].min(), df["Profit"].max()],
                                              marks={int(df["Profit"].min()): f'{int(df["Profit"].min()/1000)}k',
                                                     int(df["Profit"].max()): f'{int(df["Profit"].max()/1000)}k'},
                                              className="mt-2")
                          ])], className="shadow-sm border-0 h-100"), xs=12, md=4, className="mb-4"),
        dbc.Col(dbc.Card([dbc.CardHeader(html.H6("📅 Time Period", className="fw-bold mb-0")),
                          dbc.CardBody([
                              html.Label("Year", className="fw-bold small"),
                              dcc.Dropdown(id="product_year", options=[{"label": str(y), "value": y}
                                                                        for y in sorted(df["Year"].unique())],
                                           value=list(df["Year"].unique()), multi=True, className="mb-3"),
                              html.Label("Date Range", className="fw-bold small"),
                              dcc.DatePickerRange(id="product_date_range", min_date_allowed=df["Date"].min(),
                                                  max_date_allowed=df["Date"].max(),
                                                  start_date=df["Date"].min(), end_date=df["Date"].max(),
                                                  className="w-100")
                          ])], className="shadow-sm border-0 h-100"), xs=12, md=4, className="mb-4"),
    ]),

    # Row 1: Monthly Trend + Product Pie
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader(html.H6("📈 Monthly Profit Trend", className="fw-bold mb-0")),
                          dbc.CardBody([dcc.Graph(id="product_monthly_chart",
                                                  config={'displayModeBar': False, 'responsive': True},
                                                  style={"height": "350px", "width": "100%"})],
                                       style={"padding": "0.5rem"})],
                        className="shadow-sm border-0 h-100"), xs=12, md=8, className="mb-4"),
        dbc.Col(dbc.Card([dbc.CardHeader(html.H6("🎯 Product Profit Share", className="fw-bold mb-0")),
                          dbc.CardBody([dcc.Graph(id="product_pie_chart",
                                                  config={'displayModeBar': False, 'responsive': True},
                                                  style={"height": "350px", "width": "100%"})],
                                       style={"padding": "0.5rem"})],
                        className="shadow-sm border-0 h-100"), xs=12, md=4, className="mb-4"),
    ]),

    # Row 2: Profit by Product & Year + Monthly Profit by Year
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader(html.H6("📊 Profit by Product & Year", className="fw-bold mb-0")),
                          dbc.CardBody([dcc.Graph(id="product_profit_year_chart",
                                                  config={'displayModeBar': False, 'responsive': True},
                                                  style={"height": "350px", "width": "100%"})],
                                       style={"padding": "0.5rem"})],
                        className="shadow-sm border-0 h-100"), xs=12, md=6, className="mb-4"),
        dbc.Col(dbc.Card([dbc.CardHeader(html.H6("📅 Monthly Profit by Year", className="fw-bold mb-0")),
                          dbc.CardBody([dcc.Graph(id="product_monthly_year_chart",
                                                  config={'displayModeBar': False, 'responsive': True},
                                                  style={"height": "350px", "width": "100%"})],
                                       style={"padding": "0.5rem"})],
                        className="shadow-sm border-0 h-100"), xs=12, md=6, className="mb-4"),
    ]),

    # Row 3: Geographic Profit Map + Country Profit vs Discount
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader(html.H6("🌍 Profit by Country", className="fw-bold mb-0")),
                          dbc.CardBody([dcc.Graph(id="product_geo_map",
                                                  config={'displayModeBar': True, 'responsive': True},
                                                  style={"height": "400px", "width": "100%"})],
                                       style={"padding": "0.5rem"})],
                        className="shadow-sm border-0 h-100"), xs=12, md=6, className="mb-4"),
        dbc.Col(dbc.Card([dbc.CardHeader(html.H6("📉 Country Profit vs Discount", className="fw-bold mb-0")),
                          dbc.CardBody([dcc.Graph(id="product_country_scatter",
                                                  config={'displayModeBar': False, 'responsive': True},
                                                  style={"height": "400px", "width": "100%"})],
                                       style={"padding": "0.5rem"})],
                        className="shadow-sm border-0 h-100"), xs=12, md=6, className="mb-4"),
    ]),

    # Row 4: Product Efficiency Bar + (maybe a small insight card)
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader(html.H6("⚙️ Product Efficiency (Profit/Unit)", className="fw-bold mb-0")),
                          dbc.CardBody([dcc.Graph(id="product_efficiency_chart",
                                                  config={'displayModeBar': False, 'responsive': True},
                                                  style={"height": "350px", "width": "100%"})],
                                       style={"padding": "0.5rem"})],
                        className="shadow-sm border-0 h-100"), xs=12, md=8, className="mb-4"),
        dbc.Col(dbc.Card([dbc.CardHeader(html.H6("📌 Discount Impact", className="fw-bold mb-0")),
                          dbc.CardBody([
                              html.H5("T‑Test: Low vs High Discount", className="fw-bold"),
                              html.P(id="product_ttest_result", className="text-muted")
                          ], style={"padding": "1rem"})],
                        className="shadow-sm border-0 h-100"), xs=12, md=4, className="mb-4"),
    ]),

    # Row 5: Discount Rate vs Profit Table (AG Grid)
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader(html.H6("📋 Discount Rate vs Profit – Detailed", className="fw-bold mb-0")),
                          dbc.CardBody([
                              AgGrid(id="product_discount_table",
                                     rowData=discount_profit_table.to_dict("records"),
                                     columnDefs=[
                                         {"headerName": "Year", "field": "Year", "filter": True, "sortable": True},
                                         {"headerName": "Country", "field": "Country", "filter": True, "sortable": True},
                                         {"headerName": "Product", "field": "Product", "filter": True, "sortable": True},
                                         {"headerName": "Discount Rate (%)", "field": "Discount_Rate",
                                          "filter": "agNumberColumnFilter", "sortable": True},
                                         {"headerName": "Profit", "field": "Profit",
                                          "filter": "agNumberColumnFilter", "sortable": True,
                                          "valueFormatter": {"function": "params.value ? '$' + params.value.toLocaleString() : ''"}}
                                     ],
                                     defaultColDef={"resizable": True, "flex": 1},
                                     dashGridOptions={"pagination": True, "paginationPageSize": 10},
                                     style={"height": "400px", "width": "100%"},
                                     className="ag-theme-alpine")
                          ], style={"padding": "0.5rem"})],
                        className="shadow-sm border-0"), xs=12, className="mb-4"),
    ])
], fluid=True, className="py-4", style={'backgroundColor': '#F8F9FA'})

# ------------------------------------------------------------
# CALLBACKS
# ------------------------------------------------------------
@callback(
    [Output("product_monthly_chart", "figure"),
     Output("product_pie_chart", "figure"),
     Output("product_profit_year_chart", "figure"),
     Output("product_monthly_year_chart", "figure"),
     Output("product_geo_map", "figure"),
     Output("product_country_scatter", "figure"),
     Output("product_efficiency_chart", "figure"),
     Output("product_ttest_result", "children"),
     Output("product_discount_table", "rowData")],
    [Input("product_country", "value"),
     Input("product_name", "value"),
     Input("product_segment", "value"),
     Input("product_profit_range", "value"),
     Input("product_year", "value"),
     Input("product_date_range", "start_date"),
     Input("product_date_range", "end_date")]
)
def update_product_charts(countries, products, segments, profit_range, years, start_date, end_date):
    # Filter data
    dff = df.copy()
    dff = dff[dff["Country"].isin(countries)]
    dff = dff[dff["Product"].isin(products)]
    dff = dff[dff["Segment"].isin(segments)]
    dff = dff[dff["Year"].isin(years)]
    dff = dff[(dff["Profit"] >= profit_range[0]) & (dff["Profit"] <= profit_range[1])]
    if start_date and end_date:
        dff = dff[(dff["Date"] >= start_date) & (dff["Date"] <= end_date)]

    # 1. Monthly Profit Trend
    monthly = dff.groupby(["Year", "Month"])["Profit"].sum().reset_index()
    monthly_fig = px.line(monthly, x="Month", y="Profit", color="Year",
                          color_discrete_sequence=[COLORS['primary'], COLORS['secondary']],
                          labels={"Month": "Month", "Profit": "Profit ($)"})
    monthly_fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                              margin=dict(l=40, r=40, t=40, b=40))

    # 2. Product Pie (filtered)
    pie_data = dff.groupby("Product")["Profit"].sum().reset_index()
    pie_fig = px.pie(pie_data, names="Product", values="Profit", hole=0.4,
                     color_discrete_sequence=px.colors.qualitative.Set3)
    pie_fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20))

    # 3. Profit by Product & Year (filtered)
    prod_year = dff.groupby(["Year", "Product"])["Profit"].sum().reset_index()
    prod_year_fig = px.bar(prod_year, x="Product", y="Profit", color="Year", barmode="group",
                           color_discrete_sequence=[COLORS['primary'], COLORS['secondary']],
                           labels={"Profit": "Profit ($)"})
    prod_year_fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                                margin=dict(l=40, r=40, t=40, b=40))

    # 4. Monthly Profit by Year (filtered)
    month_year = dff.groupby(["Year", "Month_Name"])["Profit"].sum().reset_index()
    # ensure correct order
    month_order = ["January","February","March","April","May","June","July","August",
                   "September","October","November","December"]
    month_year["Month_Name"] = pd.Categorical(month_year["Month_Name"], categories=month_order, ordered=True)
    month_year = month_year.sort_values("Month_Name")
    month_year_fig = px.bar(month_year, x="Month_Name", y="Profit", color="Year", barmode="group",
                            color_discrete_sequence=[COLORS['primary'], COLORS['secondary']],
                            labels={"Profit": "Profit ($)", "Month_Name": "Month"})
    month_year_fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                                 margin=dict(l=40, r=40, t=40, b=40))

    # 5. Geographic Map (filtered)
    geo_data = dff.groupby("Country")["Profit"].sum().reset_index()
    country_map = {'United States of America': 'United States', 'USA': 'United States'}
    geo_data["Country_plot"] = geo_data["Country"].map(lambda x: country_map.get(x, x))
    geo_fig = px.choropleth(geo_data, locations="Country_plot", locationmode="country names",
                            color="Profit", hover_name="Country_plot",
                            color_continuous_scale="Viridis", projection="natural earth",
                            labels={"Profit": "Total Profit ($)"})
    geo_fig.update_geos(showcoastlines=True, coastlinecolor="Black", showland=True, landcolor="white")
    geo_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), coloraxis_colorbar=dict(title="Profit"))

    # 6. Country Profit vs Discount (filtered)
    country_scatter_data = dff.groupby("Country").agg(
        Total_Profit=("Profit", "sum"),
        Avg_Discount=("Discount_Rate", "mean"),
        Total_Sales=("Sales", "sum")
    ).reset_index()
    country_scatter_fig = px.scatter(country_scatter_data, x="Avg_Discount", y="Total_Profit",
                                     size="Total_Sales", text="Country",
                                     labels={"Avg_Discount": "Avg Discount Rate", "Total_Profit": "Total Profit ($)"})
    country_scatter_fig.update_traces(textposition="top center")
    country_scatter_fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                                      margin=dict(l=40, r=40, t=40, b=40))

    # 7. Product Efficiency (filtered – but we show overall; filtering would make it empty)
    # Use precomputed overall product_efficiency; no need to filter.
    eff_fig = px.bar(product_efficiency, x="Product", y="Avg_Profit_per_Unit",
                     color="Avg_Gross_Margin", color_continuous_scale="Viridis",
                     labels={"Avg_Profit_per_Unit": "Avg Profit per Unit ($)",
                             "Avg_Gross_Margin": "Gross Margin"})
    eff_fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                          margin=dict(l=40, r=40, t=40, b=40))

    # 8. T-Test result (overall, not filtered)
    from utils import t_stat, p_value
    ttest_text = f"T-statistic: {t_stat:.3f}, p-value: {p_value:.4f}<br>"
    if p_value < 0.05:
        ttest_text += "Significant difference: Discount strategy impacts profit."
    else:
        ttest_text += "No significant difference observed."

    # 9. Table data (filtered)
    table_data = dff[["Year", "Country", "Product", "Discount_Rate", "Profit"]].copy()
    table_data["Discount_Rate"] = (table_data["Discount_Rate"] * 100).round(2)
    table_data = table_data.sort_values(["Product", "Discount_Rate"])

    return (monthly_fig, pie_fig, prod_year_fig, month_year_fig,
            geo_fig, country_scatter_fig, eff_fig, html.Div(ttest_text),
            table_data.to_dict("records"))