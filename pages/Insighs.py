import dash
from dash import dcc, html, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from utils import df, get_kpi_metrics, get_strategic_insights

dash.register_page(__name__, path='/Insighs', name="📈 Business Insights", order=3)

COLORS = {
    'primary': '#2E86AB', 'secondary': '#A23B72', 'success': '#73AB84',
    'warning': '#F18F01', 'danger': '#C73E1D', 'dark': '#2D3047'
}

strategic_insights = get_strategic_insights()

layout = dbc.Container([
    dbc.Row([dbc.Col([html.H1("Executive Business Intelligence", className="fw-bold mb-2",
                              style={'color': COLORS['dark']}),
                      html.P("Key performance drivers and strategic opportunities – all computed from your data.",
                             className="text-muted mb-4")])]),
    # KPI Cards
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([html.H5("Total Profit", className="text-muted"),
                                        html.H2(f"${get_kpi_metrics()['total_profit']:,.0f}",
                                                className="text-success fw-bold"),
                                        html.Small("overall", className="text-muted")])],
                        className="shadow-sm border-0 text-center h-100"), xs=12, sm=6, md=3, className="mb-4"),
        dbc.Col(dbc.Card([dbc.CardBody([html.H5("Profit Margin", className="text-muted"),
                                        html.H2(f"{get_kpi_metrics()['avg_profit_margin']:.1f}%",
                                                className="text-primary fw-bold")])],
                        className="shadow-sm border-0 text-center h-100"), xs=12, sm=6, md=3, className="mb-4"),
        dbc.Col(dbc.Card([dbc.CardBody([html.H5("Profitable Products", className="text-muted"),
                                        html.H2(f"{get_kpi_metrics()['profitable_products']}",
                                                className="text-success fw-bold"),
                                        html.Small(f"out of {get_kpi_metrics()['total_products']}",
                                                   className="text-muted")])],
                        className="shadow-sm border-0 text-center h-100"), xs=12, sm=6, md=3, className="mb-4"),
        dbc.Col(dbc.Card([dbc.CardBody([html.H5("Loss‑Making", className="text-muted"),
                                        html.H2(f"{get_kpi_metrics()['loss_making_products']}",
                                                className="text-danger fw-bold")])],
                        className="shadow-sm border-0 text-center h-100"), xs=12, sm=6, md=3, className="mb-4"),
    ]),
    # Strategic Insights + Heatmap
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader([html.H5("💡 Strategic Insights", className="fw-bold mb-0"),
                                          html.Small("Derived from your data", className="text-muted")]),
                          dbc.CardBody([html.Ul([html.Li(insight, className="mb-2") for insight in strategic_insights],
                                                className="list-unstyled"),
                                        html.Hr(),
                                        html.P("Recommendation:", className="fw-bold mb-2"),
                                        html.P("Focus marketing and discount optimization on the top performing "
                                               "segment and country while reviewing pricing/manufacturing costs "
                                               "for underperforming products.", className="text-muted")])],
                        className="shadow-sm border-0 h-100"), xs=12, md=4, className="mb-4"),
        dbc.Col(dbc.Card([dbc.CardHeader(html.H5("🔥 Profit / Segment / Discount", className="fw-bold mb-0")),
                          dbc.CardBody([dcc.Graph(id="insights_heatmap",
                                                  config={'displayModeBar': False, 'responsive': True},
                                                  style={"height": "400px", "width": "100%"})],
                                       style={"padding": "0.5rem"})],
                        className="shadow-sm border-0 h-100"), xs=12, md=8, className="mb-4"),
    ]),
    # Quarterly Trends + Regional Performance
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader(html.H5("📅 Quarterly Profit & Sales", className="fw-bold mb-0")),
                          dbc.CardBody([dcc.Graph(id="insights_quarterly",
                                                  config={'displayModeBar': False, 'responsive': True},
                                                  style={"height": "400px", "width": "100%"})],
                                       style={"padding": "0.5rem"})],
                        className="shadow-sm border-0 h-100"), xs=12, md=6, className="mb-4"),
        dbc.Col(dbc.Card([dbc.CardHeader(html.H5("🌍 Regional Performance", className="fw-bold mb-0")),
                          dbc.CardBody([dcc.Graph(id="insights_regional_bar",
                                                  config={'displayModeBar': False, 'responsive': True},
                                                  style={"height": "400px", "width": "100%"})],
                                       style={"padding": "0.5rem"})],
                        className="shadow-sm border-0 h-100"), xs=12, md=6, className="mb-4"),
    ]),
    # Efficiency Scatter
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader(html.H5("⚙️ Efficiency vs Profit Margin", className="fw-bold mb-0")),
                          dbc.CardBody([dcc.Graph(id="insights_efficiency_scatter",
                                                  config={'displayModeBar': False, 'responsive': True},
                                                  style={"height": "450px", "width": "100%"})],
                                       style={"padding": "0.5rem"})],
                        className="shadow-sm border-0"), xs=12, className="mb-4"),
    ])
], fluid=True, className="py-4", style={'backgroundColor': '#F8F9FA'})

@callback(Output("insights_heatmap", "figure"), Input("insights_heatmap", "id"))
def update_heatmap(_):
    heat = df.groupby(['Segment', 'Discount Band'])['Profit'].mean().reset_index()
    pivot = heat.pivot(index='Segment', columns='Discount Band', values='Profit')
    fig = px.imshow(pivot, text_auto=True, color_continuous_scale='RdYlGn',
                    labels=dict(color="Avg Profit $"))
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                      xaxis_title="Discount Band", yaxis_title="Segment",
                      margin=dict(l=40, r=40, t=40, b=40))
    return fig

@callback(Output("insights_quarterly", "figure"), Input("insights_quarterly", "id"))
def update_quarterly(_):
    q_data = df.groupby(['Year', 'Quarter'])[['Profit', 'Sales']].sum().reset_index()
    q_data['Quarter_Label'] = q_data['Year'].astype(str) + ' Q' + q_data['Quarter'].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=q_data['Quarter_Label'], y=q_data['Sales'],
                         name='Sales', marker_color=COLORS['primary'], opacity=0.6))
    fig.add_trace(go.Scatter(x=q_data['Quarter_Label'], y=q_data['Profit'],
                             name='Profit', mode='lines+markers',
                             line=dict(color=COLORS['success'], width=3), yaxis='y2'))
    fig.update_layout(yaxis=dict(title="Sales ($)"), yaxis2=dict(title="Profit ($)", overlaying="y", side="right"),
                      plot_bgcolor='white', paper_bgcolor='white',
                      margin=dict(l=40, r=60, t=40, b=40))
    return fig

@callback(Output("insights_regional_bar", "figure"), Input("insights_regional_bar", "id"))
def update_regional(_):
    reg = df.groupby('Country')['Profit'].sum().reset_index().sort_values('Profit', ascending=False)
    fig = px.bar(reg, x='Country', y='Profit', color='Profit', color_continuous_scale='Blues',
                 labels={'Profit': 'Total Profit ($)'})
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                      xaxis_title="Country", yaxis_title="Total Profit ($)",
                      margin=dict(l=40, r=40, t=40, b=40), coloraxis_showscale=False)
    return fig

@callback(Output("insights_efficiency_scatter", "figure"), Input("insights_efficiency_scatter", "id"))
def update_efficiency(_):
    sample = df.sample(min(300, len(df)), random_state=42)
    fig = px.scatter(sample, x='Manufacturing_Efficiency', y='Gross_Margin',
                     color='Segment', size='Units Sold', hover_name='Product',
                     labels={'Manufacturing_Efficiency': 'Efficiency', 'Gross_Margin': 'Gross Margin'},
                     color_discrete_sequence=[COLORS['primary'], COLORS['secondary'], COLORS['success']])
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                      xaxis_title="Manufacturing Efficiency", yaxis_title="Gross Margin %",
                      margin=dict(l=40, r=40, t=40, b=40))
    fig.update_yaxes(tickformat=".0%")
    return fig