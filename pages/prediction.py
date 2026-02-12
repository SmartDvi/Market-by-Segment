import dash
from dash import dcc, html, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd

from utils import (
    df,
    df_with_clusters,
    predict_profit,
    predict_loss_prob,
    importance_df,
    residual_df,
    reg_r2,
    clf_acc,
    clf_f1,
    clf_precision,
    clf_recall,
    clf_cm,
    clf_auc,
    t_stat,
    p_value
)

dash.register_page(__name__, path='/prediction',
                   name="🤖 Predictive Analytics", order=4)

# ------------------------------------------------------------------
# COLORS
# ------------------------------------------------------------------
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#73AB84',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'dark': '#2D3047'
}

# Safe fallback values
if importance_df.empty:
    importance_df = pd.DataFrame({
        "Feature": ["N/A"],
        "Importance": [0]
    })

if residual_df.empty:
    residual_df = pd.DataFrame({
        "Predicted": [0],
        "Residual": [0]
    })

# ------------------------------------------------------------------
# LAYOUT
# ------------------------------------------------------------------
layout = dbc.Container([

    # ==============================================================
    # HEADER
    # ==============================================================
    dbc.Row([
        dbc.Col([
            html.H1("Predictive Analytics & Risk Assessment",
                    className="fw-bold mb-2",
                    style={'color': COLORS['dark']}),
            html.P("Machine learning models trained on real sales data.",
                   className="text-muted mb-4")
        ])
    ]),

    # ==============================================================
    # KPI CARDS
    # ==============================================================
    dbc.Row([

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("R² (Random Forest)", className="text-muted"),
                    html.H3(f"{reg_r2:.3f}", className="text-success fw-bold"),
                    html.Small("Profit prediction accuracy",
                               className="text-muted")
                ])
            ], className="shadow-sm border-0 text-center h-100")
        ], md=3, sm=6, xs=12, className="mb-4"),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Top Driver of Profit",
                            className="text-muted"),
                    html.H5(
                        importance_df.iloc[0]['Feature'],
                        className="text-warning fw-bold"
                    ),
                    html.Small(
                        f"Importance: {importance_df.iloc[0]['Importance']:.3f}",
                        className="text-muted"
                    )
                ])
            ], className="shadow-sm border-0 text-center h-100")
        ], md=3, sm=6, xs=12, className="mb-4"),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Avg |Residual|", className="text-muted"),
                    html.H3(
                        f"${residual_df['Residual'].abs().mean():,.0f}",
                        className="text-info fw-bold"
                    ),
                    html.Small("Model error magnitude",
                               className="text-muted")
                ])
            ], className="shadow-sm border-0 text-center h-100")
        ], md=3, sm=6, xs=12, className="mb-4"),

    ], className="mb-4"),

    # ==============================================================
    # FEATURE IMPORTANCE + CONFUSION MATRIX
    # ==============================================================
    dbc.Row([

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.H5("🔝 Feature Importance",
                            className="fw-bold mb-0")),
                dbc.CardBody([
                    dcc.Graph(
                        figure=px.bar(
                            importance_df.sort_values(
                                "Importance", ascending=True).head(8),
                            x="Importance",
                            y="Feature",
                            orientation="h",
                            color="Importance",
                            color_continuous_scale="Viridis"
                        ).update_layout(
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            margin=dict(l=20, r=20, t=20, b=20)
                        ),
                        config={"displayModeBar": False}
                    )
                ])
            ], className="shadow-sm border-0 h-100")
        ], md=6, xs=12, className="mb-4"),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.H5("📉 Confusion Matrix",
                            className="fw-bold mb-0")),
                dbc.CardBody([
                    dcc.Graph(
                        figure=px.imshow(
                            clf_cm,
                            text_auto=True,
                            color_continuous_scale="Blues",
                            x=["No Loss", "Loss"],
                            y=["No Loss", "Loss"]
                        ).update_layout(
                            xaxis_title="Predicted",
                            yaxis_title="Actual",
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            margin=dict(l=20, r=20, t=20, b=20)
                        ),
                        config={"displayModeBar": False}
                    )
                ])
            ], className="shadow-sm border-0 h-100")
        ], md=6, xs=12, className="mb-4")

    ]),

    # ==============================================================
    # PROFIT SIMULATOR
    # ==============================================================
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🎯 Profit Simulator",
                                       className="fw-bold")),
                dbc.CardBody([

                    dbc.Row([
                        dbc.Col([
                            html.Label("Units Sold"),
                            dcc.Slider(
                                id="units_slider",
                                min=500, max=3000,
                                step=100, value=1500
                            ),
                            html.Label("Manufacturing Price",
                                       className="mt-3"),
                            dcc.Slider(
                                id="mp_slider",
                                min=50, max=200,
                                step=5, value=100
                            )
                        ], md=6),

                        dbc.Col([
                            html.Label("Discount %"),
                            dcc.Slider(
                                id="discount_slider",
                                min=0, max=30,
                                step=2, value=10
                            ),
                            html.Label("Efficiency Multiplier",
                                       className="mt-3"),
                            dcc.Slider(
                                id="efficiency_slider",
                                min=1.0, max=3.0,
                                step=0.1, value=1.8
                            )
                        ], md=6),
                    ]),

                    html.Label("Segment", className="mt-3"),
                    dcc.Dropdown(
                        id="segment_pred",
                        options=[
                            {"label": s, "value": s}
                            for s in df["Segment"].unique()
                        ],
                        value=df["Segment"].unique()[0],
                        clearable=False
                    ),

                    html.Hr(),

                    html.Div(
                        id="prediction_output",
                        className="p-4 text-center rounded shadow-sm",
                        style={
                            "background": COLORS["primary"],
                            "color": "white"
                        }
                    )

                ])
            ], className="shadow-sm border-0")
        ])
    ]),

    # ==============================================================
    # RESIDUALS + CLUSTERS
    # ==============================================================
    dbc.Row([

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("📊 Residual Plot",
                                       className="fw-bold")),
                dbc.CardBody([
                    dcc.Graph(
                        figure=px.scatter(
                            residual_df,
                            x="Predicted",
                            y="Residual",
                            trendline="ols"
                        ).add_hline(y=0, line_dash="dash").update_layout(
                            plot_bgcolor="white",
                            paper_bgcolor="white"
                        ),
                        config={"displayModeBar": False}
                    )
                ])
            ], className="shadow-sm border-0 h-100")
        ], md=6, xs=12, className="mb-4"),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🧠 K-Means Clusters",
                                       className="fw-bold")),
                dbc.CardBody([
                    dcc.Graph(
                        figure=px.scatter(
                            df_with_clusters,
                            x="Sales",
                            y="Profit",
                            color=df_with_clusters["Cluster"].astype(str),
                            size="Units Sold",
                            hover_data=["Product", "Country"]
                        ).update_layout(
                            plot_bgcolor="white",
                            paper_bgcolor="white"
                        ),
                        config={"displayModeBar": False}
                    )
                ])
            ], className="shadow-sm border-0 h-100")
        ], md=6, xs=12, className="mb-4"),

    ]),

    # ==============================================================
    # STATISTICAL TEST
    # ==============================================================
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("📉 Discount Impact Test",
                                       className="fw-bold")),
                dbc.CardBody([
                    html.H6(
                        f"T-statistic: {t_stat:.3f} | p-value: {p_value:.4f}"
                    ),
                    html.P(
                        "Significant impact detected."
                        if p_value < 0.05
                        else "No statistically significant impact detected.",
                        className="text-muted"
                    )
                ])
            ], className="shadow-sm border-0")
        ])
    ])

], fluid=True, className="py-4",
   style={"backgroundColor": "#F8F9FA"})


# ------------------------------------------------------------------
# CALLBACK – PROFIT SIMULATOR
# ------------------------------------------------------------------
@callback(
    Output("prediction_output", "children"),
    Input("units_slider", "value"),
    Input("mp_slider", "value"),
    Input("discount_slider", "value"),
    Input("efficiency_slider", "value"),
    Input("segment_pred", "value"),
)
def update_prediction(units, mp, disc, eff, seg):

    profit = predict_profit(units, mp, disc, eff, seg)
    loss_prob = predict_loss_prob(units, mp, disc, eff, seg)

    if loss_prob > 0.6:
        badge = dbc.Badge("High Risk", color="danger")
    elif loss_prob > 0.3:
        badge = dbc.Badge("Medium Risk", color="warning")
    else:
        badge = dbc.Badge("Low Risk", color="success")

    return [
        html.H3(f"${profit:,.0f}", className="fw-bold"),
        html.P("Predicted Profit"),
        html.Div([
            html.Span(f"Loss Probability: {loss_prob:.1%} "),
            badge
        ])
    ]
