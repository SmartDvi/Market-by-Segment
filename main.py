import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME,
        'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
    ],
    suppress_callback_exceptions=True
)

app.title = "Market by Segment | Business Analytics"

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * { font-family: 'Inter', sans-serif; }
            body { background-color: #F8F9FA; }
            .ag-theme-alpine {
                --ag-header-background-color: #2D3047;
                --ag-header-foreground-color: white;
            }
            .nav-link.active {
                background: linear-gradient(90deg, #2E86AB, #3DA0C9);
                color: white !important;
            }
            .card { transition: transform 0.2s, box-shadow 0.2s; }
            .card:hover { transform: translateY(-5px); box-shadow: 0 10px 25px rgba(0,0,0,0.1) !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
'''

# Sidebar – href must exactly match the `path` set in each page's register_page()
sidebar = dbc.Nav([
    dbc.NavLink([html.I(className="fas fa-home me-2"), "Introduction"],
                href="/", active="exact"),
    dbc.NavLink([html.I(className="fas fa-chart-bar me-2"), "Product Analysis"],
                href="/product_analysis", active="exact"),
    dbc.NavLink([html.I(className="fas fa-lightbulb me-2"), "Business Insights"],
                href="/Insighs", active="exact"),          # note capital I
    dbc.NavLink([html.I(className="fas fa-robot me-2"), "Predictive Analytics"],
                href="/prediction", active="exact"),
], vertical=True, pills=True, className="sidebar bg-dark py-4")

sidebar_container = html.Div(
    sidebar,
    style={
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'bottom': 0,
        'width': '250px',
        'background': 'linear-gradient(180deg, #2D3047 0%, #1A1C2D 100%)',
        'padding': '20px 10px',
        'zIndex': 1000
    }
)

content_container = html.Div(
    dash.page_container,
    style={'marginLeft': '250px', 'padding': '20px 30px'}
)

app.layout = dbc.Container([
    sidebar_container,
    content_container
], fluid=True, className="px-0")

if __name__ == "__main__":
    app.run(debug=True, port=7080)



