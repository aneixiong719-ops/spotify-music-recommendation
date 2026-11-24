"""Dash dashboard — data story in three acts.

1. Data readiness: how many listens can actually be used (the denominator story).
2. Q1 Taste profiles: what each listener sounds like, mood mix, diversity.
3. Q2 Recommendations: model vs. popularity baseline + top-10 picks per user.

Run:  python app.py   then open http://127.0.0.1:8050
"""
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

import config as C

events = pd.read_csv(C.EVENTS_CLEAN, parse_dates=["listened_at"])
users = pd.read_csv(C.USER_INDICATORS)
tracks = pd.read_csv(C.TRACKS_CLEAN)
recs = pd.read_csv(C.RECOMMENDATIONS) if C.RECOMMENDATIONS.exists() else pd.DataFrame()
metrics = json.loads(C.MODEL_METRICS.read_text()) if C.MODEL_METRICS.exists() else {}

TEAL, AMBER, CORAL, SLATE = "#1ca58b", "#e9a74d", "#d66b66", "#7b8da0"
TEMPLATE = "plotly_white"
taste_cols = [c for c in users.columns if c.startswith("taste_") and c != "taste_spread"]
user_list = sorted(users.user_id)


def kpi(label, value, sub=""):
    return html.Div([html.Div(label, className="kpi-label"), html.Div(value, className="kpi-value"),
                     html.Div(sub, className="kpi-sub")], className="kpi")


# ---------------- act 1: readiness funnel ----------------
n = len(events)
linked = events.track_id.notna().sum()
cand = events.model_candidate.sum()
funnel = go.Figure(go.Funnel(y=["All listens", "Linked to Embeat", "High metadata agreement"],
                             x=[n, linked, cand], textinfo="value+percent initial",
                             marker_color=[SLATE, AMBER, TEAL]))
funnel.update_layout(template=TEMPLATE, margin=dict(l=10, r=10, t=10, b=10), height=260)

status_fig = px.bar(events.groupby(["user_id", "match_status"]).size().reset_index(name="listens"),
                    x="listens", y="user_id", color="match_status", orientation="h",
                    template=TEMPLATE, height=300,
                    color_discrete_sequence=[TEAL, AMBER, CORAL, SLATE])
status_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title=None, yaxis_title=None)

# ---------------- act 3: model comparison ----------------
if "models" in metrics:
    mdf = pd.DataFrame(metrics["models"]).T.reset_index(names="model")
    mlong = mdf.melt(id_vars="model", value_vars=["precision", "recall", "ndcg", "catalog_coverage"])
    model_fig = px.bar(mlong, x="variable", y="value", color="model", barmode="group",
                       template=TEMPLATE, height=300, color_discrete_sequence=[TEAL, SLATE])
    model_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None, yaxis_title=None,
                            legend_title=None)
    model_note = (f"Chronological hold-out · {metrics['users_evaluated']} of {metrics['users_total']} "
                  f"users have enough history to evaluate · K = {metrics['k']}")
else:
    model_fig, model_note = go.Figure(), metrics.get("note", "Run ml_models.py first.")

app = Dash(__name__, title="From plays to personalization")
app.layout = html.Div(className="shell", children=[
    html.H1("From plays to personalization"),
    html.P("ListenBrainz listening histories × Embeat audio features", className="lead"),
    html.Div(className="kpis", children=[
        kpi("Listening events", f"{n:,}", f"{events.user_id.nunique()} listeners"),
        kpi("Linked to Embeat", f"{linked / n:.1%}", f"{linked:,} events"),
        kpi("Model-ready", f"{cand / n:.1%}", f"{cand:,} high-agreement events"),
        kpi("Tracks with audio features", f"{len(tracks):,}", f"{tracks.mood_name.nunique()} mood clusters"),
    ]),

    html.H2("1 · How much of the data is usable?"),
    html.Div(className="row", children=[
        html.Div([dcc.Graph(figure=funnel, config={"displayModeBar": False})], className="card"),
        html.Div([dcc.Graph(figure=status_fig, config={"displayModeBar": False})], className="card"),
    ]),

    html.H2("2 · Q1 — What does each listener's taste look like?"),
    dcc.Dropdown(user_list, user_list[0], id="user", clearable=False, className="dd"),
    html.Div(className="row", children=[
        html.Div([dcc.Graph(id="radar", config={"displayModeBar": False})], className="card"),
        html.Div([dcc.Graph(id="mood", config={"displayModeBar": False})], className="card"),
    ]),
    html.Div(id="user-kpis", className="kpis"),

    html.H2("3 · Q2 — Can audio features recommend the next song?"),
    html.Div(className="row", children=[
        html.Div([dcc.Graph(figure=model_fig, config={"displayModeBar": False}),
                  html.P(model_note, className="note")], className="card"),
        html.Div([html.H3(id="rec-title"), dash_table.DataTable(
            id="recs", page_size=10, style_cell={"fontFamily": "inherit", "fontSize": 13, "textAlign": "left"},
            style_header={"fontWeight": "600"})], className="card"),
    ]),
])


@app.callback(Output("radar", "figure"), Output("mood", "figure"), Output("user-kpis", "children"),
              Output("recs", "data"), Output("rec-title", "children"), Input("user", "value"))
def update_user(user):
    u = users.set_index("user_id").loc[user]
    radar = go.Figure()
    if taste_cols:
        feats = [c.replace("taste_", "") for c in taste_cols]
        # tempo is on another scale: show it divided by 200 bpm
        scale = {"tempo": 200}
        vals = [u[c] / scale.get(f, 1) for c, f in zip(taste_cols, feats)]
        base = [tracks[f].mean() / scale.get(f, 1) for f in feats]
        radar.add_trace(go.Scatterpolar(r=base + base[:1], theta=feats + feats[:1], name="Catalog average",
                                        line_color=SLATE))
        radar.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=feats + feats[:1], name=user,
                                        fill="toself", line_color=TEAL))
    radar.update_layout(template=TEMPLATE, height=320, margin=dict(l=30, r=30, t=30, b=30),
                        polar=dict(radialaxis=dict(range=[0, 1])))

    ue = events[(events.user_id == user) & events.mood_name.notna()]
    mood = px.histogram(ue, y="mood_name", template=TEMPLATE, height=320,
                        color_discrete_sequence=[TEAL]).update_layout(
        margin=dict(l=10, r=10, t=30, b=10), yaxis_title=None, xaxis_title="listens",
        title="Mood mix of linked listens")

    cards = [kpi("Repeat share", f"{u.repeat_share:.0%}"),
             kpi("Artist diversity", f"{u.artist_diversity:.2f}", "unique artists / listens"),
             kpi("Linked share", f"{u.linked_share:.0%}"),
             kpi("Dominant mood", str(u.get("dominant_mood", "—")))]
    data = (recs[recs.user_id == user][["rank", "track_name", "artist_name", "mood_name", "similarity"]]
            .to_dict("records") if not recs.empty else [])
    return radar, mood, cards, data, f"Top-{C.K} content-based picks for {user}"


if __name__ == "__main__":
    app.run(debug=False)
