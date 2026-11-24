"""Dash dashboard of the analysis checkpoint (23 Nov 2026).

Reads only the aggregated results of the ListenBrainz x Embeat analysis
(data/checkpoint/metrics_snapshot.json). Once the raw files are in data/raw/,
app.py shows the same story at event level plus the recommender results.

Run:  python checkpoint_dashboard.py   then open http://127.0.0.1:8051
"""
import json
from itertools import combinations
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

SNAP = json.loads((Path(__file__).parent / "data/checkpoint/metrics_snapshot.json").read_text(encoding="utf-8"))
p2, p3 = SNAP["phase2"], SNAP["phase3"]
users = pd.DataFrame(SNAP["users"])

# ---- indicators derived from the reported per-user counts ----
users["repeat_share"] = 1 - users.original_unique_tracks / users.original_listens
users["artist_diversity"] = users.original_unique_artists / users.original_listens
users["phase2_coverage"] = users.phase2_matched / users.original_listens
users["phase3_coverage"] = users.phase3_matched / users.original_listens
users["model_ready"] = users.unique_tracks >= 5          # minimum history for a fair hold-out

TEAL, AMBER, CORAL, BLUE, SLATE = "#1ca58b", "#e9a74d", "#d66b66", "#5b8bd0", "#9aa9b6"
T = "plotly_white"
M = dict(l=10, r=10, t=36, b=10)


def kpi(label, value, sub=""):
    return html.Div([html.Div(label, className="kpi-label"), html.Div(value, className="kpi-value"),
                     html.Div(sub, className="kpi-sub")], className="kpi")


def card(*children):
    return html.Div(list(children), className="card")


# 1. denominator funnel
funnel = go.Figure(go.Bar(
    y=["Selected Spotify IDs found in Embeat", "Listening events with a unique match", "Events passing quality checks"],
    x=[p3["spotify_match_pct"], p3["provisional_coverage_pct"], p3["model_candidate_coverage_pct"]],
    orientation="h", marker_color=[BLUE, AMBER, TEAL],
    text=[f'{p3["spotify_match_pct"]:.2f}%  (182/218 IDs)', f'{p3["provisional_coverage_pct"]:.2f}%  (220/1,100)',
          f'{p3["model_candidate_coverage_pct"]:.2f}%  (174/1,100)'], textposition="auto"))
funnel.update_layout(template=T, margin=M, height=260, xaxis=dict(range=[0, 100], ticksuffix="%"),
                     yaxis=dict(autorange="reversed"), title="The denominator changes everything")

# 2. event status (mutually exclusive)
status = go.Figure(go.Pie(labels=["Model candidates", "Review pending", "Ambiguous IDs", "Unmatched"],
                          values=[p3["model_candidates"], p3["events_to_review"], p3["ambiguous_events"],
                                  p3["unmatched_events"]],
                          hole=.55, marker_colors=[TEAL, AMBER, CORAL, "#d5dde3"], sort=False))
status.update_layout(template=T, margin=M, height=260, title="Status of all 1,100 listens")

# 4. metadata audit before/after Unicode fix
mb, ma = SNAP["metadata_before"], SNAP["metadata_after"]
meta = go.Figure([go.Bar(name="Before Unicode fix", x=list(mb), y=list(mb.values()), marker_color=SLATE,
                         text=list(mb.values()), textposition="outside"),
                  go.Bar(name="After Unicode fix", x=list(ma), y=list(ma.values()), marker_color=TEAL,
                         text=list(ma.values()), textposition="outside")])
meta.update_layout(template=T, margin=dict(l=10, r=10, t=36, b=40), height=300, barmode="group", legend=dict(orientation="h", y=-0.12),
                   title="Metadata check on 220 linked events")

# 5. listener overlap matrix
names = users.user_id.tolist()
shared = {frozenset((e["user_a"], e["user_b"])): e["shared_tracks"] for e in SNAP["shared_pairs"]}
z = [[None if a == b else shared.get(frozenset((a, b)), 0) for b in names] for a in names]
overlap = go.Figure(go.Heatmap(z=z, x=names, y=names, colorscale=[[0, "#eef2f4"], [1, TEAL]], showscale=False,
                               text=[["" if v is None else str(v) for v in r] for r in z], texttemplate="%{text}"))
overlap.update_layout(template=T, margin=M, height=300,
                      title=f'Shared eligible tracks: {p3["user_pairs_shared"]} of {p3["user_pairs_possible"]} pairs')

app = Dash(__name__, title="Checkpoint dashboard")
app.layout = html.Div(className="shell", children=[
    html.H1("From plays to personalization"),
    html.P(f'ListenBrainz × Embeat · analysis checkpoint {SNAP["snapshot_date"]}', className="lead"),
    html.Div(className="kpis", children=[
        kpi("Listening events", f'{p2["listen_events"]:,}', f'{p2["users"]} listeners · 2025–2026'),
        kpi("Spotify IDs found in Embeat", f'{p3["spotify_match_pct"]:.2f}%', "182 of 218 selected IDs"),
        kpi("Events linked", f'{p3["provisional_coverage_pct"]:.1f}%', "220 of 1,100"),
        kpi("Model candidates", f'{p3["model_candidate_coverage_pct"]:.2f}%', "174 high-agreement events"),
    ]),
    html.H2("1 · How much of the data is usable?"),
    html.Div(className="row", children=[card(dcc.Graph(figure=funnel, config={"displayModeBar": False})),
                                        card(dcc.Graph(figure=status, config={"displayModeBar": False}))]),
    html.H2("2 · Q1 — How do listeners differ?"),
    dcc.RadioItems(id="metric", inline=True, className="radio", value="phase3_coverage", options=[
        {"label": " Coverage after mapping ", "value": "phase3_coverage"},
        {"label": " Coverage before mapping ", "value": "phase2_coverage"},
        {"label": " Repeat share ", "value": "repeat_share"},
        {"label": " Artist diversity ", "value": "artist_diversity"}]),
    html.Div(className="row", children=[card(dcc.Graph(id="user-bars", config={"displayModeBar": False})),
                                        card(dcc.Graph(figure=meta, config={"displayModeBar": False}))]),
    html.H2("3 · Q2 — Is the data ready for collaborative filtering?"),
    html.Div(className="row", children=[
        card(dcc.Graph(figure=overlap, config={"displayModeBar": False})),
        card(html.Div(className="kpis two", children=[
            kpi("User–track pairs", f'{p3["unique_user_track_pairs"]}', f'from {p3["model_candidates"]} eligible plays'),
            kpi("Unique tracks", f'{p3["unique_tracks"]}', f'{p3["repeated_events"]} repeat plays'),
            kpi("Matrix sparsity", f'{p3["sparsity_pct"]:.2f}%', f'density {p3["density_pct"]:.2f}%'),
            kpi("Users with ≥5 tracks", f'{int(users.model_ready.sum())} / {len(users)}', "enough history to evaluate"),
        ])),
    ]),
])


@app.callback(Output("user-bars", "figure"), Input("metric", "value"))
def user_bars(metric):
    d = users.sort_values(metric)
    fig = px.bar(d, x=metric, y="user_id", orientation="h", template=T, height=300,
                 text=d[metric].map(lambda v: f"{v:.1%}" if metric != "artist_diversity" else f"{v:.2f}"),
                 color_discrete_sequence=[TEAL])
    fig.update_layout(margin=M, xaxis_title=None, yaxis_title=None, title="Per listener")
    if metric != "artist_diversity":
        fig.update_xaxes(tickformat=".0%")
    return fig


if __name__ == "__main__":
    app.run(debug=False, port=8051)
