from dash import dcc
from dash import html
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import dash_ag_grid as dag
from dash import Dash, Input, Output, html, dcc, callback

import dash_bootstrap_components as dbc


def model_div(app, data, hidden=True):
    """
    Results section/page div block layout.
    :param app: DASH/Flask webapp object
    :param data: dataframe with simulation outputs/inputs
    :param hidden: True/False if block is visible
    :return: div block
    """

    rules_data_table = [{"Name": rule.name,
                         "Description": rule.rule_str,
                         "Local": rule.local,
                         "Shared": not(rule.local),
                         "graph": rule.get_DAG(),
                         "Add": "Add to \npolicy editor"
                         }
                        for rule in data["rules"]]

    rules_df = pd.DataFrame.from_records(rules_data_table)
    for i, r in rules_df.iterrows():
        fig = get_DAG_fig(data["rules"][i].get_DAG())
        rules_df.at[i, "graph"] = fig

    # local
    local_rules_data_table = [{"Name": rule.name,
                         "Description": rule.rule_str,
                         "graph": rule.get_DAG(),
                         "Add": "Add to \npolicy editor"
                         }
                        for rule in data["local"][0].rules]

    local_rules_df = pd.DataFrame.from_records(local_rules_data_table)
    for i, r in local_rules_df.iterrows():
        fig = get_DAG_fig(r["graph"])
        local_rules_df.at[i, "graph"] = fig

    # shared
    shared_rules_data_table = [{"Name": rule.name,
                         "Description": rule.rule_str,
                         "graph": rule.get_DAG(),
                         "Add": "Add to \npolicy editor"
                         }
                        for rule in data["shared"][0].rules]

    shared_rules_df = pd.DataFrame.from_records(shared_rules_data_table)
    for i, r in shared_rules_df.iterrows():
        fig = get_DAG_fig(r["graph"])
        shared_rules_df.at[i, "graph"] = fig


    rules_table_columns = [
        {
            "field": "Name",
        },
        {
            "field": "Description",
            "headerName": "Description",
        },
        {
            "field": "Local",
            "headerName": "Local",
        },
        {
            "field": "Shared",
            "headerName": "Shared",
        },
        {
            "field": "graph",
            "cellRenderer": "DCC_GraphClickData",
            "headerName": "DAG",
            "maxWidth": 900,
            "minWidth": 500,
        },
        {
            "field": "Add",
            "cellRenderer": "DBC_Button_Simple",
            "cellRendererParams": {"color": "success"},
        },
    ]

    short_set_columns = [
        {
            "field": "Name",
        },
        {
            "field": "Description",
            "headerName": "Description",
        },
        {
            "field": "graph",
            "cellRenderer": "DCC_GraphClickData",
            "headerName": "DAG",
            "maxWidth": 900,
            "minWidth": 500,
        }
    ]
    table_div = html.Div(
        [
            dcc.Markdown("Causal Rules"),
            dag.AgGrid(
                id="custom-component-graph-grid",
                rowData=rules_df.to_dict("records"),
                columnSize="sizeToFit",
                columnDefs=rules_table_columns,
                defaultColDef={"filter": True, "minWidth": 60},
                dashGridOptions={"rowHeight": 200, "animateRows": False},
                style={"height": 600, 'display': 'none'} if hidden else {"height": 600, 'display': 'inline-block'}
            ),
        ]
    )

    local_table_div = html.Div(
        [
            dag.AgGrid(
                id="local_table_div",
                rowData=local_rules_df.to_dict("records"),
                columnSize="sizeToFit",
                columnDefs=short_set_columns,
                defaultColDef={"filter": True, "minWidth": 60},
                dashGridOptions={"rowHeight": 150, "animateRows": False},
                style={"height": 400, 'display': 'none'} if hidden else {"height": 400, 'display': 'inline-block'}
            ),
        ]
    )

    shared_table_div = html.Div(
        [
            dag.AgGrid(
                id="shared_table_div",
                rowData=shared_rules_df.to_dict("records"),
                columnSize="sizeToFit",
                columnDefs=short_set_columns,
                defaultColDef={"filter": True, "minWidth": 60},
                dashGridOptions={"rowHeight": 150, "animateRows": False},
                style={"height": 400, 'display': 'none'} if hidden else {"height": 400, 'display': 'inline-block'}
            ),
        ]
    )


    res_div = (html.Div(id="results_div",
                        children=[html.Div(id="local_div",
                                           children=[html.H2("Local causal DAG and rules:"),
                                                     dcc.Dropdown([r.name for r in data["local"]], 'Local sleep policy', id='local-dropdown'),
                                                     html.Div(id='local-output-container'),
                                                     html.I('Policy name: '),
                                                     dcc.Input(id='local_rule', type='string',
                                                               value=data["local"][0].name, debounce=True),
                                                     html.Br(),
                                                     html.I('Rules: '),
                                                     local_table_div,
                                                     html.Br(),
                                                     html.I('DAG:'),
                                                     dcc.Graph(id='local_dag',
                                                               figure=get_DAG_fig(data["local"][0].get_DAG())),
                                            ],
                                           style={'width': '50%', 'padding': '10px 10px 20px 20px',
                                                  'display': 'none'} if hidden else {'width': '50%',
                                                                                     'padding': '10px 10px 20px 20px', 'display': 'inline-block'}
                                           ),
                                  html.Div(id="shared_div",
                                           children=[html.H2("Shared causal DAG and rules:"),
                                                     dcc.Dropdown([r.name for r in data["shared"]], 'Shared policy on number of posts',
                                                                  id='shared-dropdown'),
                                                     html.Div(id='dd-output-container_shared'),
                                                     html.I('Policy name: '),
                                                     dcc.Input(id='shared_rule', type='string',
                                                               value=data["shared"][0].name, debounce=True),
                                                     html.Br(),
                                                     html.I('Rules: '),
                                                     shared_table_div,
                                                     html.Br(),
                                                     html.I('DAG:'),
                                                     dcc.Graph(id='shared_dag',
                                                               figure=get_DAG_fig(data["shared"][0].get_DAG())),
                                                     ],
                                           style={'width': '50%', 'padding': '10px 10px 20px 20px',
                                                  'display': 'none'} if hidden else {'width': '50%',
                                                                                     'padding': '10px 10px 20px 20px', 'display': 'inline-block'}
                                           ),
                                  table_div,
                                  ],
                       )
               )

    return res_div


def get_DAG_fig(G):
    # edgelist = [("sleep_hours", "ban"), ("total_number_of_posts", "sleep_hours")]
    # if n_nodes == 2:
    #     edgelist = [("total_number_of_posts", "ban")]
    # G = nx.DiGraph(edgelist)  # use Graph constructor
    lay = nx.layout.circular_layout(G)
    position = {node: lay[node] for node in G.nodes}

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = position[edge[0]]
        x1, y1 = position[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0, color='white'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = position[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        textposition="bottom center",
        marker=dict(
            showscale=False,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Bluered',
            reversescale=True,
            color=[],
            size=60,
            line_width=2))

    node_adjacencies = []
    node_text = list(G.nodes)
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=1, l=1, r=1, t=1),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    x_end = [x for idx, x in enumerate(edge_x) if idx % 2 == 1]
    y_end = [x for idx, x in enumerate(edge_y) if idx % 2 == 1]
    x_start = [x for idx, x in enumerate(edge_x) if idx % 2 == 0]
    y_start = [x for idx, x in enumerate(edge_y) if idx % 2 == 0]

    list_of_all_arrows = []
    for x0, y0, x1, y1 in zip(x_end, y_end, x_start, y_start):
        arrow = go.layout.Annotation(dict(
            x=x0,
            y=y0,
            xref="x", yref="y",
            showarrow=True,
            axref="x", ayref='y',
            ax=x1,
            ay=y1,
            arrowhead=3,
            arrowwidth=2.5,
            arrowcolor='black', )
        )
        list_of_all_arrows.append(arrow)

    fig.update_layout(annotations=list_of_all_arrows)

    return fig

