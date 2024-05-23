from dash import dcc
from dash import html
import math
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import dash_ag_grid as dag
from dash import Dash, Input, Output, html, dcc, callback, State

import dash_bootstrap_components as dbc


def model_div(app, data, username, hidden=True):
    """
    Results section/page div block layout.
    :param app: DASH/Flask webapp object
    :param data: dataframe with simulation outputs/inputs
    :param hidden: True/False if block is visible
    :return: div block
    """
    if username is None or username not in data:
        username = "s1"
    ####################################################################################################################
    # Prepare data table from data object
    ####################################################################################################################
    # all rules table
    rules_data_table = [{"Name": rule.name,
                         "Description": rule.rule_str,
                         "Local": rule.local,
                         "Shared": not(rule.local),
                         "graph": rule.get_DAG(),
                         "Add to editor": "Add to \npolicy editor"
                         }
                        for rule in data[username]["rules"]]
    rules_df = pd.DataFrame.from_records(rules_data_table)
    for i, r in rules_df.iterrows():
        fig = get_DAG_fig(data[username]["rules"][i].get_DAG())
        rules_df.at[i, "graph"] = fig

    # local policy view table
    local_rules_data_table = [{"Name": rule.name,
                         "Description": rule.rule_str,
                         "graph": rule.get_DAG(),
                         "Add to editor": "Add to \npolicy editor"
                         }
                        for rule in data[username]["rules"]]
    local_rules_df = pd.DataFrame.from_records(local_rules_data_table)
    for i, r in local_rules_df.iterrows():
        fig = get_DAG_fig(r["graph"])
        local_rules_df.at[i, "graph"] = fig

    # shared policy view table
    shared_rules_data_table = [{"Name": rule.name,
                                "Description": rule.rule_str,
                                "graph": rule.get_DAG(),
                                "Add to editor": "Add to \npolicy editor"
                                }
                               for rule in data["shared"][0].rules]
    shared_rules_df = pd.DataFrame.from_records(shared_rules_data_table)
    for i, r in shared_rules_df.iterrows():
        fig = get_DAG_fig(r["graph"])
        shared_rules_df.at[i, "graph"] = fig

    # policy editor panel
    editor_rules_data_table = [{"Name": rule.name,
                                "Description": rule.rule_str,
                                "Thresholds (fit data)": ["Local fit data"],
                                "graph": rule.get_DAG(),
                                "Remove": "Remove",
                                }
                               for rule in data[username]["editor"].rules]
    editor_rules_df = pd.DataFrame.from_records(editor_rules_data_table)
    for i, r in editor_rules_df.iterrows():
        fig = get_DAG_fig(r["graph"])
        editor_rules_df.at[i, "graph"] = fig

    ####################################################################################################################
    # Prepare table formating for panels
    ####################################################################################################################
    # all rules table
    rules_table_columns = [
        {
            "field": "Name",
            "resizable": True,
            "cellStyle": {"wordBreak": "normal"},
            "wrapText": True,
            "autoHeight": True,
        },
        {
            "field": "Description",
            "headerName": "Description",
            "resizable": True,
            "cellStyle": {"wordBreak": "normal"},
            "wrapText": True,
            "autoHeight": True,
            "minWidth": 200,
            "maxWidth": 600,
        },
        {
            "field": "Local",
            "headerName": "Local",
            "maxWidth": 100,
            "minWidth": 70,

        },
        {
            "field": "Shared",
            "headerName": "Shared",
            "maxWidth": 100,
            "minWidth": 70,
        },
        {
            "field": "graph",
            "cellRenderer": "DCC_GraphClickData",
            "headerName": "DAG",
            "maxWidth": 900,
            "minWidth": 500,
        },
        {
            "field": "Add to editor",
            "cellRenderer": "DBC_Button_Simple",
            "cellRendererParams": {"color": "success"},
            "minWidth": 200,
            "maxWidth": 250,

        },
    ]

    # shared and local policy view table uses short_set_columns
    short_set_columns = [
        {
            "field": "Name",
            "resizable": True,
            "cellStyle": {"wordBreak": "normal"},
            "wrapText": True,
            "autoHeight": True,
        },
        {
            "field": "Description",
            "headerName": "Description",
            "resizable": True,
            "cellStyle": {"wordBreak": "normal"},
            "wrapText": True,
            "autoHeight": True,
        },
        {
            "field": "graph",
            "cellRenderer": "DCC_GraphClickData",
            "headerName": "DAG",
            "maxWidth": 900,
            "minWidth": 500,
        }
    ]

    # editor panel table with rules
    editor_columns = [
        {
            "field": "Name",
            "resizable": True,
            "cellStyle": {"wordBreak": "normal"},
            "wrapText": True,
            "autoHeight": True,
        },
        {
            "field": "Description",
            "headerName": "Description",
            "resizable": True,
            "cellStyle": {"wordBreak": "normal"},
            "wrapText": True,
            "autoHeight": True,
        },
        {
             "field": "Thresholds (fit data)",
             "headerName": "Thresholds (fit data)",
             "resizable": True,
             "cellStyle": {"wordBreak": "normal"},
             "wrapText": True,
             "autoHeight": True,
         },
        {
            "field": "graph",
            "cellRenderer": "DCC_GraphClickData",
            "headerName": "DAG",
            "maxWidth": 900,
            "minWidth": 500,
        },
        {
            "field": "Remove",
            "cellRenderer": "DBC_Button_Simple",
            "cellRendererParams": {"color": "success"},
            "minWidth": 100,
            "maxWidth": 200,

        },
    ]

    ####################################################################################################################
    # Prepare div blocks for panels
    ####################################################################################################################
    rules_table_div = html.Div(
        [
            dag.AgGrid(
                id="rules_table_div_id",
                rowData=rules_df.to_dict("records"),
                columnSize="sizeToFit",
                columnDefs=rules_table_columns,
                defaultColDef={"filter": True, "minWidth": 60},
                dashGridOptions={"rowHeight": 200, "animateRows": False},
                style={"height": 800, 'display': 'inline-block', 'resize': 'both', 'overflow': 'auto'},

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
                style={"height": 400, 'display': 'inline-block', 'resize': 'both', 'overflow': 'auto'}
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
                style={"height": 400, 'display': 'inline-block', 'resize': 'both', 'overflow': 'auto'}
            ),
        ]
    )
    editor_table_div = html.Div(
        [
            dag.AgGrid(
                id="editor_table_div",
                rowData=editor_rules_df.to_dict("records"),
                columnSize="sizeToFit",
                columnDefs=editor_columns,
                defaultColDef={"filter": True, "minWidth": 60},
                dashGridOptions={"rowHeight": 150, "animateRows": False},
                style={"height": 800, 'display': 'inline-block', 'resize': 'both', 'overflow': 'auto'}
            ),
        ]
    )
    # div with all panels together
    panels_div = (html.Div(id="panels_div",
                           style={'display': 'none'} if hidden else {'display': 'block'},
                           children=[html.Div(id="local_div",
                                           children=[html.H2("Local causal DAG and rules"),
                                                     dcc.Dropdown([r.name for r in data[username]["local"]],
                                                                  data[username]["local"][0].name, id='local-dropdown'),
                                                     html.I('Policy name: '),
                                                     html.B(data[username]["local"][0].name, id='local_rule_name'),
                                                     html.Br(),
                                                     html.I('Rules: '),
                                                     local_table_div,
                                                     html.Br(),
                                                     html.I('DAG:'),
                                                     dcc.Checklist(
                                                         ['Show rule name on edges'],
                                                         ['Show rule name on edges', ],
                                                         id="show_rule_names_local",
                                                         inline=True
                                                     ),
                                                     dcc.Graph(id='local_dag',
                                                               figure=get_DAG_fig(data[username]["local"][0].get_DAG())),
                                            ],
                                           style={'width': '50%', 'padding': '10px 10px 20px 20px', 'display': 'inline-block'}
                                           ),
                                  html.Div(id="shared_div",
                                           children=[html.H2("Shared causal DAG and rules"),
                                                     dcc.Dropdown([r.name for r in data["shared"]],
                                                                  data["shared"][0].name, id='shared-dropdown'),
                                                     html.I('Policy name: '),
                                                     html.B(data["shared"][0].name, id='shared_rule_name'),
                                                     html.Br(),
                                                     html.I('Rules: '),
                                                     shared_table_div,
                                                     html.Br(),
                                                     html.I('DAG:'),
                                                     dcc.Checklist(
                                                         ['Show rule name on edges'],
                                                         ['Show rule name on edges', ],
                                                         id="show_rule_names_shared",
                                                         inline=True
                                                     ),
                                                     dcc.Graph(id='shared_dag',
                                                               figure=get_DAG_fig(data["shared"][0].get_DAG())),
                                                     ],
                                           style={'width': '50%', 'padding': '10px 10px 20px 20px',
                                                  'display': 'inline-block'}
                                           ),
                                  html.Div(id="all_rules_div",
                                          children=[html.H2("Rules"),
                                                    html.Br(),
                                                    rules_table_div,
                                                    ],
                                          style={'width': '100%', 'padding': '10px 10px 20px 20px',
                                                 'display': 'inline-block'}
                                          ),
                                  html.Div(id="editor_div",
                                           children=[html.H2("Policy editor"),
                                                     html.Div([
                                                         html.I('Policy name: '),
                                                         dcc.Input(id='editor_policy_name', type='string',
                                                                   value=data[username]["editor"].name, debounce=True),
                                                         html.Br(),
                                                         html.Button('Save', id='save_btn', n_clicks=0, style={'padding': '10px 10px 10px 10px'}),
                                                         html.Button('Fit', id='fit_btn', n_clicks=0, style={'padding': '10px 10px 10px 10px'}),
                                                         html.Button('Share', id='share_btn', n_clicks=0, style={'padding': '10px 10px 10px 10px'}),
                                                         html.Button('Evaluate', id='evaluate_btn', n_clicks=0, style={'padding': '10px 10px 10px 10px'}),
                                                         html.Div(id='editor_notification', children='Saved.')
                                                     ]),
                                                     html.Br(),
                                                     html.Div(id="editor_dag_div",
                                                              children=[html.I('DAG: '),
                                                                        dcc.Checklist(
                                                                            ['Show rule name on edges'],
                                                                            ['Show rule name on edges',],
                                                                            inline=True,
                                                                        id="show_rule_names_editor",
                                                                        ),
                                                                        dcc.Graph(id='editor_dag',
                                                                                  figure=get_DAG_fig(data[username][
                                                                                                         "editor"].get_DAG()),
                                                                                  style={"height": 800, 'width': '100%',
                                                                                         'display': 'inline-block',
                                                                                         'resize': 'both',
                                                                                         'overflow': 'auto'}),
                                                                        ],
                                                              style={'width': '40%', 'padding': '0px 0px 0px 0px',
                                                                     'display': 'inline-block'}
                                                              ),
                                                     html.Div(id="editor_rules_div",
                                                              children=[html.I('Rules: '),
                                                                        editor_table_div,
                                                                        ],
                                                              style={'width': '60%', 'padding': '0px 0px 0px 0px',
                                                                     'display': 'inline-block'}
                                                              ),
                                                     ],
                                           style={'width': '100%', 'padding': '10px 10px 20px 20px',
                                                  'display': 'inline-block'}
                                           ),
                                  ],
                       )
               )

    @app.callback(
        Output('editor_notification', 'children'),
        Input('save_btn', 'n_clicks'),
        State('editor_policy_name', 'value'),
        prevent_initial_call=True
    )
    def update_output(n_clicks, value):
        return f"Policy \"{value}\" saved."

    return panels_div


def hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
    '''If there is a cycle that is reachable from root, then result will not be a hierarchy.

       G: the graph
       root: the root node of current branch
       width: horizontal space allocated for this branch - avoids overlap with other branches
       vert_gap: gap between levels of hierarchy
       vert_loc: vertical location of root
       xcenter: horizontal location of root
    '''

    def reverse_edges(G):
        original_edges = list(G.edges(data=True))
        for a, b, edge_data in original_edges:
            G.remove_edge(a, b)
            G.add_edge(b, a, label=edge_data["label"], color=edge_data["color"])
    def h_recur(G, root, width=1., vert_gap = 0.2, vert_loc = 0.0, xcenter = 0.5,
                  pos = None, parent = None, parsed = [] ):
        if(root not in parsed):
            parsed.append(root)
            if pos == None:
                pos = {root:(xcenter,vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            neighbors = list(G.neighbors(root))
            if parent != None and parent in neighbors:
                neighbors.remove(parent)
            if len(neighbors) != 0:
                dx = width/len(neighbors)
                nextx = xcenter - width/2 - dx/2
                for neighbor in neighbors:
                    nextx += dx
                    pos = h_recur(G, neighbor, width = dx, vert_gap = vert_gap,
                                        vert_loc = vert_loc-vert_gap, xcenter=nextx, pos=pos,
                                        parent = root, parsed = parsed)
        return pos

    reverse_edges(G)
    pos = h_recur(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5)
    reverse_edges(G)
    return pos


def get_DAG_fig(G):
    lay = nx.layout.circular_layout(G)
    if len(list(G.nodes)) > 2 and "ban" in G.nodes:
        lay = hierarchy_pos(G, "ban")
    position = {node: lay[node] for node in G.nodes}

    edge_x = []
    edge_y = []
    edge_labels = list()
    edge_colors = list()
    for n0, n1, edge_data in G.edges(data=True):
        x0, y0 = position[n0]
        x1, y1 = position[n1]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_labels.append(edge_data["label"] if len(edge_data["label"]) < 25 else edge_data["label"][:20]+"..." if "label" in edge_data else "no label")
        edge_colors.append(edge_data["color"])

    node_x = []
    node_y = []
    node_colors = []
    for node, node_data in G.nodes(data=True):
        x, y = position[node]
        node_x.append(x)
        node_y.append(y)
        node_colors.append(node_data["color"])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        textposition="bottom center",
        marker=dict(
            size=60,
            line_width=2))

    node_adjacencies = []
    node_text = [str(n)[:20]+"..." if len(str(n)) >24 else str(n) for n in G.nodes]
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))

    node_trace.marker.color = node_colors
    node_trace.text = node_text

    fig = go.Figure(data=[node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=1, l=1, r=1, t=1),
                        font=dict(size=18),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    x_end = [x for idx, x in enumerate(edge_x) if idx % 2 == 1]
    y_end = [x for idx, x in enumerate(edge_y) if idx % 2 == 1]
    x_start = [x for idx, x in enumerate(edge_x) if idx % 2 == 0]
    y_start = [x for idx, x in enumerate(edge_y) if idx % 2 == 0]

    list_of_all_arrows = []
    for x0, y0, x1, y1, edge_color, edge_label in zip(x_end, y_end, x_start, y_start, edge_colors, edge_labels):
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
            arrowcolor=edge_color,
            )
        )
        list_of_all_arrows.append(arrow)
        angle = (math.atan((x1 - x0) / (y1 - y0)) / math.pi)*180
        arrow_label = go.layout.Annotation(dict(
            x=(x0 + x1) / 2 if angle > 0 else (x0 + x1) / 2 * 1.02,
            y=(y0 + y1) / 2 if angle > 0 else (y0 + y1) / 2 * 1.05,
            xref="x", yref="y",
            text = edge_label,
            textangle = int(angle - 90) if angle > 0 else  int(angle + 90),
            font = dict(color=edge_color)
            )
        )
        list_of_all_arrows.append(arrow_label)

    fig.update_layout(annotations=list_of_all_arrows)

    return fig

