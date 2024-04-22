from dash import dcc
from dash import html

from dash.dependencies import Input, Output
from data_processing.dag_processing import user_graph, get_nodes_and_edges, get_tree


def model_div(app, data, hidden=True):
    """
    Results section/page div block layout.
    :param app: DASH/Flask webapp object
    :param data: dataframe with simulation outputs/inputs
    :param hidden: True/False if block is visible
    :return: div block
    """
    nodes_and_edges_cache = get_nodes_and_edges(data)
    res_div = (html.Div(id="results_div",
                        children=[html.Div(id="local_div",
                                           children=[html.H2("Local causal DAG and rules:"),
                                                     html.I('Rule:'),
                                                     dcc.Input(id='local_rule', type='string',
                                                               value='narrative_ratio > 0.1', debounce=True),
                                                     dcc.Graph(id='local_dag',
                                                               figure=get_tree()), #data, 4, nodes_and_edges_cache)),
                                            ],
                                           style={'width': '40%', 'padding': '10px 10px 20px 20px',
                                                  'display': 'none'} if hidden else {'width': '40%',
                                                                                     'padding': '10px 10px 20px 20px', 'display': 'inline-block'}
                                           ),
                                  html.Div(id="shared_div",
                                           children=[html.H2("Shared causal DAG and rules:"),
                                                     html.I('Rule:'),
                                                     dcc.Input(id='shared_rule', type='string',
                                                               value='narrative_ratio > 0.1', debounce=True),
                                                     dcc.Graph(id='shared_dag',
                                                               figure=get_tree()), #user_graph(data, 4, nodes_and_edges_cache)),
                                                     ],
                                           style={'width': '40%', 'padding': '10px 10px 20px 20px',
                                                  'display': 'none'} if hidden else {'width': '40%',
                                                                                     'padding': '10px 10px 20px 20px', 'display': 'inline-block'}
                                           ),
                                ],
                       )
               )

    # @app.callback(
    #     Output('user_graph', 'figure'),
    #     [Input('day_index', 'value')],
    # )
    # def update_output(day_index):
    #     day_index = int(day_index)
    #     if day_index >= len(nodes_and_edges_cache):
    #         day_index = len(nodes_and_edges_cache) - 1
    #     return user_graph(data, day_index, nodes_and_edges_cache)

    return res_div
