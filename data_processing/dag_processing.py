import plotly.graph_objects as go
import networkx as nx

import igraph
from igraph import Graph, EdgeSeq


def user_graph(data, day_index, nodes_and_edges_cache=None):
    """This is a demo layout. Actual data is not currently loaded."""
    nodes_and_edges = get_nodes_and_edges(data) if nodes_and_edges_cache is None else nodes_and_edges_cache
    G = nx.Graph()
    G.add_edges_from(nodes_and_edges[int(day_index)][1])
    G.add_nodes_from(nodes_and_edges[int(day_index)][0])
    pos = nx.spring_layout(G)

    # edges trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(color='black', width=1),
        hoverinfo='none',
        showlegend=False,
        mode='lines')

    # nodes trace
    node_x = []
    node_y = []
    text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y, text=text,
        mode='markers+text',
        showlegend=False,
        hoverinfo='none',
        marker=dict(
            color='pink',
            size=5,
            line=dict(color='black', width=1)))

    # layout
    layout = dict(plot_bgcolor='white',
                  paper_bgcolor='white',
                  margin=dict(t=10, b=10, l=10, r=10, pad=0),
                  xaxis=dict(linecolor='black',
                             showgrid=False,
                             showticklabels=False,
                             mirror=True),
                  yaxis=dict(linecolor='black',
                             showgrid=False,
                             showticklabels=False,
                             mirror=True))

    # figure
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)

    return fig


def get_nodes_and_edges(data, verbose=False):
    """
    For each day returns a set of users that interacted as nodes and a all interactions as edges.
    :param data: data in DASH simulation format
    :param verbose:
    :return: dictionary: each day is a keys and value is (nodes, edges) tuple.
    """
    data = data.assign(day=lambda x: x['nodeTime'] // (24 * 3600))

    day_index_min = data['day'].min()
    day_index_max = data['day'].max()

    nodes_and_edges = dict()

    if verbose:
        print(f"days: {len(sorted(list(data['day'].unique())))}, min: {day_index_min}, max: {day_index_max}")
    for day_index in range((day_index_max - day_index_min) + 1):
        day_data = data[data['day'] == day_index_min + day_index]
        if verbose:
            print(f"day {day_index}: {len(day_data)}")
        edges = set()
        nodes = set()
        for index, event in day_data.iterrows():
            node_user = event['rootUserID']
            node_parent = event['parentUserID']
            nodes.add(node_user)
            nodes.add(node_parent)
            if node_user != node_parent:
                edges.add((node_user, node_parent))
        nodes_and_edges[day_index] = (nodes, edges)

    return nodes_and_edges


def get_tree():
    nr_vertices = 6
    v_label = list(map(str, range(nr_vertices)))
    G = Graph.Tree(nr_vertices, 2) # 2 stands for children number
    lay = G.layout('rt')

    position = {k: lay[k] for k in range(nr_vertices)}
    Y = [lay[k][1] for k in range(nr_vertices)]
    M = max(Y)

    es = EdgeSeq(G) # sequence of edges
    E = [e.tuple for e in G.es] # list of edges

    L = len(position)
    Xn = [position[k][0] for k in range(L)]
    Yn = [2*M-position[k][1] for k in range(L)]
    Xe = []
    Ye = []
    for edge in E:
        Xe+=[position[edge[0]][0],position[edge[1]][0], None]
        Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]

    labels = v_label
    labels[0] = "ban"
    labels[1] = "narrative_ratio"
    labels[2] = "sleep_hours"
    labels[3] = "total_number_of_posts"

    def make_annotations(pos, text, font_size=10, font_color='rgb(250,250,250)'):
        L = len(pos)
        if len(text) != L:
            raise ValueError('The lists pos and text must have the same len')
        annotations = []
        for k in range(L):
            annotations.append(
                dict(
                    text=labels[k],  # or replace labels with a different list for the text within the circle
                    x=pos[k][0], y=2 * M - position[k][1],
                    xref='x1', yref='y1',
                    font=dict(color=font_color, size=font_size),
                    showarrow=False)
            )
        return annotations

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Xe,
                       y=Ye,
                       mode='lines',
                       line=dict(color='rgb(210,210,210)', width=1),
                       hoverinfo='none'
                       ))
    fig.add_trace(go.Scatter(x=Xn,
                      y=Yn,
                      mode='markers',
                      name='bla',
                      marker=dict(symbol='circle-dot',
                                    size=70,
                                    color='#6175c1',    #'#DB4551',
                                    line=dict(color='rgb(50,50,50)', width=1)
                                    ),
                      text=labels,
                      hoverinfo='text',
                      opacity=0.8
                      ))

    axis = dict(showline=False,  # hide axis line, grid, ticklabels and  title
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                )

    fig.update_layout(title='Tree with Reingold-Tilford Layout',
                      annotations=make_annotations(position, v_label),
                      font_size=12,
                      showlegend=False,
                      xaxis=axis,
                      yaxis=axis,
                      margin=dict(l=40, r=40, b=85, t=100),
                      hovermode='closest',
                      plot_bgcolor='rgb(248,248,248)'
                      )
    return fig


