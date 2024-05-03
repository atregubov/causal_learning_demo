import plotly.graph_objects as go
import matplotlib.pyplot as plt
import networkx as nx


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


