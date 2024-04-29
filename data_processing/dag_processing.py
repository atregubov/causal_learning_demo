import plotly.graph_objects as go
import matplotlib.pyplot as plt
import networkx as nx


def get_tree(n_nodes = 3):
    edgelist = [("sleep_hours", "ban"), ("total_number_of_posts", "sleep_hours")]
    if n_nodes == 2:
        edgelist = [("total_number_of_posts", "ban")]
    G = nx.DiGraph(edgelist)  # use Graph constructor


    lay = nx.layout.circular_layout(G)

    position = {node: lay[node] for node in G.nodes}
    X = [lay[k][0] for k in list(G.nodes)]
    M = max(X)

    Yn = [position[k][0] for k in list(G.nodes)]
    Xn = [2*M-position[k][1] for k in list(G.nodes)]
    Xe = []
    Ye = []
    for edge in list(G.edges):
        Ye+=[position[edge[0]][0],position[edge[1]][0], None]
        Xe+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]

    labels = list(G.nodes)

    def make_annotations(Xn_, Yn_, text, font_size=10, font_color='rgb(250,250,250)'):
        annotations = []
        for idx, node in enumerate(G.nodes):
            annotations.append(
                dict(
                    text=text[idx],  # or replace labels with a different list for the text within the circle
                    y=Yn_[idx], x=Xn_[idx],
                    xref='x1', yref='y1',
                    font=dict(color=font_color, size=font_size),
                    showarrow=False)
            )
        return annotations

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Xe,
                             y=Ye,
                             mode='lines+markers',
                             marker=dict(size=10, symbol="arrow-bar-up"),
                             line=dict(color='rgb(10,10,10)', width=4),
                             hoverinfo='none'
                             ))
    fig.add_trace(go.Scatter(x=Xn,
                             y=Yn,
                             mode='markers',
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

    fig.update_layout(annotations=make_annotations(Xn, Yn, list(G.nodes)),
                      font_size=12,
                      showlegend=False,
                      xaxis=axis,
                      yaxis=axis,
                      margin=dict(l=1, r=1, b=1, t=1),
                      hovermode='closest',
                      plot_bgcolor='rgb(248,248,248)'
                      )
    return fig


