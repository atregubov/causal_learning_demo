import random
import pydot
import networkx as nx


def to_nx_DAG(d_model, verbose=False):
    """
    Convert DAGitty model (as string from https://dagitty.net) to nx DiGraph
    """
    dot_G = pydot.graph_from_dot_data("graph " + d_model)[0]
    nx_multi_G = nx.nx_pydot.from_pydot(dot_G)
    G = nx.MultiDiGraph()  # MultiDiGraph
    G.add_nodes_from(nx_multi_G.nodes)
    G.add_edges_from(list(map(lambda x: (x[1], x[0]), nx_multi_G.edges)))
    return G


def sample_schedules_with_given_dist(start_time, n_actions, site_index, narrative_dist, activity_dist,
                                     n_lines_per_post):
    """
    Sample sequences of agent actions from given distributions.
    :param start_time: epoc time in seconds, can be 0 if absolute time vlues do not matter (relative time)
    :param n_actions: number of actions taken by each agent
    :param site_index: site index label, used to retrieve the associated distribution
    :param narrative_dist: narrative distribution in the following dictionary format {'site_index': {"narr_1": int_freq, ..,}}
    :param activity_dist: agent daily activity distribution in the following dictionary format {'site_index': {"hour_1": int_freq, ..,}}
    :param n_lines_per_post: number of lines per post distribution in the following dictionary format {'site_index': [min_lines, max_lines]}
    :return: list of actions (schedule) for each agent.
    """
    actions = list()
    last_event_day = 0

    for action_index in range(n_actions):
        action_type = "post"
        days_between_events = random.randint(0, 3)
        hour_of_day = random.sample([h for h in range(len(activity_dist))], counts=activity_dist, k=1)[0]
        event_time = start_time + last_event_day + days_between_events * 24 * 3600 + hour_of_day * 3600
        narrative = random.sample([k for k in narrative_dist.keys()], counts=[v for v in narrative_dist.values()],
                                  k=1)[0]
        # make a post
        post_event = dict()
        post_event['actionType'] = action_type
        post_event['nodeTime'] = int(event_time)
        post_event["informationID"] = narrative
        post_event["reassigned"] = 0
        post_event["n_lines"] = random.randint(n_lines_per_post[site_index][0], n_lines_per_post[site_index][1])
        actions.append(post_event)
        last_event_day = start_time + last_event_day + days_between_events * 24 * 3600

    actions.sort(key=lambda e: e['nodeTime'])
    return actions


def generate_script(n_agents, n_actions, n_sites, start_time, activity_dist, narrative_dist,
                    n_lines_per_post={0: [1, 3],
                                      1: [1, 3],
                                      2: [1, 3]}):
    """
    Generate agent actions from given distributions for all sites.
    :param n_agents number of agents
    :param n_actions: number of actions taken by each agent
    :param n_sites number of sites
    :param start_time: epoc time in seconds, can be 0 if absolute time vlues do not matter (relative time)
    :param narrative_dist: narrative distribution in the following dictionary format {'site_index': {"narr_1": int_freq, ..,}}
    :param activity_dist: agent daily activity distribution in the following dictionary format {'site_index': {"hour_1": int_freq, ..,}}
    :param n_lines_per_post: number of lines per post distribution in the following dictionary format {'site_index': [min_lines, max_lines]}
    :return: list of actions (schedule) for each agent.
    """
    users = dict()
    for site_index in range(n_sites):
        for agent_index in range(n_agents):
            user_id = f'puppet_{site_index}_{agent_index}'
            users[user_id] = {"reddit": {'last_event_time': 0, 'role': 'puppet'}}
            users[user_id]['id'] = user_id
            users[user_id]['operator_id'] = f'operator_{site_index}'
            users[user_id]['log'] = []
            users[user_id]['training'] = []
            users[user_id]['site'] = site_index
            users[user_id]['reddit']['script'] = sample_schedules_with_given_dist(start_time,
                                                                                  random.randint(
                                                                                      n_actions[site_index][0],
                                                                                      n_actions[site_index][1]),
                                                                                  site_index,
                                                                                  narrative_dist[site_index],
                                                                                  activity_dist[site_index],
                                                                                  n_lines_per_post)
    return users
