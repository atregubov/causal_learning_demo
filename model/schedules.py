import random
import pydot
import networkx as nx
from model.rules import Rule


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


def generate_script(n_agents, n_actions, site_idx, start_time, activity_dist, narrative_dist,
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
    for agent_index in range(n_agents):
        user_id = f'puppet_{site_idx}_{agent_index}'
        users[user_id] = {"reddit": {'last_event_time': 0, 'role': 'puppet'}}
        users[user_id]['id'] = user_id
        users[user_id]['operator_id'] = f'operator_{site_idx}'
        users[user_id]['log'] = []
        users[user_id]['training'] = []
        users[user_id]['site'] = site_idx
        users[user_id]['reddit']['script'] = sample_schedules_with_given_dist(start_time,
                                                                              random.randint(
                                                                                  n_actions[site_idx][0],
                                                                                  n_actions[site_idx][1]),
                                                                              site_idx,
                                                                              narrative_dist[site_idx],
                                                                              activity_dist[site_idx],
                                                                              n_lines_per_post)
    return users


def generate_site_schedule(site_idx, n_agents=100, start_time=0):
    # keep ranges of the thresholds for each site
    activity_dist = {0: [0, 0, 0, 0, 0, 1, 10, 5, 10, 9, 1, 1, 10, 10, 10, 10, 5, 4, 4, 4, 10, 6, 1, 1],
                     1: [0, 0, 0, 1, 1, 1, 1, 5, 1, 9, 1, 1, 10, 10, 1, 1, 5, 4, 4, 4, 1, 6, 1, 1],
                     2: [0, 0, 0, 0, 0, 1, 1, 5, 1, 9, 1, 1, 10, 10, 1, 1, 5, 4, 4, 4, 1, 6, 1, 1],
                     }  # site-0 # site-1 # site-2
    narrative_dist = {0: {'baseball': 60, 'travel': 20, 'mistreatment': 15, 'prejudice': 5},
                      1: {'baseball': 40, 'anti': 20, 'travel': 10, 'pro': 15, 'infrastructure': 15},
                      2: {'environmentalism': 20, 'covid': 20, 'debt': 20, 'pro': 25, 'baseball': 10}
                      }  # %
    number_of_actions_per_agent = {0: [2, 25],
                                   1: [3, 25],
                                   2: [3, 25]
                                   }
    number_of_lines_per_post = {0: [1, 3],
                                1: [1, 4],
                                2: [1, 2]
                                }
    # Initial schedule
    schedule = generate_script(n_agents, number_of_actions_per_agent, site_idx, start_time,
                               activity_dist=activity_dist,
                               narrative_dist=narrative_dist,
                               n_lines_per_post=number_of_lines_per_post)
    return schedule


def get_random_thresholds(narrative):
    return {'sleep_hours': random.randint(4, 9),
            'narrative_ratio': random.uniform(0.5, 0.8),
            f'narrative_ratio_{narrative}': random.uniform(0.5, 0.8),
            'narrative': narrative,
            'total_number_of_posts': random.randint(40, 50),
            'total_lines_of_posts': random.randint(40, 50),
            'narrative_lines_of_posts': random.randint(10, 20),
            f"narrative_number_of_posts_{narrative}": random.randint(10, 30),
           }


def generate_historical_data(schedule: dict, thresholds: dict, policy:Rule) -> dict:
    policy.pred(fit_data=thresholds, schedule=schedule, start_time=0, curr_time=0)
    # for user, u_schedule in schedule.items():
    #     u_schedule[policy.platform]["triggered_rules"] = dict()

    return schedule

