import json
import random
import matplotlib.pyplot as plt
import sklearn
from sklearn import tree
import re
import pydot
import pandas as pd
import networkx as nx
from abc import ABC, abstractmethod


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


class Rule(ABC):
    """
    Rule class defines causal relationship between ban node (outcome) causing factors node (features).
    The schedule must have the following structure:
    schedule[user_id][self.platform]
                                    [ "script"             | "ban"      | "triggered_rules" | "features"]
                                        [[list_of_actions] | 1_0_-1_ban | rule_name         | feature_name ]
    """

    def __init__(self, name: str, rule_str: str, outcome_name: str, rules: list, depends_on: list = [],
                 platform="reddit"):
        self.name = name
        self.rule_str = rule_str
        self.rules = rules
        self.fit_data = None
        self.platform = platform
        self.outcome_name = outcome_name
        self.depends_on = depends_on
        self.local = True


    def rule_str(self) -> str:
        return self.rule_str

    def __str__(self):
        return self.rule_str

    def __repr__(self):
        return f"Rule: {self.rule_str}"

    def get_DAG(self) -> nx.MultiDiGraph:
        """
        Creates an NX graph with features nodes pointing at the ban node.
        """
        G = nx.MultiDiGraph()
        for rule in self.rules:
            rule_G = rule.get_DAG()
            G.add_nodes_from(list(rule_G.nodes))
            G.add_edges_from(list(rule_G.edges))

        return G

    @abstractmethod
    def pred(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        """
        Predict outcome value (e.g. ban), predicted outcome is added to the schedule ( in schedule[user_id][self.platform]["triggered_rules"][self.rule_name] ).
        """
        return None

    @abstractmethod
    def fit(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        """
        Updates self.fit_data (e.g. thresholds, weights, and other rule parameters) to fit observations.
        The schedule must have ban labels in schedule[user_id][self.platform]["ban"]
        """
        return None


class AggregateRule(Rule):
    """
    Rule class defines causal relationship between ban node (outcome) causing factors node (features).
    The schedule must have the following structure:
    schedule[user_id][self.platform]
                                    [ "script"             | "ban"      | "triggered_rules" | "features"]
                                        [[list_of_actions] | 1_0_-1_ban | rule_name         | feature_name ]
    """

    def __init__(self, name: str, rule_str: str, outcome_name: str, rules: list, platform="reddit"):
        super().__init__(name, rule_str, outcome_name, rules, [], platform)
        self.fit_data = None

    def get_DAG(self) -> nx.MultiDiGraph:
        """
        Creates an NX graph with features nodes pointing at the ban node.
        """
        G = nx.MultiDiGraph()
        for rule in self.rules:
            rule_G = rule.get_DAG()
            G.add_nodes_from(list(rule_G.nodes))
            G.add_edges_from(list(rule_G.edges))

        return G

    def pred(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        """
        Predict outcome value (e.g. ban), predicted outcome is added to the schedule
        ( in schedule[user_id][self.platform]["triggered_rules"][self.rule_name] ).
        """
        pred_vals = list()
        for rule in self.rules:
            pred_vals.append(rule.pred(schedule, start_time, curr_time))
        return max(pred_vals)

    def fit(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        """
        Updates self.fit_data (e.g. thresholds, weights, and other rule parameters) to fit observations.
        The schedule must have ban labels in schedule[user_id][self.platform]["ban"]
        """
        for rule in self.rules:
            rule.fit(schedule, start_time, curr_time)


class BasicRule(Rule):
    """
    Rule class defines causal relationship between ban node (outcome) causing factors node (features).
    The schedule must have the following structure:
    schedule[user_id][self.platform]
                                    [ "script"             | "ban"      | "triggered_rules" | "features"]
                                        [[list_of_actions] | 1_0_-1_ban | rule_name         | feature_name ]
    """

    def __init__(self, name: str, rule_str: str, feature_names: list, outcome_name: str = "ban", platform="reddit"):
        super().__init__(name, rule_str, outcome_name, [], [], platform)
        self.feature_names = feature_names
        self.feature_fns = {f_name: None for f_name in feature_names}
        self.fit_data = None

    def get_DAG(self) -> nx.MultiDiGraph:
        """
        Creates an NX graph with features nodes pointing at the ban node.
        """
        G = nx.MultiDiGraph()
        G.add_nodes_from(self.feature_names)
        G.add_nodes_from([self.outcome_name])
        G.add_edges_from([(f, self.outcome_name) for f in self.feature_names])

        return G

    @abstractmethod
    def pred(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        """
        Predict outcome value (e.g. ban), predicted outcome is added to the schedule ( in schedule[user_id][self.platform]["triggered_rules"][self.rule_name] ).
        """
        return None

    def feature_values(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        """
        Computes features needed for the rule, feature values are added in the schedule ( in schedule[user_id][self.platform]["feaures"][feature_name] ).
        """
        for user_id, u_schedule in schedule.items():
            for feature_name in self.feature_names:
                if "features" not in u_schedule[self.platform]:
                    u_schedule[self.platform]["features"] = {feature_name: None}
                u_schedule[self.platform]["features"][feature_name] = self.feature_fns[feature_name](
                    u_schedule[self.platform]["script"], feature_name, start_time, curr_time)

    @abstractmethod
    def fit(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        """
        Updates self.fit_data (e.g. thresholds, weights, and other rule parameters) to fit observations.
        The schedule must have ban labels in schedule[user_id][self.platform]["ban"]
        """
        return None


class SleepHoursRule(BasicRule):
    def __init__(self):
        super().__init__("sleep_hours_rule",
                         "if sleep_hours in to_ban_range -> ban and if sleep_hours in not_to_ban_range -> no ban",
                         ["sleep_hours"], "ban")
        self.feature_fns['sleep_hours'] = self._get_sleep_hours

    def _get_sleep_hours(self, actions_script: list, feature_name: str, start_time: int = 0, curr_time: int = None):
        hours_of_sleep_dist = [0 for _ in range(24)]
        for action in actions_script:
            hours_of_sleep_dist[int(((action['nodeTime'] - start_time) % (24 * 3600)) / 3600)] += 1

        return hours_of_sleep_dist

    def pred(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        # update feature values
        self.feature_values(schedule, start_time, curr_time)

        # predict/evaluate ban
        for user, u_schedule in schedule.items():
            if "triggered_rules" not in u_schedule[self.platform]:
                u_schedule[self.platform]["triggered_rules"] = dict()
            u_schedule[self.platform]["triggered_rules"][self.name] = 0
            hours_of_sleep_dist = u_schedule[self.platform]["features"]["sleep_hours"]
            banned_by_min_sleep_hours = False
            ban_status_unknown = False
            for index, h in enumerate(hours_of_sleep_dist):
                if h != 0:
                    if isinstance(self.fit_data['sleep_hours'], dict):
                        if self.fit_data['sleep_hours']['banned']['min'] <= index <= \
                                self.fit_data['sleep_hours']['banned'][
                                    'max']:  # this assumes that first k-elements in hours_of_sleep_dist are sleep hours.
                            banned_by_min_sleep_hours = True
                        elif self.fit_data['sleep_hours']['notbanned']['min'] <= index <= \
                                self.fit_data['sleep_hours']['notbanned']['max']:
                            ban_status_unknown = True
                        break
                    else:
                        if index < self.fit_data[
                            'sleep_hours']:  # this assumes that first k-elements in hours_of_sleep_dist are sleep hours.
                            banned_by_min_sleep_hours = True
                        break

            if isinstance(self.fit_data['sleep_hours'], dict):
                u_schedule[self.platform]["triggered_rules"][
                    self.name] = 1 if banned_by_min_sleep_hours else -1 if ban_status_unknown else 0
            else:
                u_schedule[self.platform]["triggered_rules"][self.name] = 1 if banned_by_min_sleep_hours else 0

            if u_schedule[self.platform]["triggered_rules"][self.name] == 1:
                u_schedule[self.platform]["ban"] = 1
            else:
                u_schedule[self.platform]["ban"] = 0

    def fit(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        data = [{"ban": u_data[self.platform]["ban"],
                 "sleep_bgn": min([hour_idx for hour_idx, val
                                   in enumerate(u_data[self.platform]["features"]["sleep_hours"]) if val > 0]),
                 "sleep_end": max([hour_idx for hour_idx, val
                                   in enumerate(u_data[self.platform]["features"]["sleep_hours"]) if val > 0])
                 }
                for u_id, u_data in schedule.items()]
        bans_df = pd.DataFrame.from_records(data)

        for feature_name in self.feature_names:
            if feature_name not in self.fit_data:
                self.fit_data[feature_name] = {"sleep_hours": {"banned": {"min": None, "max": None},
                                                               "notbanned": {"min": None, "max": None}}}
            f_data = self.fit_data[feature_name]
            f_data['banned']['min'] = bans_df[bans_df['ban'] == 1]["sleep_bgn"].min()
            f_data['banned']['max'] = bans_df[bans_df['ban'] == 1]["sleep_end"].max()
            f_data['notbanned']['min'] = bans_df[bans_df['ban'] == 0]["sleep_bgn"].min()
            f_data['notbanned']['max'] = bans_df[bans_df['ban'] == 0]["sleep_end"].max()


class TotalNumberOfPostsRule(BasicRule):
    def __init__(self):
        super().__init__("total_number_of_posts",
                         "if total_number_of_posts in to_ban_range -> ban and if total_number_of_posts in not_to_ban_range -> no ban",
                         ["total_number_of_posts"], "ban")
        self.feature_fns['total_number_of_posts'] = self._get_total_number_of_posts

    def _get_total_number_of_posts(self, actions_script: list, feature_name: str, start_time: int = 0,
                                   curr_time: int = None):
        return len(actions_script)

    def pred(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        # update feature values
        self.feature_values(schedule, start_time, curr_time)

        # predict/evaluate ban
        for user, u_schedule in schedule.items():
            if "triggered_rules" not in u_schedule[self.platform]:
                u_schedule[self.platform]["triggered_rules"] = dict()
            u_schedule[self.platform]["triggered_rules"][self.name] = 0
            n_posts = u_schedule[self.platform]["features"]["total_number_of_posts"]

            if isinstance(self.fit_data['total_number_of_posts'], dict):
                banned = False
                not_banned = False
                ban_unknown = False

                if n_posts > self.fit_data["total_number_of_posts"]['banned']['min'] and n_posts < \
                        self.fit_data["total_number_of_posts"]['banned']['max']:
                    banned = True
                elif n_posts > self.fit_data["total_number_of_posts"]['notbanned']['min'] and n_posts < \
                        self.fit_data["total_number_of_posts"]['notbanned']['max']:
                    not_banned = True
                else:
                    ban_unknown = True

                u_schedule[self.platform]["triggered_rules"][self.name] = 1 if banned else 0 if not_banned else -1

            else:
                u_schedule[self.platform]["triggered_rules"][self.name] = 1 if n_posts > self.fit_data[
                    "total_number_of_posts"] else 0

            if u_schedule[self.platform]["triggered_rules"][self.name] == 1:
                u_schedule[self.platform]["ban"] = 1
            else:
                u_schedule[self.platform]["ban"] = 0

    def fit(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        data = [{"ban": u_data[self.platform]["ban"],
                 "n_posts": u_data[self.platform]["features"]["total_number_of_posts"]
                 } for u_id, u_data in schedule.items()]
        bans_df = pd.DataFrame.from_records(data)

        for feature_name in self.feature_names:
            if feature_name not in self.fit_data:
                self.fit_data[feature_name] = {"total_number_of_posts": {"banned": {"min": None, "max": None},
                                                                         "notbanned": {"min": None, "max": None}}}
            f_data = self.fit_data[feature_name]
            f_data['banned']['min'] = bans_df[bans_df['ban'] == 1]["n_posts"].min()
            f_data['banned']['max'] = bans_df[bans_df['ban'] == 1]["n_posts"].max()
            f_data['notbanned']['min'] = bans_df[bans_df['ban'] == 0]["n_posts"].min()
            f_data['notbanned']['max'] = bans_df[bans_df['ban'] == 0]["n_posts"].max()


if __name__ == '__main__':
    # Ground Truth policy and thresholds
    gt_DAG = """
    dag {
    bb="-2.474,-3.873,2.319,2.622"
    ban [pos="0.717,-2.587"]
    narrative_ratio [pos="-0.565,-2.824"]
    sleep_hours [pos="0.086,-1.307"]
    total_lines_of_posts [pos="-0.990,-1.251"]
    total_number_of_posts [pos="0.113,0.560"]
    narrative_ratio -> ban
    sleep_hours -> ban
    total_lines_of_posts -> ban
    total_lines_of_posts -> sleep_hours
    total_number_of_posts -> ban
    total_number_of_posts -> sleep_hours
    }
    """
    gt_DAG = to_nx_DAG(gt_DAG)

    gt_feature_thresholds = {'sleep_hours': 6,
                             'narrative_ratio': 0.7,
                             'narrative': "un",
                             'total_number_of_posts': 23,
                             'total_lines_of_posts': 45
                             }

    # keep ranges of the thresholds for each site
    activity_dist = {0: [0, 0, 0, 0, 0, 1, 10, 5, 10, 9, 1, 1, 10, 10, 10, 10, 5, 4, 4, 4, 10, 6, 1, 1],
                     1: [0, 0, 0, 1, 1, 1, 1, 5, 1, 9, 1, 1, 10, 10, 1, 1, 5, 4, 4, 4, 1, 6, 1, 1],
                     2: [0, 0, 0, 0, 0, 1, 1, 5, 1, 9, 1, 1, 10, 10, 1, 1, 5, 4, 4, 4, 1, 6, 1, 1],
                     }  # site-0
    # site-1
    # site-2
    narrative_dist = {0: {'un': 60, 'travel': 20, 'mistreatment': 15, 'prejudice': 5},
                      1: {'un': 40, 'anti': 20, 'travel': 10, 'pro': 15, 'infrastructure': 15},
                      2: {'environmentalism': 20, 'covid': 20, 'debt': 20, 'pro': 25, 'un': 10}
                      }  # %
    number_of_actions_per_agent = {0: [2, 25],
                                   1: [3, 25],
                                   2: [3, 25]
                                   }
    number_of_lines_per_post = {0: [1, 3],
                                1: [1, 4],
                                2: [1, 2]
                                }
    # Initial schedule before rescheduling
    schedule_before_rescheduling = generate_script(n_agents=100, n_sites=3, start_time=0,
                                                   n_actions=number_of_actions_per_agent,
                                                   activity_dist=activity_dist,
                                                   narrative_dist=narrative_dist,
                                                   n_lines_per_post=number_of_lines_per_post)

    # schedule_before_rescheduling["puppet_0_0"]["reddit"]

    sleep_h_rule = SleepHoursRule()
    sleep_h_rule.fit_data = {"sleep_hours": {"banned": {"min": 0, "max": 6}, "notbanned": {"min": 8, "max": 22}}}
    sleep_h_rule.pred(schedule_before_rescheduling, 0)
    sleep_h_rule.fit(schedule_before_rescheduling, 0)

    sleep_h_rule2 = SleepHoursRule()
    sleep_h_rule2.fit_data = {"sleep_hours": {"banned": {"min": 0, "max": 6}, "notbanned": {"min": 8, "max": 22}}}
    sleep_h_rule2.pred(schedule_before_rescheduling, 0)
    sleep_h_rule2.fit(schedule_before_rescheduling, 0)

    complex_rule = AggregateRule("Complex Rule", "Two sleep rules", "ban",
                                 [sleep_h_rule, sleep_h_rule2])

    print(complex_rule.get_DAG())
    #print(schedule_before_rescheduling)
