from dowhy import CausalModel
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures
from time import time
import warnings
warnings.filterwarnings("ignore")


class GenerateCausalMap:
    """This class performs the causal inference and generates the casual map."""

    def __init__(self, environment):
        """This method initializes the environment variables."""

        self.environment = environment
        self.samples_to_collect = 10000

        self.history = {'Action Right': [], 'Action Left': [], 'Action Up': [], 'Action Down': [],
                        'State At Breeze': [], 'State At Stench': [], 'State At Pit': [], 'State At Wumpus': [],
                        'State At Gold': [],
                        'State Left Breeze': [], 'State Left Stench': [], 'State Left Pit': [], 'State Left Wumpus': [],
                        'State Left Gold': [],
                        'State Top Left Breeze': [], 'State Top Left Stench': [], 'State Top Left Pit': [],
                        'State Top Left Wumpus': [], 'State Top Left Gold': [],
                        'State Up Breeze': [], 'State Up Stench': [], 'State Up Pit': [], 'State Up Wumpus': [],
                        'State Up Gold': [],
                        'State Top Right Breeze': [], 'State Top Right Stench': [], 'State Top Right Pit': [],
                        'State Top Right Wumpus': [], 'State Top Right Gold': [],
                        'State Right Breeze': [], 'State Right Stench': [], 'State Right Pit': [],
                        'State Right Wumpus': [], 'State Right Gold': [],
                        'State Bottom Right Breeze': [], 'State Bottom Right Stench': [], 'State Bottom Right Pit': [],
                        'State Bottom Right Wumpus': [], 'State Bottom Right Gold': [],
                        'State Down Breeze': [], 'State Down Stench': [], 'State Down Pit': [], 'State Down Wumpus': [],
                        'State Down Gold': [],
                        'State Bottom Left Breeze': [], 'State Bottom Left Stench': [], 'State Bottom Left Pit': [],
                        'State Bottom Left Wumpus': [], 'State Bottom Left Gold': [],
                        'Next State At Breeze': [], 'Next State At Stench': [], 'Next State At Pit': [],
                        'Next State At Wumpus': [], 'Next State At Gold': [],
                        'Next State Left Breeze': [], 'Next State Left Stench': [], 'Next State Left Pit': [],
                        'Next State Left Wumpus': [], 'Next State Left Gold': [],
                        'Next State Top Left Breeze': [], 'Next State Top Left Stench': [],
                        'Next State Top Left Pit': [], 'Next State Top Left Wumpus': [], 'Next State Top Left Gold': [],
                        'Next State Up Breeze': [], 'Next State Up Stench': [], 'Next State Up Pit': [],
                        'Next State Up Wumpus': [], 'Next State Up Gold': [],
                        'Next State Top Right Breeze': [], 'Next State Top Right Stench': [],
                        'Next State Top Right Pit': [], 'Next State Top Right Wumpus': [],
                        'Next State Top Right Gold': [],
                        'Next State Right Breeze': [], 'Next State Right Stench': [], 'Next State Right Pit': [],
                        'Next State Right Wumpus': [], 'Next State Right Gold': [],
                        'Next State Bottom Right Breeze': [], 'Next State Bottom Right Stench': [],
                        'Next State Bottom Right Pit': [], 'Next State Bottom Right Wumpus': [],
                        'Next State Bottom Right Gold': [],
                        'Next State Down Breeze': [], 'Next State Down Stench': [], 'Next State Down Pit': [],
                        'Next State Down Wumpus': [], 'Next State Down Gold': [],
                        'Next State Bottom Left Breeze': [], 'Next State Bottom Left Stench': [],
                        'Next State Bottom Left Pit': [], 'Next State Bottom Left Wumpus': [],
                        'Next State Bottom Left Gold': [],
                        'Reward': []}

        self.list_of_positions = ['At', 'Left', 'Top Left', 'Up', 'Top Right', 'Right', 'Bottom Right', 'Down',
                                  'Bottom Left']

    def generate_random_data(self):
        """This method generates data through random exploration."""

        while len(self.history['Action Right']) < self.samples_to_collect:
            state = self.environment.reset(random_start=True)
            done = False

            while not done:

                action = self.environment.action_space.sample()

                next_state, reward, done, info = self.environment.step(action)

                if action == 0:
                    self.history['Action Right'].append(True)
                else:
                    self.history['Action Right'].append(False)

                if action == 1:
                    self.history['Action Left'].append(True)
                else:
                    self.history['Action Left'].append(False)

                if action == 2:
                    self.history['Action Up'].append(True)
                else:
                    self.history['Action Up'].append(False)

                if action == 3:
                    self.history['Action Down'].append(True)
                else:
                    self.history['Action Down'].append(False)

                self.history['Reward'].append(reward)

                index = 0
                for i in range(9):

                    if state[index] == 1:
                        self.history['State ' + self.list_of_positions[i] + ' Breeze'].append(True)
                    elif state[index] == 0:
                        self.history['State ' + self.list_of_positions[i] + ' Breeze'].append(False)

                    if state[index + 1] == 1:
                        self.history['State ' + self.list_of_positions[i] + ' Stench'].append(True)
                    elif state[index + 1] == 0:
                        self.history['State ' + self.list_of_positions[i] + ' Stench'].append(False)

                    if state[index + 2] == 1:
                        self.history['State ' + self.list_of_positions[i] + ' Pit'].append(True)
                    elif state[index + 2] == 0:
                        self.history['State ' + self.list_of_positions[i] + ' Pit'].append(False)

                    if state[index + 3] == 1:
                        self.history['State ' + self.list_of_positions[i] + ' Wumpus'].append(True)
                    elif state[index + 3] == 0:
                        self.history['State ' + self.list_of_positions[i] + ' Wumpus'].append(False)

                    if state[index + 4] == 1:
                        self.history['State ' + self.list_of_positions[i] + ' Gold'].append(True)
                    elif state[index + 4] == 0:
                        self.history['State ' + self.list_of_positions[i] + ' Gold'].append(False)

                    if next_state[index] == 1:
                        self.history['Next State ' + self.list_of_positions[i] + ' Breeze'].append(True)
                    elif next_state[index] == 0:
                        self.history['Next State ' + self.list_of_positions[i] + ' Breeze'].append(False)

                    if next_state[index + 1] == 1:
                        self.history['Next State ' + self.list_of_positions[i] + ' Stench'].append(True)
                    elif next_state[index + 1] == 0:
                        self.history['Next State ' + self.list_of_positions[i] + ' Stench'].append(False)

                    if next_state[index + 2] == 1:
                        self.history['Next State ' + self.list_of_positions[i] + ' Pit'].append(True)
                    elif next_state[index + 2] == 0:
                        self.history['Next State ' + self.list_of_positions[i] + ' Pit'].append(False)

                    if next_state[index + 3] == 1:
                        self.history['Next State ' + self.list_of_positions[i] + ' Wumpus'].append(True)
                    elif next_state[index + 3] == 0:
                        self.history['Next State ' + self.list_of_positions[i] + ' Wumpus'].append(False)

                    if next_state[index + 4] == 1:
                        self.history['Next State ' + self.list_of_positions[i] + ' Gold'].append(True)
                    elif next_state[index + 4] == 0:
                        self.history['Next State ' + self.list_of_positions[i] + ' Gold'].append(False)
                    index += 5

                state = next_state

                if len(self.history['Action Right']) == self.samples_to_collect:
                    break

    def reward_to_object(self):
        """This method computes the causal estimate from the reward to the objects."""

        data = pd.DataFrame.from_dict(self.history)

        graph_pit = 'graph[directed 1node[ id "Reward" label "Reward"]node[ id "Next State At Pit" label "Next State ' \
                    'At Pit"]edge[source "Reward" target "Next State At Pit"]] '
        model_pit = CausalModel(data=data, treatment=['Reward'], outcome=['Next State At Pit'], graph=graph_pit)

        graph_gold = 'graph[directed 1node[ id "Reward" label "Reward"]node[ id "Next State At Gold" label "Next ' \
                     'State At Gold"]edge[source "Reward" target "Next State At Gold"]] '
        model_gold = CausalModel(data=data, treatment=['Reward'], outcome=['Next State At Gold'], graph=graph_gold)

        graph_wumpus = 'graph[directed 1node[ id "Reward" label "Reward"]node[ id "Next State At Wumpus" label "Next ' \
                       'State At Wumpus"]edge[source "Reward" target "Next State At Wumpus"]] '
        model_wumpus = CausalModel(data=data, treatment=['Reward'], outcome=['Next State At Wumpus'],
                                   graph=graph_wumpus)

        graph_breeze = 'graph[directed 1node[ id "Reward" label "Reward"]node[ id "Next State At Breeze" label "Next ' \
                       'State At Breeze"]edge[source "Reward" target "Next State At Breeze"]] '
        model_breeze = CausalModel(data=data, treatment=['Reward'], outcome=['Next State At Breeze'],
                                   graph=graph_breeze)

        graph_stench = 'graph[directed 1node[ id "Reward" label "Reward"]node[ id "Next State At Stench" label "Next ' \
                       'State At Stench"]edge[source "Reward" target "Next State At Stench"]] '
        model_stench = CausalModel(data=data, treatment=['Reward'], outcome=['Next State At Stench'],
                                   graph=graph_stench)

        objects = ['Pit', 'Gold', 'Wumpus', 'Breeze', 'Stench']
        models = [model_pit, model_gold, model_wumpus, model_breeze, model_stench]

        for i in range(len(objects)):
            print('\033[1m' + f'\n\n\n{objects[i]}:\n' + '\033[0m')
            # models[i].view_model()

            # II. Identify causal effect and return target estimands
            identified_estimand = models[i].identify_effect()

            # III. Estimate the target estimand using a statistical method.
            print('Common Causes:', models[i]._common_causes)

            start = time()
            estimate = models[i].estimate_effect(identified_estimand,
                                                 method_name="backdoor.linear_regression",
                                                 test_significance=True)
            #     estimate = models[i].estimate_effect(identified_estimand,0
            #                                      method_name="backdoor.propensity_score_matching")
            print('Time to estimate:', time() - start)
            print('Estimate:\n', estimate)

            # IV. Refute the obtained estimate using multiple robustness checks.
            refute_results = models[i].refute_estimate(identified_estimand, estimate,
                                                       method_name="random_common_cause")
            print(refute_results)
            refute_results = models[i].refute_estimate(identified_estimand, estimate,
                                                       method_name="placebo_treatment_refuter")
            print(refute_results)
            refute_results = models[i].refute_estimate(identified_estimand, estimate,
                                                       method_name="data_subset_refuter")
            print(refute_results)
            refute_results = models[i].refute_estimate(identified_estimand, estimate,
                                                       method_name="bootstrap_refuter")
            print(refute_results)
            # refute_results = models[i].refute_estimate(identified_estimand, estimate,
            #                                            method_name="add_unobserved_common_cause",
            #                                            effect_strength_on_outcome=0)
            # print(refute_results)
            # refute_results = models[i].refute_estimate(identified_estimand, estimate,
            #                                            method_name="dummy_outcome_refuter")
            # print(refute_results)

        return

    def object_to_reward(self):
        """This method computes the causal estimate from the objects to the rewards not considering the common
        causes."""

        data = pd.DataFrame.from_dict(self.history)

        graph_pit = 'graph[directed 1node[ id "Reward" label "Reward"]node[ id "Next State At Pit" label "Next State ' \
                    'At Pit"]edge[source "Next State At Pit" target "Reward"]] '
        model_pit = CausalModel(data=data, treatment=['Next State At Pit'], outcome=['Reward'], graph=graph_pit)

        graph_gold = 'graph[directed 1node[ id "Reward" label "Reward"]node[ id "Next State At Gold" label "Next ' \
                     'State At Gold"]edge[source "Next State At Gold" target "Reward"]] '
        model_gold = CausalModel(data=data, treatment=['Next State At Gold'], outcome=['Reward'], graph=graph_gold)

        graph_wumpus = 'graph[directed 1node[ id "Reward" label "Reward"]node[ id "Next State At Wumpus" label "Next ' \
                       'State At Wumpus"]edge[source "Next State At Wumpus" target "Reward"]] '
        model_wumpus = CausalModel(data=data, treatment=['Next State At Wumpus'], outcome=['Reward'],
                                   graph=graph_wumpus)

        graph_breeze = 'graph[directed 1node[ id "Reward" label "Reward"]node[ id "Next State At Breeze" label "Next ' \
                       'State At Breeze"]edge[source "Next State At Breeze" target "Reward"]] '
        model_breeze = CausalModel(data=data, treatment=['Next State At Breeze'], outcome=['Reward'],
                                   graph=graph_breeze)

        graph_stench = 'graph[directed 1node[ id "Reward" label "Reward"]node[ id "Next State At Stench" label "Next ' \
                       'State At Stench"]edge[source "Next State At Stench" target "Reward"]] '
        model_stench = CausalModel(data=data, treatment=['Next State At Stench'], outcome=['Reward'],
                                   graph=graph_stench)

        objects = ['Pit', 'Gold', 'Wumpus', 'Breeze', 'Stench']
        models = [model_pit, model_gold, model_wumpus, model_breeze, model_stench]
        causal_estimates = []

        for i in range(len(objects)):
            print('\033[1m' + f'\n\n\n{objects[i]}:\n' + '\033[0m')
            models[i].view_model()

            # II. Identify causal effect and return target estimands
            identified_estimand = models[i].identify_effect()

            # III. Estimate the target estimand using a statistical method.
            print('Common Causes:', models[i]._common_causes)

            start = time()
            estimate = models[i].estimate_effect(identified_estimand,
                                                 method_name="backdoor.linear_regression",
                                                 test_significance=True)

            causal_estimates.append(estimate)
            print('Time to estimate:', time() - start)
            print('Estimate:\n', estimate)

            # IV. Refute the obtained estimate using multiple robustness checks.
            refute_results = models[i].refute_estimate(identified_estimand, estimate,
                                                       method_name="random_common_cause")
            print(refute_results)
            refute_results = models[i].refute_estimate(identified_estimand, estimate,
                                                       method_name="placebo_treatment_refuter")
            print(refute_results)
            refute_results = models[i].refute_estimate(identified_estimand, estimate,
                                                       method_name="data_subset_refuter")
            print(refute_results)
            refute_results = models[i].refute_estimate(identified_estimand, estimate,
                                                       method_name="bootstrap_refuter")
            print(refute_results)
            # refute_results = models[i].refute_estimate(identified_estimand, estimate,
            #                                            method_name="add_unobserved_common_cause",
            #                                            effect_strength_on_outcome=0)
            # print(refute_results)
            # refute_results = models[i].refute_estimate(identified_estimand, estimate,
            #                                            method_name="dummy_outcome_refuter")
            # print(refute_results)

        return

    def object_to_reward_common_causes(self):
        """This method computes the causal estimate from the objects to the rewards considering the common causes."""

        data = pd.DataFrame.from_dict(self.history)

        model_pit = CausalModel(data=data, outcome=['Reward'], treatment=['Next State At Pit'],
                                common_causes=['Next State At Gold', 'Next State At Wumpus', 'Next State At Breeze'])

        model_gold = CausalModel(data=data, outcome=['Reward'], treatment=['Next State At Gold'],
                                 common_causes=['Next State At Pit', 'Next State At Wumpus', 'Next State At Breeze'])

        model_wumpus = CausalModel(data=data, outcome=['Reward'], treatment=['Next State At Wumpus'],
                                   common_causes=['Next State At Pit', 'Next State At Gold', 'Next State At Breeze'])

        model_breeze = CausalModel(data=data, outcome=['Reward'], treatment=['Next State At Breeze'],
                                   common_causes=['Next State At Pit', 'Next State At Gold', 'Next State At Wumpus'])

        model_stench = CausalModel(data=data, outcome=['Reward'], treatment=['Next State At Stench'],
                                   common_causes=['Next State At Pit', 'Next State At Gold', 'Next State At Wumpus',
                                                  'Next State At Breeze'])

        objects = ['Pit', 'Gold', 'Wumpus', 'Breeze', 'Stench']
        models = [model_pit, model_gold, model_wumpus, model_breeze, model_stench]
        causal_estimates = []

        for i in range(len(objects)):
            print('\033[1m' + f'\n\n\n{objects[i]}:\n' + '\033[0m')
            # models[i].view_model()

            # II. Identify causal effect and return target estimands
            identified_estimand = models[i].identify_effect()

            # III. Estimate the target estimand using a statistical method.
            print('Common Causes:', models[i]._common_causes)
            start = time()
            estimate = models[i].estimate_effect(identified_estimand,
                                                 method_name="backdoor.linear_regression",
                                                 test_significance=True, control_value=0, treatment_value=1)
            #     estimate = models[i].estimate_effect(identified_estimand,
            #                                      method_name="backdoor.propensity_score_matching")
            causal_estimates.append(estimate)
            print('Time to estimate:', time() - start)
            print('Estimate:\n', estimate)

            # IV. Refute the obtained estimate using multiple robustness checks.
            refute_results = models[i].refute_estimate(identified_estimand, estimate,
                                                       method_name="random_common_cause")
            print(refute_results)
            refute_results = models[i].refute_estimate(identified_estimand, estimate,
                                                       method_name="placebo_treatment_refuter")
            print(refute_results)
            refute_results = models[i].refute_estimate(identified_estimand, estimate,
                                                       method_name="data_subset_refuter")
            print(refute_results)
            refute_results = models[i].refute_estimate(identified_estimand, estimate,
                                                       method_name="bootstrap_refuter")
            print(refute_results)
            # refute_results = models[i].refute_estimate(identified_estimand, estimate,
            #                                            method_name="add_unobserved_common_cause",
            #                                            effect_strength_on_outcome=0)
            # print(refute_results)
            # refute_results = models[i].refute_estimate(identified_estimand, estimate,
            #                                            method_name="dummy_outcome_refuter")
            # print(refute_results)

        return

    def state_to_next_state(self):
        """This method computes the causal estimate from the state to the next state considering the actions as effect
        modifiers."""

        data = pd.DataFrame.from_dict(self.history)

        models = {'Breeze': [], 'Gold': [], 'Pit': [], 'Stench': [], 'Wumpus': []}
        estimands = {'Breeze': [], 'Gold': [], 'Pit': [], 'Stench': [], 'Wumpus': []}
        estimates = {'Breeze': [], 'Gold': [], 'Pit': [], 'Stench': [], 'Wumpus': []}
        refutation_estimates = {'Breeze': [], 'Gold': [], 'Pit': [], 'Stench': [], 'Wumpus': []}
        actions = ['Left', 'Right', 'Up', 'Down']
        positions = ['Left', 'Right', 'Up', 'Down']
        environment_objects = ['Breeze', 'Gold', 'Pit', 'Stench', 'Wumpus']

        for environment_object in environment_objects:
            for action in actions:
                graph = f'graph[directed 1node[ id "Next State At {environment_object}" label "Next State At ' \
                        f'{environment_object}"]node[ id "State {action} {environment_object}" label "State ' \
                        f'{action} {environment_object}"]edge[source "State {action} {environment_object}" target ' \
                        f'"Next State At {environment_object}"]]'
                common_causes = ['Next State At Gold', 'Next State At Wumpus', 'Next State At Breeze',
                                 'Next State At Stench', 'Next State At Pit']
                common_causes.remove(f'Next State At {environment_object}')
                model = CausalModel(data=data, treatment=[f'State {action} {environment_object}'],
                                    outcome=[f'Next State At {environment_object}'], common_causes=common_causes,
                                    effect_modifiers=['Action Left', 'Action Right', 'Action Up', 'Action Down'])
                models[f'{environment_object}'].append(model)

        # II. Identify causal effect and return target estimands
        for environment_object in environment_objects:
            for i in range(len(actions)):
                estimands[f'{environment_object}'].append(
                    models[f'{environment_object}'][i].identify_effect(proceed_when_unidentifiable=True))

        # III. Estimate the target estimand using a statistical method.
        for environment_object in environment_objects:
            for i, position in enumerate(positions):
                for action in actions:
                    dml_estimate = models[f'{environment_object}'][i].estimate_effect(
                        estimands[f'{environment_object}'][i], method_name="backdoor.econml.dml.DML",
                        control_value=0, treatment_value=1, target_units=lambda df: df[f"Action {action}"] > 0,
                        confidence_intervals=False, method_params={
                            "init_params": {'model_y': GradientBoostingRegressor(),
                                            'model_t': GradientBoostingRegressor(),
                                            "model_final": LassoCV(fit_intercept=False),
                                            'featurizer': PolynomialFeatures(degree=1, include_bias=False)},
                            "fit_params": {}})
                    estimates[f'{environment_object}'].append(dml_estimate.value)
        print('\nEstimates:', estimates)

        # IV. Refute the obtained estimate using multiple robustness checks.
        # for environment_object in environment_objects:
        #     index = 0
        #     for i, action in enumerate(actions):
        #         while index < 16:
        #             refute_results = models[f'{environment_object}'][i].refute_estimate(
        #                 estimands[f'{environment_object}'][i], estimates[f'{environment_object}'][index],
        #                 method_name="random_common_cause")
        #             refutation_estimates[f'{environment_object}'].append(refute_results)
        #             index += 1
        #             if index % 4 == 0:
        #                 break
        # print('Refute Results - Add Random Cause:', refutation_estimates)
        # refute_results = model.refute_estimate(identified_estimand, estimate,
        #                                        method_name="placebo_treatment_refuter")
        # print('Refute Results - Placebo Treatment:', refute_results)
        # refute_results = model.refute_estimate(identified_estimand, estimate,
        #                                        method_name="data_subset_refuter")
        # print('Refute Results - Data Subsets Validation:', refute_results)
        # refute_results = model.refute_estimate(identified_estimand, estimate,
        #                                        method_name="bootstrap_refuter")
        # print('Refute Results - Bootstrap Refuter:', refute_results)
        # refute_results = model.refute_estimate(identified_estimand, estimate,
        #                                        method_name="add_unobserved_common_cause",
        #                                        effect_strength_on_outcome=0)
        # print('Refute Results - Add Unobserved Common Cause:', refute_results)
        # refute_results = model.refute_estimate(identified_estimand, estimate,
        #                                        method_name="dummy_outcome_refuter")
        # print('Refute Results - Dummy Outcome Refuter:', refute_results)

        return

    def state_to_reward(self):
        """This method computes the causal estimate from the state to the rewards considering the actions as effect
        modifiers."""

        data = pd.DataFrame.from_dict(self.history)

        models = {'Breeze': [], 'Gold': [], 'Pit': [], 'Stench': [], 'Wumpus': []}
        estimands = {'Breeze': [], 'Gold': [], 'Pit': [], 'Stench': [], 'Wumpus': []}
        estimates = {'Breeze': [], 'Gold': [], 'Pit': [], 'Stench': [], 'Wumpus': []}
        estimate_values = {'Breeze': [], 'Gold': [], 'Pit': [], 'Stench': [], 'Wumpus': []}
        refutation_estimates = {'Breeze': [], 'Gold': [], 'Pit': [], 'Stench': [], 'Wumpus': []}
        actions = ['Left', 'Right', 'Up', 'Down']
        positions = ['Left', 'Right', 'Up', 'Down']
        environment_objects = ['Breeze', 'Gold', 'Pit', 'Stench', 'Wumpus']

        for environment_object in environment_objects:
            for action in actions:
                common_causes = ['Next State At Gold', 'Next State At Wumpus', 'Next State At Breeze',
                                 'Next State At Stench', 'Next State At Pit']
                common_causes.remove(f'Next State At {environment_object}')
                model = CausalModel(data=data, treatment=[f'State {action} {environment_object}'],
                                    outcome=['Reward'], common_causes=common_causes,
                                    effect_modifiers=['Action Left', 'Action Right', 'Action Up', 'Action Down'])
                models[f'{environment_object}'].append(model)

        # II. Identify causal effect and return target estimands
        for environment_object in environment_objects:
            for i in range(len(actions)):
                estimands[f'{environment_object}'].append(
                    models[f'{environment_object}'][i].identify_effect(proceed_when_unidentifiable=True))

        # III. Estimate the target estimand using a statistical method.
        for environment_object in environment_objects:
            for i, position in enumerate(positions):
                for action in actions:
                    dml_estimate = models[f'{environment_object}'][i].estimate_effect(
                        estimands[f'{environment_object}'][i], method_name="backdoor.econml.dml.DML",
                        control_value=0, treatment_value=1, target_units=lambda df: df[f"Action {action}"] > 0,
                        confidence_intervals=False, method_params={
                            "init_params": {'model_y': GradientBoostingRegressor(),
                                            'model_t': GradientBoostingRegressor(),
                                            "model_final": LassoCV(fit_intercept=False),
                                            'featurizer': PolynomialFeatures(degree=1, include_bias=False)},
                            "fit_params": {}})
                    estimates[f'{environment_object}'].append(dml_estimate)
                    estimate_values[f'{environment_object}'].append(dml_estimate.value)

        print('Estimates:', estimate_values)

        # # IV. Refute the obtained estimate using multiple robustness checks.
        for environment_object in environment_objects:
            index = 0
            for i, action in enumerate(actions):
                while index < 16:
                    refute_results = models[f'{environment_object}'][i].refute_estimate(
                        estimands[f'{environment_object}'][i], estimates[f'{environment_object}'][index],
                        method_name="random_common_cause")
                    refutation_estimates[f'{environment_object}'].append(refute_results)
                    index += 1
                    if index % 4 == 0:
                        break
                    # print('Refute Results - Add Random Cause:', refute_results)
    #     print('Refutation Estimates:', refutation_estimates)
    #     refute_results = model.refute_estimate(identified_estimand, estimate,
    #                                            method_name="placebo_treatment_refuter")
    #     print('Refute Results - Placebo Treatment:', refute_results)
    #     refute_results = model.refute_estimate(identified_estimand, estimate,
    #                                            method_name="data_subset_refuter")
    #     print('Refute Results - Data Subsets Validation:', refute_results)
    #     refute_results = model.refute_estimate(identified_estimand, estimate,
    #                                            method_name="bootstrap_refuter")
    #     print('Refute Results - Bootstrap Refuter:', refute_results)
    #     refute_results = model.refute_estimate(identified_estimand, estimate,
    #                                            method_name="add_unobserved_common_cause",
    #                                            effect_strength_on_outcome=0)
    #     print('Refute Results - Add Unobserved Common Cause:', refute_results)
    #     refute_results = model.refute_estimate(identified_estimand, estimate,
    #                                            method_name="dummy_outcome_refuter")
    #     print('Refute Results - Dummy Outcome Refuter:', refute_results)

        return
