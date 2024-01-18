import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
from scipy.optimize import minimize
from pprint import pprint
import warnings

warnings.filterwarnings('ignore')

fd_1 = pd.read_csv('Results 20:21.csv')
fd_2 = pd.read_csv('Results 21:22.csv')
fd_3 = pd.read_csv('Results 22:23.csv')

csv_files = glob.glob('*.{}'.format('csv'))
csv_files

fixtures = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
fixtures

fixtures = fixtures[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]

sns.set()
#plt.hist(fixtures[['FTHG', 'FTAG']].values, range(10), label=['Home Team', 'Away Team'], density=True)
#plt.xticks([i-0.5 for i in range(10)], [i for i in range(10)])
#plt.xlabel('Goals')
#plt.ylabel('Proportion of matches')
#plt.legend(loc='upper right', fontsize=13)
#plt.title('number of goals scored per match', size=12)

def log_probability(goals_home_observed,goals_away_observed,home_attack,home_defence,away_attack,away_defence,home_advantage):
    goal_expectation_home = np.exp(home_attack+away_defence+home_advantage)
    goal_expectation_away = np.exp(away_attack+home_defence)
    
    if goal_expectation_home < 0 or goal_expectation_away < 0:
        return 10000
    
    home_lp = poisson.pmf(goals_home_observed, goal_expectation_home)
    away_lp = poisson,pmf(goals_away_observed, goal_expectation_away)
    
    log_lp = np.log(home_lp)+np.log(away_lp)
    
    return -log_lp

def log_likelihood(goals_home_observed,goals_away_observed,home_attack,home_defence,away_attack,away_defence,home_advantage):
    
    goal_expectation_home = np.exp(home_attack + away_defence + home_advantage)
    goal_expectation_away = np.exp(away_attack + home_defence)

    if goal_expectation_home < 0 or goal_expectation_away < 0:
        return 10000    

    home_llk = poisson.pmf(goals_home_observed, goal_expectation_home)
    away_llk = poisson.pmf(goals_away_observed, goal_expectation_away)

    log_llk = np.log(home_llk) + np.log(away_llk)

    return -log_llk

def fit_poisson_model():
    teams = np.sort(np.unique(np.concatenate([fixtures["HomeTeam"], fixtures["AwayTeam"]])))
    n_teams = len(teams)

    params = np.concatenate((np.random.uniform(0.5, 1.5, (n_teams)),np.random.uniform(0, -1, (n_teams)),[0.25],))

    def _fit(params, fixtures, teams):
        attack_params = dict(zip(teams, params[:n_teams]))
        defence_params = dict(zip(teams, params[n_teams : (2 * n_teams)]))
        home_advantage = params[-1]

        llk = list()
        for idx, row in fixtures.iterrows():

            tmp = log_likelihood(row["FTHG"],row["FTAG"],attack_params[row["HomeTeam"]],defence_params[row["HomeTeam"]],attack_params[row["AwayTeam"]],defence_params[row["AwayTeam"]],home_advantage,)
            llk.append(tmp)

        return np.sum(llk)

    options = {
        "maxiter": 100,
        "disp": False,
    }

    constraints = [{"type": "eq", "fun": lambda x: sum(x[:n_teams]) - n_teams}]

    res = minimize(_fit,params,args=(fixtures, teams),constraints=constraints,options=options,)

    model_params = dict(zip(["attack_"+team for team in teams]+["defence_"+team for team in teams]+["home_adv"],res["x"],))

    return model_params

model_params=fit_poisson_model()

def match_prediction(home_team, away_team, params, max_goals=10):
    
    home_attack = params['attack_'+home_team]
    home_defence = params['defence_'+home_team]
    away_attack = params['attack_'+away_team]
    away_defence = params['defence_'+away_team]
    home_advantage = params['home_adv']
    
    home_goal_expectation = np.exp(home_attack+away_defence+home_advantage)
    away_goal_expectation = np.exp(away_attack+home_defence)
    
    home_probs = poisson.pmf(list(range(max_goals+1)), home_goal_expectation)
    away_probs = poisson.pmf(range(max_goals+1), away_goal_expectation)
    
    probability_matrix = np.outer(home_probs, away_probs)
    
    return probability_matrix

#probs = match_prediction('Man United', 'Liverpool', model_params, 4)
#pprint(probs)

pprint(model_params)
