#COVID_dashboard
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm.notebook import tqdm
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_log_error, mean_squared_error
import datetime
 
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
 
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import time
 
#Remember to be connected to VPN
state_data = pd.read_csv('https://api.covid19india.org/csv/latest/state_wise_daily.csv')
state_data['Date'] = pd.to_datetime(state_data['Date'], infer_datetime_format = True, format = '%d-%m-%y')
 
pop_data = pd.read_csv('State_population.csv')
abb_dict = dict(zip(pop_data['Abbreviation'], pop_data['State or union territory']))
abb_dict['TT'] = 'India'
abb_pop_dict = dict(zip(pop_data['Abbreviation'], pop_data['Population']))
abb_pop_dict['TT'] = sum(abb_pop_dict.values())
 
df_conf = state_data[state_data['Status'] == 'Confirmed']
df_recov = state_data[state_data['Status'] == 'Recovered']
df_dead = state_data[state_data['Status'] == 'Deceased']
 
for df in [df_conf, df_recov, df_dead]:
    for name in df.columns[2:-1]:
        df[name + '_cumsum'] = df[name].cumsum()
 
def prepare_data(valid_days = 7, forecast_days = 10):
    data = pd.DataFrame(columns= ['Area', 'Date', 'Id', 'Province_State', 'Country_Region','ConfirmedCases','Fatalities','Recovered', 'CurrentInfections'])
    #Preparing the dataframe for existing data
    for st_name in list(state_data.columns)[2:-1]:
        temp_data = pd.DataFrame(columns= ['Area', 'Date', 'Id', 'Province_State', 'Country_Region',
                                           'ConfirmedCases','Fatalities','Recovered', 'Removed', 'CurrentInfections'], 
                             index = list(range(data.shape[0] + 0 , data.shape[0] + df_conf[st_name].shape[0])))
        temp_data['Area'] = st_name
        temp_data['Date'] = list(df_conf['Date'])
        temp_data['Id'] = list(temp_data.index)
        temp_data['Province_State'] = abb_dict[st_name]
        temp_data['Country_Region'] = 'India'
        temp_data['ConfirmedCases'] = list(df_conf[st_name + '_cumsum'])
        temp_data['Fatalities'] = list(df_dead[st_name + '_cumsum'])
        temp_data['Recovered'] = list(df_recov[st_name + '_cumsum'])
 
        temp_data['Removed'] = temp_data['Fatalities'] + temp_data['Recovered']
        temp_data['CurrentInfections'] = temp_data['ConfirmedCases'] - temp_data['Removed']
        data = pd.concat([data, temp_data])
 
    TEST_MIN_DATE = pd.Timestamp(data['Date'].max() - datetime.timedelta(days= valid_days))
    print("Test_min_date:", TEST_MIN_DATE)
    DATE_BORDER = data['Date'].max()
    print("date BORDER:", DATE_BORDER)
    train_full = data[data['Date'] <= DATE_BORDER]
    train = data[data['Date'] <= TEST_MIN_DATE]
    valid = data[(data['Date'] > TEST_MIN_DATE) & (data['Date'] <= DATE_BORDER)]
 
    base = data['Date'].max()
    date_list = [pd.Timestamp(base + datetime.timedelta(days=x+1)) for x in range(forecast_days)]
 
    #Preparing the dataframe for rows of forecasting days
    for st_name in list(state_data.columns)[2:-1]:
        temp_data = pd.DataFrame(columns= ['Area', 'Date', 'Id', 'Province_State', 'Country_Region',
                                           'ConfirmedCases','Fatalities','Recovered', 'Removed', 'CurrentInfections'], 
                             index = list(range(data.shape[0] + 0 , data.shape[0] + forecast_days)))
        temp_data['Area'] = st_name
        temp_data['Date'] = date_list
        temp_data['Id'] = list(temp_data.index)
        temp_data['Province_State'] = abb_dict[st_name]
        temp_data['Country_Region'] = 'India'
        data = pd.concat([data, temp_data])
 
    test = data[data['Date'] > TEST_MIN_DATE]
    test = test.rename(columns={'Id': 'ForecastId'})
 
    #DATE_BORDER_2 = data['Date'].max()
 
    # Split the test into public & private
    test_public = test[test['Date'] <= DATE_BORDER]
    test_private = test[test['Date'] > DATE_BORDER]
 
    submission = pd.DataFrame(columns = ['ForecastId', 'ConfirmedCases', 'Fatalities', 'Recovered', 'Removed', 'CurrentInfections'])
    submission['ForecastId'] = list(test['ForecastId'])
    submission['ConfirmedCases'] = 0
    submission['Fatalities'] = 0
    submission['Recovered'] = 0
    submission['Removed'] = 0
    submission['CurrentInfections'] = 0
    submission = submission.set_index(['ForecastId'])
    #print(submission)
 
 
    # Use a multi-index for easier slicing
    train_full.set_index(['Area', 'Date'], inplace=True)
    train.set_index(['Area', 'Date'], inplace=True)
    valid.set_index(['Area', 'Date'], inplace=True)
    test_public.set_index(['Area', 'Date'], inplace=True)
    test_private.set_index(['Area', 'Date'], inplace=True)
   
    return train_full, train, valid, test, test_public, test_private, submission
 
train_full, train, valid, test, test_public, test_private, submission = prepare_data(valid_days= 10, forecast_days=30)
 
# Susceptible equation
def dS_dt(S, I, R_t, t_inf):
    return -(R_t * (t_inf**-1)) * I * S
 
 
# Exposed equation
def dE_dt(S, E, I, R_t, t_inf, t_inc):
    return (R_t * (t_inf**-1)) * I * S - (E * (t_inc**-1))
 
 
# Infected equation
def dI_dt(I, E, t_inc, t_inf):
    return (E * (t_inc**-1)) - (I * (t_inf**-1))
 
# Recovered equation
def dR_dt(I, t_inf):
    return ( I * (t_inf**-1))
 
def SEIR_model(t, y, R_t, t_inc=2.9, t_inf=5.2):
    """
 
    :param t: Time step for solve_ivp
    :param y: Previous solution or initial values
   :param R_t: Reproduction number
    :param t_inc: Average incubation period. Default 5.2 days
    :param t_inf: Average infectious period. Default 2.9 days
    :return:
    """
    if callable(R_t):
        reprod = R_t(t)
    else:
        reprod = R_t
       
    S, E, I, R = y
   
    S_out = dS_dt(S, I, reprod, t_inf)
    E_out = dE_dt(S, E, I, reprod, t_inf, t_inc)
    I_out = dI_dt(I, E, t_inc, t_inf)
    R_out = dR_dt(I,t_inf)
    
    return [S_out, E_out, I_out, R_out]
 
OPTIM_DAYS = 30  # Number of days to use for the optimisation evaluation
 
date_0 = pd.to_datetime('14-Mar-2020') #starting date of dataset
date_1 = pd.to_datetime('26-Mar-2020') #phase 1 begins
date_2 = pd.to_datetime('15-Apr-2020') #phase 1 ends, phase 2 begins
date_3 = pd.to_datetime('03-May-2020') #phase 2 ends, phase 3 begins
date_4 = pd.to_datetime('17-May-2020') #phase 3 ends, phase 4 begins
date_5 = pd.to_datetime('31-May-2020') #phase 4 ends
 
# Use a Hill decayed reproduction number (with Lockdown effects)
def eval_model_decay(params, data, population, return_solution=False, forecast_days=0, extra_days = 0):
    R_0, t_inc, t_inf, k, L = params 
    N = population
    n_infected = data['ConfirmedCases'].iloc[0]
    max_days = len(data) + forecast_days
    max_days_2 = len(data) + forecast_days + extra_days
   
    # https://github.com/SwissTPH/openmalaria/wiki/ModelDecayFunctions  
    # Hill decay. Initial values: R_0=2.2, k=2, L=50
    def time_varying_reproduction(t):
        return R_0/ (1 + (t/L)**k)
   
    initial_state = [(N - n_infected)/ N, 0, n_infected / N, 0]
    args = (time_varying_reproduction, t_inc, t_inf)
           
    sol = solve_ivp(SEIR_model, [0, max_days_2], initial_state, args=args, t_eval=np.arange(0, max_days_2))
   
    sus, exp, inf, rec = sol.y
   
    y_pred_cases = np.clip(inf + rec, 0, np.inf) * population
    y_true_cases = data['ConfirmedCases'].values.copy()
    y_pred_recov = np.clip(rec , 0, np.inf) * population
    y_true_recov = data['Removed'].values
    y_pred_inf = np.clip(inf , 0, np.inf) * population 
    y_true_inf = data['CurrentInfections'].values
       
    optim_days = min(OPTIM_DAYS, len(data))  # Days to optimise for
    weights = (1 / np.arange(1, optim_days+1)**3)[::-1]  # Recent data is more heavily weighted
   
    msle_cases = mean_squared_log_error(y_true_cases[-optim_days:], y_pred_cases[ : max_days][-optim_days:], weights)
    msle_recov = mean_squared_log_error(y_true_recov[-optim_days:], y_pred_recov[ : max_days][-optim_days:], weights)
    msle_inf = mean_squared_log_error(y_true_inf[-optim_days:], y_pred_inf[ : max_days][-optim_days:], weights)
   
    msle_final = np.average([msle_cases, msle_recov, msle_inf], weights = [0.4, 0.1, 0.5])
       
    if return_solution:
        return msle_final, sol
    else:
        return msle_final
 
# Use a Hill decayed reproduction number
def eval_model_decay_modified(params, data, population, return_solution=False, forecast_days=0, extra_days = 0):
    R_0, t_inc, t_inf, k, L = params 
    N = population
    n_infected = data['ConfirmedCases'].iloc[0]
    max_days = len(data) + forecast_days
    max_days_2 = len(data) + forecast_days + extra_days
   
    # https://github.com/SwissTPH/openmalaria/wiki/ModelDecayFunctions  
    # Hill decay. Initial values: R_0=2.2, k=2, L=50
    def time_varying_reproduction(t):
        if (t >= (date_1 - date_0)/ datetime.timedelta(days=1)) & (t < (date_2 - date_0)/ datetime.timedelta(days=1)):
            alpha = 0.7
        elif (t >= (date_2 - date_0)/ datetime.timedelta(days=1)) & (t < (date_3 - date_0)/ datetime.timedelta(days=1)):
            alpha = 0.5
        elif (t >= (date_3 - date_0)/ datetime.timedelta(days=1)) & (t < (date_4 - date_0)/ datetime.timedelta(days=1)):
            alpha = 0.3
        elif (t >= (date_4 - date_0)/ datetime.timedelta(days=1)) & (t < (date_5 - date_0)/ datetime.timedelta(days=1)):
            alpha = 0.6
        else:
            alpha = 1
        return (R_0 * alpha)/ (1 + (t/L)**k)
   
    initial_state = [(N - n_infected)/ N, 0, n_infected / N, 0]
    args = (time_varying_reproduction, t_inc, t_inf)
           
    sol = solve_ivp(SEIR_model, [0, max_days_2], initial_state, args=args, t_eval=np.arange(0, max_days_2))
   
    sus, exp, inf, rec = sol.y
   
    y_pred_cases = np.clip(inf + rec, 0, np.inf) * population
    y_true_cases = data['ConfirmedCases'].values.copy()
    y_pred_recov = np.clip(rec , 0, np.inf) * population
    y_true_recov = data['Removed'].values
    y_pred_inf = np.clip(inf , 0, np.inf) * population 
    y_true_inf = data['CurrentInfections'].values
       
    optim_days = min(OPTIM_DAYS, len(data))  # Days to optimise for
    weights = (1 / np.arange(1, optim_days+1)**3)[::-1]  # Recent data is more heavily weighted
   
    msle_cases = mean_squared_log_error(y_true_cases[-optim_days:], y_pred_cases[ : max_days][-optim_days:], weights)
    msle_recov = mean_squared_log_error(y_true_recov[-optim_days:], y_pred_recov[ : max_days][-optim_days:], weights)
    msle_inf = mean_squared_log_error(y_true_inf[-optim_days:], y_pred_inf[ : max_days][-optim_days:], weights)
   
    msle_final = np.average([msle_cases, msle_recov, msle_inf], weights = [0.4, 0.1, 0.5])
       
    if return_solution:
        return msle_final, sol
    else:
        return msle_final
  
def use_last_value(train_data, valid_data, test_data):
    lv = train_data[['ConfirmedCases', 'Fatalities']].iloc[-1].values
   
    forecast_ids = test_data['ForecastId']
    submission.loc[forecast_ids, ['ConfirmedCases', 'Fatalities']] = lv
   
    if valid_data is not None:
        y_pred_valid = np.ones((len(valid_data), 2)) * lv.reshape(1, 2)
        y_true_valid = valid_data[['ConfirmedCases', 'Fatalities']]
 
        msle_cases = mean_squared_log_error(y_true_valid['ConfirmedCases'], y_pred_valid[:, 0])
        msle_fat = mean_squared_log_error(y_true_valid['Fatalities'], y_pred_valid[:, 1])
        msle_final = np.mean([msle_cases, msle_fat])
 
        return msle_final
 
def plot_model_results_plotly(y_pred, train_data, state, valid_data=None, res = None, valid_mlse = None, test_index = None):
    # Initialize figure with subplots
    fig = make_subplots(
            rows=6, cols=2,
            #column_widths=[0.4, 0.6],
            #row_heights=[0.25, 0.25, 0.25, 0.25],
            horizontal_spacing=0.1,
            #vertical_spacing=0.2,
            specs=[[{"rowspan": 2, "secondary_y": True},{"rowspan": 3, "secondary_y": True}],
                   [None,None],
                   [{"rowspan": 2, "secondary_y": True},None],
                   [None,{"rowspan": 3, "secondary_y": True}],
                   [{"rowspan": 2, "secondary_y": True},None],
                   [None,None]],
            subplot_titles=("Confirmed Cases (Forecast of %s days)"%(len(test_index)),"Confirmed Cases (Long-term forecast of %s days)"%(len(y_pred)-len(train_data)-len(test_index)),
                            "Current infections (Forecast of %s days)"%(len(test_index)),
                            "Current infections (Long-term forecast of %s days)"%(len(y_pred)-len(train_data)-len(test_index)),
                            "Removed(Dead + Recovered) (Forecast of %s days)"%(len(test_index))
                             ),
           
                        )
 
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)
    styling = {'linewidth':2.5, 'actual_opacity' : 0.9, 'model_opacity' : 0.9 }
    
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)
 
    trace_1a = go.Scatter(y = train_data['ConfirmedCases'],
                          x = train_data.index,
                          mode = 'lines',
                          line = dict(color = 'green', width = styling['linewidth']),
                          name = 'Actual confirmed cases',
                          opacity = styling['actual_opacity'],
                          hovertemplate = '%{y:.0f}'
                        )
    trace_1b = go.Scatter(y = y_pred['ConfirmedCases'],
                          x = train_data.index,
                          mode = 'lines',
                          line = dict(color = 'red', width = styling['linewidth']),
                          name = 'Modeled confirmed cases',
                          opacity = styling['model_opacity'],
                          hovertemplate = '%{y:.0f}'
                        )
    trace_1s = go.Scatter(y = y_pred['R'],
                          x = train_data.index,
                          mode = 'lines',
                          line = dict(color = 'blue', width = styling['linewidth']),
                          name = 'Reproduction number',
                          opacity = styling['model_opacity'],
                          hovertemplate = '%{y:.3f}'
                         
                        )
    fig.add_trace(trace_1a, row = 1, col = 1)
    fig.add_trace(trace_1b, row = 1, col = 1)
    fig.add_trace(trace_1s, row = 1, col = 1,secondary_y = True)
 
 
    trace_2a = go.Scatter(y = train_data['CurrentInfections'],
                          x = train_data.index,
                          mode = 'lines',
                          line = dict(color = 'green', width = styling['linewidth']),
                          name = 'Actual Current Infections',
                          opacity = styling['actual_opacity'],
                          hovertemplate = '%{y:.0f}'
                        )
    trace_2b = go.Scatter(y = y_pred['CurrentInfections'],
                          x = train_data.index,
                          mode = 'lines',
                          line = dict(color = 'red', width = styling['linewidth']),
                          name = 'Modeled Current Infections',
                          opacity = styling['model_opacity'],
                          hovertemplate = '%{y:.0f}'
                        )
    fig.add_trace(trace_2a, row = 3, col = 1)
    fig.add_trace(trace_2b, row = 3, col = 1)
 
 
    trace_3a = go.Scatter(y = train_data['Removed'],
                          x = train_data.index,
                          mode = 'lines',
                          line = dict(color = 'green', width = styling['linewidth']),
                          name = 'Actual Removed',
                          opacity = styling['actual_opacity'],
                          hovertemplate = '%{y:.0f}'
                        )
    trace_3b = go.Scatter(y = y_pred['Removed'],
                          x = train_data.index,
                          mode = 'lines',
                          line = dict(color = 'red', width = styling['linewidth']),
                          name = 'Modeled Removed',
                          opacity = styling['model_opacity'],
                          hovertemplate = '%{y:.0f}'
                        )
    fig.add_trace(trace_3a, row = 5, col = 1)
    fig.add_trace(trace_3b, row = 5, col = 1)
 
    trace_4a = go.Scatter(y = train_data['ConfirmedCases'],
                          x = y_pred.index,
                          mode = 'lines',
                          line = dict(color = 'green', width = styling['linewidth']),
                          name = 'Actual confirmed',
                          opacity = styling['actual_opacity'],
                          hovertemplate = '%{y:.0f}'
                        )
    trace_4b = go.Scatter(y = y_pred['ConfirmedCases'],
                          x = y_pred.index,
                          mode = 'lines',
                          line = dict(color = 'red', width = styling['linewidth']),
                          name = 'Modeled Confirmed',
                          opacity = styling['model_opacity'],
                          hovertemplate = '%{y:.0f}'
                        )
    trace_4s = go.Scatter(y = y_pred['R'],
                          x = y_pred.index,
                          mode = 'lines',
                          line = dict(color = 'blue', width = styling['linewidth']),
                          name = 'Reproduction number',
                          opacity = styling['model_opacity'],
                          hovertemplate = '%{y:.3f}'                         
                        )
    fig.add_trace(trace_4a, row = 1, col = 2)
    fig.add_trace(trace_4b, row = 1, col = 2)
    fig.add_trace(trace_4s, row = 1, col = 2, secondary_y = True)
 
 
    trace_5a = go.Scatter(y = train_data['CurrentInfections'],
                          x = y_pred.index,
                          mode = 'lines',
                          line = dict(color = 'green', width = styling['linewidth']),
                          name = 'Actual current infections',
                          opacity = styling['actual_opacity'],
                          hovertemplate = '%{y:.0f}'
                        )
    trace_5b = go.Scatter(y = y_pred['CurrentInfections'],
                          x = y_pred.index,
                          mode = 'lines',
                          line = dict(color = 'red', width = styling['linewidth']),
                          name = 'Model current infections',
                          opacity = styling['model_opacity'],
                          hovertemplate = '%{y:.0f}'
                        )
    fig.add_trace(trace_5a, row = 4, col = 2)
    fig.add_trace(trace_5b, row = 4, col = 2)
   
    fig.update_layout(hovermode = "x unified", hoverlabel = {'font_size':16})
      
    
    if res is not None:
        if len(res.x) != 2:
            fig.update_layout(title = 'For ' + '<b>'+str(abb_dict[state])+'</b>' + ' ' + str(f'Initial R_0: {res.x[0]:0.3f}, t_inc: {res.x[1]:0.3f}, t_inf: {res.x[2]:0.3f}, '
              f'k: {res.x[3]:0.3f}, l: {res.x[4]:0.3f}'), title_x = 0.5,
                              showlegend = False,
                              template = 'plotly_dark')
        else:
            fig.update_layout(title = 'For ' + '<b>'+str(abb_dict[state])+'</b>' + ' ' + str(f't_inc: {res.x[0]:0.3f}, t_inf: {res.x[1]:0.3f}, '))
    else:
        fig.update_layout(title ='For ' + str(abb_dict[state]), title_x = 0.5,
                          showlegend = False,
                          template = 'plotly_dark')
       
    if valid_data is not None:
       
        trace_1c = go.Scatter(y = valid_data['ConfirmedCases'],
                          x = valid_data.index,
                          mode = 'lines',
                          line = dict(color = 'green', width = styling['linewidth'], dash = 'dash'),
                          name = 'Actual confirmed cases',
                          opacity = styling['actual_opacity'],
                          hovertemplate = '%{y:.0f}'
                        )
        trace_1d = go.Scatter(y = y_pred.loc[valid_data.index,'ConfirmedCases'],
                              x = valid_data.index,
                              mode = 'lines',
                              line = dict(color = 'red', width = styling['linewidth'], dash = 'dash'),
                              name = 'Modeled confirmed cases (validation)',
                              opacity = styling['model_opacity'],
                              hovertemplate = '%{y:.0f}'
                            )
        trace_1ds = go.Scatter(y = y_pred.loc[valid_data.index,'R'],
                              x = valid_data.index,
                              mode = 'lines',
                              line = dict(color = 'blue', width = styling['linewidth']),
                              name = 'Reproduction number',
                              opacity = styling['model_opacity'],
                              hovertemplate = '%{y:.3f}'
                              
                            )
        fig.add_trace(trace_1c, row = 1, col = 1)
        fig.add_trace(trace_1d, row = 1, col = 1)
        fig.add_trace(trace_1ds, row = 1, col = 1, secondary_y = True)
 
       
        trace_2c = go.Scatter(y = valid_data['CurrentInfections'],
                              x = valid_data.index,
                              mode = 'lines',
                              line = dict(color = 'green', width = styling['linewidth'], dash = 'dash'),
                              name = 'Actual Current Infections',
                              opacity = styling['actual_opacity'],
                              hovertemplate = '%{y:.0f}'
                            )
        trace_2d = go.Scatter(y = y_pred.loc[valid_data.index,'CurrentInfections'],
                              x = valid_data.index,
                              mode = 'lines',
                              line = dict(color = 'red', width = styling['linewidth'], dash = 'dash'),
                              name = 'Modeled Current Infections(validation)',
                              opacity = styling['model_opacity'],
                              hovertemplate = '%{y:.0f}'
                            )
        fig.add_trace(trace_2c, row = 3, col = 1)
        fig.add_trace(trace_2d, row = 3, col = 1)
 
 
        trace_3c = go.Scatter(y = valid_data['Removed'],
                              x = valid_data.index,
                              mode = 'lines',
                              line = dict(color = 'green', width = styling['linewidth'], dash = 'dash'),
                              name = 'Actual Removed',
                              opacity = styling['actual_opacity'],
                              hovertemplate = '%{y:.0f}'
                            )
        trace_3d = go.Scatter(y = y_pred.loc[valid_data.index,'Removed'],
                              x = valid_data.index,
                              mode = 'lines',
                              line = dict(color = 'red', width = styling['linewidth'], dash = 'dash'),
                              name = 'Modeled Removed(validation)',
                              opacity = styling['model_opacity'],
                              hovertemplate = '%{y:.0f}'
                            )
        fig.add_trace(trace_3c, row = 5, col = 1)
        fig.add_trace(trace_3d, row = 5, col = 1)
 
        trace_4c = go.Scatter(y = valid_data['ConfirmedCases'],
                              x = valid_data.index,
                              mode = 'lines',
                              line = dict(color = 'green', width = styling['linewidth'], dash = 'dash'),
                              name = 'Actual confirmed',
                              opacity = styling['actual_opacity'],
                              hovertemplate = '%{y:.0f}'
                            )
        trace_4d = go.Scatter(y = y_pred.loc[valid_data.index,'ConfirmedCases'],
                              x = valid_data.index,
                              mode = 'lines',
                              line = dict(color = 'red', width = styling['linewidth'], dash = 'dash'),
                              name = 'Modeled Confirmed(validation)',
                              opacity = styling['model_opacity'],
                              hovertemplate = '%{y:.0f}'
                            )
        trace_4ds = go.Scatter(y = y_pred.loc[valid_data.index,'R'],
                              x = valid_data.index,
                              mode = 'lines',
                              line = dict(color = 'blue', width = styling['linewidth'], dash = 'dash'),
                              name = 'Reproduction number',
                              opacity = styling['model_opacity'],
                              hovertemplate = '%{y:.3f}'
                             
                            )
        fig.add_trace(trace_4c, row = 1, col = 2)
        fig.add_trace(trace_4d, row = 1, col = 2)
        fig.add_trace(trace_4ds, row = 1, col = 2, secondary_y=True)
 
 
        trace_5c = go.Scatter(y = valid_data['CurrentInfections'],
                              x = valid_data.index,
                              mode = 'lines',
                              line = dict(color = 'green', width = styling['linewidth'], dash = 'dash'),
                              name = 'Actual current infections',
                              opacity = styling['actual_opacity'],
                              hovertemplate = '%{y:.0f}'
                            )
        trace_5d = go.Scatter(y = y_pred.loc[valid_data.index,'CurrentInfections'],
                              x = valid_data.index,
                              mode = 'lines',
                              line = dict(color = 'red', width = styling['linewidth'], dash = 'dash'),
                              name = 'Model current infections(validation)',
                              opacity = styling['model_opacity'],
                              hovertemplate = '%{y:.0f}'
                            )
 
        fig.add_trace(trace_5c, row = 4, col = 2)
        fig.add_trace(trace_5d, row = 4, col = 2)
       
    else:
       
        if test_index is not None:
            trace_1e = go.Scatter(y = y_pred.loc[test_index,'ConfirmedCases'],
                              x = test_index,
                              mode = 'lines',
                              line = dict(color = 'red', width = styling['linewidth'], dash = 'dot'),
                              name = 'Modeled confirmed cases (forecast)',
                              opacity = styling['model_opacity'],
                              hovertemplate = '%{y:.0f}'
                            )
            trace_1es = go.Scatter(y = y_pred.loc[test_index,'R'],
                              x = test_index,
                              mode = 'lines',
                              line = dict(color = 'blue', width = styling['linewidth']),
                              name = 'Reproduction number',
                              opacity = styling['model_opacity'],
                              hovertemplate = '%{y:.3f}'
                             
                            )
            fig.add_trace(trace_1e, row = 1, col = 1)
            fig.add_trace(trace_1es, row = 1, col = 1, secondary_y=True)
           
            trace_2e = go.Scatter(y = y_pred.loc[test_index,'CurrentInfections'],
                              x = test_index,
                              mode = 'lines',
                              line = dict(color = 'red', width = styling['linewidth'], dash = 'dot'),
                              name = 'Modeled Current Infections (forecast)',
                              opacity = styling['model_opacity'],
                              hovertemplate = '%{y:.0f}'
                            )
            fig.add_trace(trace_2e, row = 3, col = 1)
           
            trace_3e = go.Scatter(y = y_pred.loc[test_index,'Removed'],
                              x = test_index,
                              mode = 'lines',
                              line = dict(color = 'red', width = styling['linewidth'], dash = 'dot'),
                              name = 'Modeled Removed(Dead + Recovered) (forecast)',
                              opacity = styling['model_opacity'],
                              hovertemplate = '%{y:.0f}'
                            )
            fig.add_trace(trace_3e, row = 5, col = 1)
    
    """fig.update_xaxes(fixedrange=True, row = 1, col = 1)
    fig.update_xaxes(fixedrange=True, row = 3, col = 1)
    fig.update_xaxes(fixedrange=True, row = 5, col = 1)
    fig.update_xaxes(fixedrange=True, row = 1, col = 2)
    fig.update_xaxes(fixedrange=True, row = 4, col = 2)

    fig.update_yaxes(fixedrange=True, row = 1, col = 1)
    fig.update_yaxes(fixedrange=True, row = 3, col = 1)
    fig.update_yaxes(fixedrange=True, row = 5, col = 1)
    fig.update_yaxes(fixedrange=True, row = 1, col = 2)
    fig.update_yaxes(fixedrange=True, row = 4, col = 2)    """
       
    
    fig.show()
   
    return fig
 
def fit_model_public(area_name
                     , initial_guess=[1.5, 2.9, 5.2, 2, 50, 0.5] #R_0, t_inc, t_inf, k, L , pop_frac
                     , bounds=((1, 4), # R bounds
                             (2.8, 3.0), (5, 6),
                              (1, 5), (1, 100), (0, 1)) # fraction time param bounds
                     , make_plot=True, extra_days = 0):
       
    train_data = train.loc[area_name].query('ConfirmedCases > 0')
    valid_data = valid.loc[area_name]
    test_data = test_public.loc[area_name]
    try:
        population = abb_pop_dict[area_name]
    except KeyError:
        print("Key not found")
        return
       
    cases_per_million = train_data['ConfirmedCases'].max() * 10**6 / population
    n_infected = train_data['ConfirmedCases'].iloc[0]
       
    if cases_per_million < 1:
        return use_last_value(train_data, valid_data, test_data)
   
    print("for res_decay")
    res_decay = minimize(eval_model_decay, initial_guess[:-1], bounds=bounds[:-1],
                         args=(train_data, population, False),
                         method='L-BFGS-B')
#    print("for res_calc")
#    res_calc = minimize(eval_model_calc, initial_guess[1:-3], bounds=bounds[1:-3],
#                         args=(train_data, population, False),
#                         method='L-BFGS-B')
  
    dates_all = train_data.index.append(test_data.index)
    dates_all = dates_all.append(pd.date_range(start = dates_all[-1], freq ='D',  periods = extra_days))
    dates_val = train_data.index.append(valid_data.index)
   
    print("Decaying R_0 used")
    msle, sol = eval_model_decay(res_decay.x, train_data, population, True, len(test_data), extra_days)
    res = res_decay
 
    # Calculate the R_t values
    #t = np.arange(len(dates_val))
    t = np.arange(len(dates_all))
    R_0, t_inc, t_inf, k, L  = res.x
    R_t = pd.Series(R_0 / (1 + (t/L)**k), dates_all)
     
    sus, exp, inf, rec = sol.y
   
    y_pred = pd.DataFrame({
        'ConfirmedCases': np.clip(inf + rec, 0, np.inf) * population,
        'Removed': np.clip(rec, 0, np.inf) * population,
        'CurrentInfections': np.clip(inf, 0, np.inf) * population,
        'R': R_t,
    }, index=dates_all)   
    y_pred_valid = y_pred.iloc[len(train_data): len(train_data)+len(valid_data)]
    y_pred_test = y_pred.iloc[len(train_data):(-extra_days)]
    y_true_valid = valid_data[['ConfirmedCases', 'Removed', 'CurrentInfections']]
       
    valid_msle_cases = mean_squared_log_error(y_true_valid['ConfirmedCases'], y_pred_valid['ConfirmedCases'])
    valid_msle_recov = mean_squared_log_error(y_true_valid['Removed'], y_pred_valid['Removed'])
    valid_msle_inf = mean_squared_log_error(y_true_valid['CurrentInfections'], y_pred_valid['CurrentInfections'])
    valid_msle = np.average([valid_msle_cases, valid_msle_recov, valid_msle_inf], weights = [0.4, 0.1, 0.5])
    if make_plot:
        print(f'State: {abb_dict[area_name]}')
        print(f'Validation MSLE: {valid_msle:0.5f}')
        print(f'R: {res.x[0]:0.3f}, t_inc: {res.x[1]:0.3f}, t_inf: {res.x[2]:0.3f}, '
                  f'k: {res.x[3]:0.3f}, L: {res.x[4]:0.3f}')

        fig = plot_model_results_plotly(y_pred, train_data, area_name, valid_data, res , valid_msle)
       
    # Put the forecast in the submission
    forecast_ids = test_data['ForecastId']
    submission.loc[forecast_ids, ['ConfirmedCases', 'Removed']] = y_pred_test[['ConfirmedCases', 'Removed']].values.copy()
    return fig
 
def fit_model_private(area_name,
                     initial_guess=[1.5, 2.9, 5.2, 2, 50, 0.5], #R_0, t_inc, t_inf, k, L , pop_frac
                      bounds=((2, 5), # R bounds
                             (0.5, 10), (2, 20),
                              (1, 5), (1, 100), (0, 1)), # fraction time param bounds
                     make_plot=True, extra_days = 0,
                     valid_days= 10, forecast_days=30):
   
    train_full, train, valid, test, test_public, test_private, submission = prepare_data(valid_days, forecast_days)
    train_data = train_full.loc[area_name].query('ConfirmedCases > 0')
    global train_data_out
    train_data_out = train_data
    test_data = test_private.loc[area_name]
    global test_data_out
    test_data_out = test_data
    try:
        population = abb_pop_dict[area_name]
    except KeyError:
        print("Key not found")
        return
       
    cases_per_million = train_data['ConfirmedCases'].max() * 10**6 / population
    n_infected = train_data['ConfirmedCases'].iloc[0]
       
    if cases_per_million < 1:
        return use_last_value(train_data, test_data)
   
    print("for res_decay")
    res_decay = minimize(eval_model_decay, initial_guess[:-1], bounds=bounds[:-1],
                         args=(train_data, population, False),
                         method='L-BFGS-B')

    dates_all = train_data.index.append(test_data.index)
    dates_all = dates_all.append(pd.date_range(start = dates_all[-1], freq ='D',  periods = extra_days))
   
    print("Decaying R_0 used")
    msle, sol = eval_model_decay(res_decay.x, train_data, population, True, len(test_data), extra_days)
    res = res_decay
    print("res_decay calculation complete")
 
    # Calculate the R_t values
    t= np.arange(len(dates_all))
    R_0, t_inc, t_inf, k, L  = res.x
    R_t = pd.Series(R_0 / (1 + (t/L)**k), dates_all)  
   
    sus, exp, inf, rec = sol.y
    y_pred = pd.DataFrame({
        'ConfirmedCases': np.clip(inf + rec, 0, np.inf) * population,
        'Removed': np.clip(rec, 0, np.inf) * population,
        'CurrentInfections': np.clip(inf, 0, np.inf) * population,
        'R': R_t,
    }, index = dates_all)
   
    y_pred_test = y_pred.iloc[len(train_data):(-extra_days)]
    global y_pred_test_out
    y_pred_test_out = y_pred_test
    y_pred_test_index = y_pred_test.index
   
    if make_plot:
        print(f'State: {abb_dict[area_name]}')
        print(f'R: {res.x[0]:0.3f}, t_inc: {res.x[1]:0.3f}, t_inf: {res.x[2]:0.3f}, '
                  f'k: {res.x[3]:0.3f}, L: {res.x[4]:0.3f}')
        
        #fig = plot_model_results_plotly(y_pred, train_data, area_name, res = res , test_index = y_pred_test_index)
        fig = plot_model_results_plotly(y_pred = y_pred, train_data = train_data, state = area_name, valid_data=None, res = res, valid_mlse = None, test_index = y_pred_test_index)
       
    # Put the forecast in the submission
    #forecast_ids = test_data['ForecastId']
    #submission.loc[forecast_ids, ['ConfirmedCases', 'Removed']] = y_pred_test[['ConfirmedCases', 'Removed']].values.copy()
    return fig


app =  dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], external_scripts=[
  'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML'])
app.title = 'COVID-19 Predictions for India'
server = app.server
 
options = []
state_keys = list(abb_dict.keys())
state_keys.sort()
state_keys_mod = ['TT']
for key in state_keys:
    if key != 'TT':
        state_keys_mod.append(key)
for key in state_keys_mod:
    options.append({'label': abb_dict[key], 'value': key})
   
app.layout = html.Div([
                        html.Div([
                            html.Div([
                                dcc.Markdown('''
                                             ### ** COVID-19 Predictions -India **
                                             ###### Parameters of the SEIR model:
                                            - *R_0* : Basic Reproduction Number  
                                            - *t_inc* : Incubation period (average)  
                                            - *t_inf* : Infectious period (average)  
                                                 
                                            - __*R(t)* is assumed to decay according to the following model:  *R(t)* = *R_0*/(1+(t/L)^k)__
                                            - The <span style = "color:green">GREEN </span> lines denote the actual numbers, the RED lines show the predicted
                                            - The BLUE Line represents the Reproduction number
                                            
                                             ** Hover on the graphs for more info **
                                             
                                             ''', dangerously_allow_html=True
                                    )], style ={'width':'70%'}),
                                 
                                ] ) ,
            dcc.Markdown('''##### Please select any State/UT of India from the following dropdown :'''),
            html.Div([dcc.Dropdown(id = 'state-in-dropdown',
                options = options,
                value='DL',
                style = {'color':'black'},
                    )
                      ]),
            
            dcc.Markdown('''##### Enter the number of days you want to forecast (Press enter to confirm):'''),
            html.Div([dcc.Input(id = 'forecast_days',
                                placeholder='Enter the number of days you want to forecast...',
                                type='number',
                                value=10,
                                #style = {'fontSize': 30, 'width':'30%'},
                                debounce = True
                            )] ),
            html.Div([dcc.Loading(
                                id="loading-1",
                                type="default",
                                fullscreen = "True",
                                children=dcc.Graph(id = 'corona-line', figure = {}),
                                #loading_state = {'is_loading':True,
                                #                 'component_name': 'corona-line'}
                    )          
        ])
])

@app.callback(Output("loading-output-1", "children"), [Input("state-in-dropdown", "value")])
def input_triggers_spinner(value):
    time.sleep(1)
    return None    
    
@app.callback(Output('corona-line', 'figure'),
             [Input('state-in-dropdown', 'value'), Input('forecast_days', 'value')])
def output(state, forecast_days):
    fig = fit_model_private(str(state), extra_days = 500, forecast_days= forecast_days)
    #fig.update_annotations({'font':{'size': 24}})
    fig.update_layout({
                    #'font':{'size':24},
                       #'width':2400,
                       'height': 900
                       },
                     dragmode = False,
                     hovermode = "x",
                     hoverlabel = {'namelength':50},
                     yaxis2 = dict(
                                    title="Reproduction number",
                                    titlefont=dict(
                                        color="blue"
                                    ),
                                    tickfont=dict(
                                        color="blue"
                                    ),
                                    anchor="x",
                                    overlaying="y",
                                    side="right"
                                ) ,
                    yaxis4=dict(
                                    title="Reproduction number",
                                    titlefont=dict(
                                        color="blue"
                                    ),
                                    tickfont=dict(
                                        color="blue"
                                    ),
                                    anchor="free",
                                    overlaying="y3",
                                    side="right",
                                    position=0.94
                                ) )
    return fig
if __name__=='__main__':
    app.run_server()           