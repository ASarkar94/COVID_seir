import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from scipy import stats as sps
from scipy.interpolate import interp1d

from IPython.display import clear_output

FILTERED_REGIONS = []

FILTERED_REGION_CODES = []

 

#%config InlineBackend.figure_format = 'retina'

 

# Column vector of k

k = np.arange(0, 70)[:, None]

 

# Different values of Lambda

lambdas = [10, 20, 30, 40]

 

# Evaluated the Probability Mass Function (remember: poisson is discrete)

y = sps.poisson.pmf(k, lambdas)

 

# Show the resulting shape

print(y.shape)

 

fig, ax = plt.subplots()

 

ax.set(title='Poisson Distribution of Cases\n $p(k|\lambda)$')

 

plt.plot(k, y,

         marker='o',

         markersize=3,

         lw=0)

 

plt.legend(title="$\lambda$", labels=lambdas);

 

k = 20

 

lam = np.linspace(1, 45, 90)

 

likelihood = pd.Series(data=sps.poisson.pmf(k, lam),

                       index=pd.Index(lam, name='$\lambda$'),

                       name='lambda')

 

likelihood.plot(title=r'Likelihood $P\left(k_t=20|\lambda\right)$');

 

k = 20

 

lam = np.linspace(1, 45, 90)

 

likelihood = pd.Series(data=sps.poisson.pmf(k, lam),

                       index=pd.Index(lam, name='$\lambda$'),

                       name='lambda')

 

likelihood.plot(title=r'Likelihood $P\left(k_t=20|\lambda\right)$');

 

sigmas = np.linspace(1/20, 1, 20)

k = np.array([20, 40, 55, 90])

 

# We create an array for every possible value of Rt

R_T_MAX = 12

r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)

 

# Gamma is 1/serial interval

# https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article

# https://www.nejm.org/doi/full/10.1056/NEJMoa2001316

GAMMA = 1/7

 

# Map Rt into lambda so we can substitute it into the equation below

# Note that we have N-1 lambdas because on the first day of an outbreak

# you do not know what to expect.

lam = k[:-1] * np.exp(GAMMA * (r_t_range[:, None] - 1))

 

# Evaluate the likelihood on each day and normalize sum of each day to 1.0

likelihood_r_t = sps.poisson.pmf(k[1:], lam)

likelihood_r_t /= np.sum(likelihood_r_t, axis=0)

 

# Plot it

ax = pd.DataFrame(

    data = likelihood_r_t,

    index = r_t_range

).plot(

    title='Likelihood of $R_t$ given $k$',

    xlim=(0,10),

    figsize=(6, 2.5)

)

 

ax.legend(labels=k[1:], title='New Cases')

ax.set_xlabel('$R_t$');

 

posteriors = likelihood_r_t.cumprod(axis=1)

posteriors = posteriors / np.sum(posteriors, axis=0)

 

columns = pd.Index(range(1, posteriors.shape[1]+1), name='Day')

posteriors = pd.DataFrame(

    data = posteriors,

    index = r_t_range,

    columns = columns)

 

ax = posteriors.plot(

    title='Posterior $P(R_t|k)$',

    xlim=(0,10),

    figsize=(6,2.5)

)

ax.legend(title='Day')

ax.set_xlabel('$R_t$');

 

most_likely_values = posteriors.idxmax(axis=0)

 

def highest_density_interval(pmf, p=.9, debug=False):

    # If we pass a DataFrame, just call this recursively on the columns

    if(isinstance(pmf, pd.DataFrame)):

        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],

                            index=pmf.columns)

   

    cumsum = np.cumsum(pmf.values)

   

    # N x N matrix of total probability mass for each low, high

    total_p = cumsum - cumsum[:, None]

   

    # Return all indices with total_p > p

    lows, highs = (total_p > p).nonzero()

   

    # Find the smallest range (highest density)

    best = (highs - lows).argmin()

   

    low = pmf.index[lows[best]]

    high = pmf.index[highs[best]]

   

    return pd.Series([low, high],

                     index=[f'Low_{p*100:.0f}',

                            f'High_{p*100:.0f}'])

 

hdi = highest_density_interval(posteriors, debug=True)

hdi.tail()

 

ax = most_likely_values.plot(marker='o',

                             label='Most Likely',

                             title=f'$R_t$ by day',

                             c='k',

                             markersize=4)

 

ax.fill_between(hdi.index,

                hdi['Low_90'],

                hdi['High_90'],

                color='k',

                alpha=.1,

                lw=0,

                label='HDI')

 

ax.legend();

 

dfi = pd.read_csv("http://api.covid19india.org/states_daily_csv/confirmed.csv")

dfi = dfi.drop(columns=['Unnamed: 40'], axis=1)

cols = dfi.columns[1:]

dfi.loc[:, cols] = dfi.loc[:, cols].cumsum(axis=0)

 

nstates = dfi.columns[1:].tolist()

 

dfa = pd.DataFrame()

 

for i, state in enumerate(nstates):

    dfc = dfi[['date', state]].copy()

    dfc['state'] = state

    dfc = dfc.rename({state: 'cases'}, axis=1)

    dfc['date'] = pd.to_datetime(dfc['date'])

    # dfc = dfc[dfc.cases > 5].copy()

    dfa = dfa.append(dfc)

    #if (len(dfc) > 2):

    #  dfa = dfa.append(dfc)

    #else:

    #  print(f"Excluding state {state}")

dfa['date'] = pd.to_datetime(dfa['date'])

states = dfa.set_index(['state', 'date']).squeeze()

 

def prepare_cases(cases, cutoff=25):

    new_cases = cases.diff()

 

    smoothed = new_cases.rolling(7,

        win_type='gaussian',

        min_periods=1,

        center=True).mean(std=2).round()

   

    idx_start = np.searchsorted(smoothed, cutoff)

   

    smoothed = smoothed.iloc[idx_start:]

    original = new_cases.loc[smoothed.index]

   

    return original, smoothed

 

 

def get_posteriors(sr, sigma=0.15):

 

    # (1) Calculate Lambda

    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

 

   

    # (2) Calculate each day's likelihood

    likelihoods = pd.DataFrame(

        data = sps.poisson.pmf(sr[1:].values, lam),

        index = r_t_range,

        columns = sr.index[1:])

   

    # (3) Create the Gaussian Matrix

    process_matrix = sps.norm(loc=r_t_range,

                              scale=sigma

                             ).pdf(r_t_range[:, None])

 

    # (3a) Normalize all rows to sum to 1

    process_matrix /= process_matrix.sum(axis=0)

   

    # (4) Calculate the initial prior

    #prior0 = sps.gamma(a=4).pdf(r_t_range)

    prior0 = np.ones_like(r_t_range)/len(r_t_range)

    prior0 /= prior0.sum()

 

    # Create a DataFrame that will hold our posteriors for each day

    # Insert our prior as the first posterior.

    posteriors = pd.DataFrame(

        index=r_t_range,

        columns=sr.index,

        data={sr.index[0]: prior0}

    )

   

    # We said we'd keep track of the sum of the log of the probability

    # of the data for maximum likelihood calculation.

    log_likelihood = 0.0

 

    # (5) Iteratively apply Bayes' rule

    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

 

        #(5a) Calculate the new prior

        current_prior = process_matrix @ posteriors[previous_day]

       

        #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)

        numerator = likelihoods[current_day] * current_prior

       

        #(5c) Calcluate the denominator of Bayes' Rule P(k)

        denominator = np.sum(numerator)

       

        # Execute full Bayes' Rule

        posteriors[current_day] = numerator/denominator

       

        # Add to the running sum of log likelihoods

        log_likelihood += np.log(denominator)

   

    return posteriors, log_likelihood

 

sigmas = np.linspace(1/20, 1, 20)

 

targets = ~states.index.get_level_values('state').isin(FILTERED_REGION_CODES)

states_to_process = states.loc[targets]

 

results = {}

failed_states = []

 

for state_name, cases in states_to_process.groupby(level='state'):

    

    print(state_name)

   

    # Only difference with KS code

    new, smoothed = prepare_cases(cases, cutoff=1)

   

    # KS uses cutoff of 25 followed by 10

    #new, smoothed = prepare_cases(cases, cutoff=25)

       

    #if len(smoothed) == 0:

    #    new, smoothed = prepare_cases(cases, cutoff=10)

   

    result = {}

   

    # Holds all posteriors with every given value of sigma

    result['posteriors'] = []

   

    # Holds the log likelihood across all k for each value of sigma

    result['log_likelihoods'] = []

   

    try:

        for sigma in sigmas:

            posteriors, log_likelihood = get_posteriors(smoothed, sigma=sigma)

            result['posteriors'].append(posteriors)

            result['log_likelihoods'].append(log_likelihood)

   

    # Store all results keyed off of state name

        results[state_name] = result

        clear_output(wait=True)

    except:

        print(f"Error for state {state_name}")

        failed_states.append(state_name)

 

print('Done.')

 

# Each index of this array holds the total of the log likelihoods for

# the corresponding index of the sigmas array.

total_log_likelihoods = np.zeros_like(sigmas)

 

# Loop through each state's results and add the log likelihoods to the running total.

for state_name, result in results.items():

    total_log_likelihoods += result['log_likelihoods']

 

# Select the index with the largest log likelihood total

max_likelihood_index = total_log_likelihoods.argmax()

 

# Select the value that has the highest log likelihood

sigma = sigmas[max_likelihood_index]

 

# Plot it

fig, ax = plt.subplots()

ax.set_title(f"Maximum Likelihood value for $\sigma$ = {sigma:.2f}");

ax.plot(sigmas, total_log_likelihoods)

ax.axvline(sigma, color='k', linestyle=":")

 

final_results = None

hdi_error_list = []

 

for state_name, result in results.items():

    print(state_name)

    try:

        posteriors = result['posteriors'][max_likelihood_index]

        hdis_90 = highest_density_interval(posteriors, p=.9)

        hdis_50 = highest_density_interval(posteriors, p=.5)

        most_likely = posteriors.idxmax().rename('ML')

        result = pd.concat([most_likely, hdis_90, hdis_50], axis=1)

        if final_results is None:

            final_results = result

        else:

            final_results = pd.concat([final_results, result])

        clear_output(wait=True)

    except:

        print(f'hdi failed for {state_name}')

        hdi_error_list.append(state_name)

        pass

 

print(f'HDI error list: {hdi_error_list}')

print('Done.')

 

def plot_rt(result, ax, state_name):

   

    ax.set_title(f"{state_name}")

   

    # Colors

    ABOVE = [1,0,0]

    MIDDLE = [1,1,1]

    BELOW = [0,0,0]

    cmap = ListedColormap(np.r_[

        np.linspace(BELOW,MIDDLE,25),

        np.linspace(MIDDLE,ABOVE,25)

    ])

    color_mapped = lambda y: np.clip(y, .5, 1.5)-.5

   

    index = result['ML'].index.get_level_values('date')

    values = result['ML'].values

   

    # Plot dots and line

    ax.plot(index, values, c='k', zorder=1, alpha=.25)

    ax.scatter(index,

               values,

               s=40,

               lw=.5,

               c=cmap(color_mapped(values)),

               edgecolors='k', zorder=2)

   

    # Aesthetically, extrapolate credible interval by 1 day either side

    lowfn = interp1d(date2num(index),

                     result['Low_90'].values,

                     bounds_error=False,

                     fill_value='extrapolate')

   

    highfn = interp1d(date2num(index),

                      result['High_90'].values,

                      bounds_error=False,

                      fill_value='extrapolate')

   

    extended = pd.date_range(start=pd.Timestamp('2020-03-01'),

                             end=index[-1]+pd.Timedelta(days=1))

   

    ax.fill_between(extended,

                    lowfn(date2num(extended)),

                    highfn(date2num(extended)),

                    color='k',

                    alpha=.1,

                    lw=0,

                    zorder=3)

 

    ax.axhline(1.0, c='k', lw=1, label='$R_t=1.0$', alpha=.25);

   

    # Formatting

    ax.xaxis.set_major_locator(mdates.MonthLocator())

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    ax.xaxis.set_minor_locator(mdates.DayLocator())

   

    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))

    ax.yaxis.tick_right()

    ax.spines['left'].set_visible(False)

    ax.spines['bottom'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.margins(0)

    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)

    ax.margins(0)

    ax.set_ylim(0.0, 5.0)

    ax.set_xlim(pd.Timestamp('2020-03-01'), result.index.get_level_values('date')[-1]+pd.Timedelta(days=1))

    fig.set_facecolor('w')

 

def plot_rt_new(result, ax, state_name):

   

    ax.set_title(f"{state_name}")

   

    # Colors

    ABOVE = [1,0,0]

    MIDDLE = [1,1,1]

    BELOW = [0,0,0]

    cmap = ListedColormap(np.r_[

        np.linspace(BELOW,MIDDLE,25),

        np.linspace(MIDDLE,ABOVE,25)

    ])

    color_mapped = lambda y: np.clip(y, .5, 1.5)-.5

   

    index = result['ML'].index.get_level_values('date')

    values = result['ML'].values

   

    # Plot dots and line

    ax.plot(index, values, c='k', zorder=1, alpha=.25)

    ax.scatter(index,

               values,

               s=40,

               lw=.5,

               c=cmap(color_mapped(values)),

               edgecolors='k', zorder=2)

   

    # Aesthetically, extrapolate credible interval by 1 day either side

    lowfn = interp1d(date2num(index),

                     result['Low_90'].values,

                     bounds_error=False,

                     fill_value='extrapolate')

   

    highfn = interp1d(date2num(index),

                      result['High_90'].values,

                      bounds_error=False,

                      fill_value='extrapolate')

   

    extended = pd.date_range(start=pd.Timestamp('2020-03-01'),

                             end=index[-1]+pd.Timedelta(days=1))

   

    ax.fill_between(extended,

                    lowfn(date2num(extended)),

                    highfn(date2num(extended)),

                    color='k',

                    alpha=.1,

                    lw=0,

                    zorder=3)

 

    ax.axhline(1.0, c='k', lw=1, label='$R_t=1.0$', alpha=.25);

   

    ax.axvline(x=pd.Timestamp('2020-03-24'), ls='--', c='k', alpha=.25)

    ax.axvline(x=pd.Timestamp('2020-04-14'), ls='--', c='k', alpha=.25)

    ax.axvline(x=pd.Timestamp('2020-05-03'), ls='--', c='k', alpha=.25)

   

    ax.text(pd.Timestamp('2020-03-22'), 0.2, 'March 24', bbox=dict(facecolor='white', alpha=0.5))

    ax.text(pd.Timestamp('2020-04-12'), 0.2, 'April 14', bbox=dict(facecolor='white', alpha=0.5))

    ax.text(pd.Timestamp('2020-05-02'), 0.2, 'May 3', bbox=dict(facecolor='white', alpha=0.5))

   

    result2 = result.reset_index()

    t1 = pd.Timestamp('2020-03-24')

    t2 = pd.Timestamp('2020-04-14')

    t3 = pd.Timestamp('2020-05-03')

 

    r1m = result2[result2.state==state_name][result2.date <= t1]['ML'].mean()

    r1l = result2[result2.state==state_name][result2.date <= t1]['Low_90'].mean()

    r1h = result2[result2.state==state_name][result2.date <= t1]['High_90'].mean()

    text1 = f"Pre-lockdown:\n {r1m:.2f} [{r1l:.2f}-{r1h:.2f}]"

 

    r2m = result2[result2.state==state_name][(result2.date > t1) & (result2.date <= t2)]['ML'].mean()

    r2l = result2[result2.state==state_name][(result2.date > t1) & (result2.date <= t2)]['Low_90'].mean()

    r2h = result2[result2.state==state_name][(result2.date > t1) & (result2.date <= t2)]['High_90'].mean()

    text2 = f"Initial Lockdown:\n {r2m:.2f} [{r2l:.2f}-{r2h:.2f}]"

 

    r3m = result2[result2.state==state_name][(result2.date > t2) & (result2.date <= t3)]['ML'].mean()

    r3l = result2[result2.state==state_name][(result2.date > t2) & (result2.date <= t3)]['Low_90'].mean()

    r3h = result2[result2.state==state_name][(result2.date > t2) & (result2.date <= t3)]['High_90'].mean()

    text3 = f"First Extension:\n {r3m:.2f} [{r3l:.2f}-{r3h:.2f}]"

 

    r4m = result2[result2.state==state_name][result2.date > t3]['ML'].mean()

    r4l = result2[result2.state==state_name][result2.date > t3]['Low_90'].mean()

    r4h = result2[result2.state==state_name][result2.date > t3]['High_90'].mean()

    text4 = f"Second Extension:\n {r4m:.2f} [{r4l:.2f}-{r4h:.2f}]"

   

    if r1m > 0:

        ax.text(pd.Timestamp('2020-03-08'), 3.5, text1, bbox=dict(facecolor='white', alpha=0.5))

    if r2m > 0:

        ax.text(pd.Timestamp('2020-03-30'), 3.5, text2, bbox=dict(facecolor='white', alpha=0.5))

    if r3m > 0:

        ax.text(pd.Timestamp('2020-04-20'), 3.5, text3, bbox=dict(facecolor='white', alpha=0.5))

    if r4m > 0:

        ax.text(pd.Timestamp('2020-05-04'), 3.5, text4, bbox=dict(facecolor='white', alpha=0.5))

   

    # Formatting

    ax.xaxis.set_major_locator(mdates.MonthLocator())

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    ax.xaxis.set_minor_locator(mdates.DayLocator())

   

    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))

    ax.yaxis.tick_right()

    ax.spines['left'].set_visible(False)

    ax.spines['bottom'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.margins(0)

    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)

    ax.margins(0)

    ax.set_ylim(0.0, 5.0)

    ax.set_xlim(pd.Timestamp('2020-03-01'), result.index.get_level_values('date')[-1]+pd.Timedelta(days=1))

    fig.set_facecolor('w')

def plot_rt_top6(final_results, ax, state_list):

   

    ax.set_title(f"All India & Top States")

   

    # Colors

    ABOVE = [1,0,0]

    MIDDLE = [1,1,1]

    BELOW = [0,0,0]

    cmap = ListedColormap(np.r_[

        np.linspace(BELOW,MIDDLE,25),

        np.linspace(MIDDLE,ABOVE,25)

    ])

    color_mapped = lambda y: np.clip(y, .5, 1.5)-.5

   

    for i, (state_name, result) in enumerate(final_results.groupby('state')):

        if (state_name not in state_list):

            continue

   

        index = result['ML'].index.get_level_values('date')

        values = result['ML'].values

 

        # Plot dots and line

        ax.plot(index, values, c='k', zorder=1, alpha=.25)

        ax.scatter(index,

                   values,

                   s=40,

                   lw=.5,

                   label=state_name)

       

    ax.legend()

    ax.axhline(1.0, c='k', lw=1, label='$R_t=1.0$', alpha=.25);

   

    ax.axvline(x=pd.Timestamp('2020-03-24'), ls='--', c='k', alpha=.25)

    ax.axvline(x=pd.Timestamp('2020-04-14'), ls='--', c='k', alpha=.25)

    ax.axvline(x=pd.Timestamp('2020-05-03'), ls='--', c='k', alpha=.25)

   

    ax.text(pd.Timestamp('2020-03-22'), 0.2, 'March 24', bbox=dict(facecolor='white', alpha=0.5))

    ax.text(pd.Timestamp('2020-04-12'), 0.2, 'April 14', bbox=dict(facecolor='white', alpha=0.5))

    ax.text(pd.Timestamp('2020-05-02'), 0.2, 'May 3', bbox=dict(facecolor='white', alpha=0.5))

   

    result2 = result.reset_index()

    text1 = f"Pre-lockdown:"

    text2 = f"Initial Lockdown:"

    text3 = f"First Extension:"

    text4 = f"Second Extension:"

 

    ax.text(pd.Timestamp('2020-03-08'), 3.5, text1, bbox=dict(facecolor='white', alpha=0.5))

    ax.text(pd.Timestamp('2020-03-30'), 3.5, text2, bbox=dict(facecolor='white', alpha=0.5))

    ax.text(pd.Timestamp('2020-04-20'), 3.5, text3, bbox=dict(facecolor='white', alpha=0.5))

    ax.text(pd.Timestamp('2020-05-04'), 3.5, text4, bbox=dict(facecolor='white', alpha=0.5))

   

    # Formatting

    ax.xaxis.set_major_locator(mdates.MonthLocator())

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    ax.xaxis.set_minor_locator(mdates.DayLocator())

   

    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))

    ax.yaxis.tick_right()

    ax.spines['left'].set_visible(False)

    ax.spines['bottom'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.margins(0)

    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)

    ax.margins(0)

    ax.set_ylim(0.0, 5.0)

    ax.set_xlim(pd.Timestamp('2020-03-01'), result.index.get_level_values('date')[-1]+pd.Timedelta(days=1))

    fig.set_facecolor('w')

 

final_results.to_csv('r_t.csv')
print("CSV file is ready")
