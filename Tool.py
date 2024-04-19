import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(0)

student_nr = "s5792010"
data_path = "C:/Users/elvel/Desktop/RUG/COURSES/AM/Assignment-3/"


def data_preparation(machine_data):
    # Calculate the "Duration"
    machine_data['Duration'] = machine_data['Time'].diff().abs()

    # correct 1st line for duration
    machine_data.loc[0, 'Duration'] = machine_data.loc[0, 'Time']
    # Sort by "Duration".
    machine_data_sorted = machine_data.sort_values(by='Duration', ignore_index=True)

    # Round to two decimals.
    machine_data_sorted = machine_data_sorted.round(2)

    return machine_data_sorted


def create_kaplanmeier_data(prepared_data):
    prepared_data['Sort_Priority'] = prepared_data['Event'].apply(lambda x: 0 if x == 'failure' else 1)
    # Sort by Duration and then by Event to prioritize event durations
    prepared_data = prepared_data.sort_values(by=['Duration', "Sort_Priority"])

    # Assign equal probability to all durations
    row_counter = len(prepared_data)
    prepared_data['Probability'] = 1.0 / row_counter

    # Adjust probabilities for censored durations (PM)
    for i in range(row_counter):
        if prepared_data.iloc[i]['Event'] == 'PM':
            nr_rows_after = row_counter - i - 1
            divided_probability = prepared_data.loc[i, "Probability"] / nr_rows_after

            if nr_rows_after > 0:
                prepared_data.iloc[i + 1:,
                prepared_data.columns.get_loc('Probability')] += divided_probability

    # Set Probability of censored durations to 0 after distribution
    prepared_data.loc[prepared_data['Event'] == 'PM', 'Probability'] = 0

    # Calculate Reliability
    prepared_data['Reliability'] = 1 - prepared_data['Probability'].cumsum()
    # To ensure the last reliability value is exactly 0 for the last failure event
    if prepared_data.iloc[-1]['Event'] == 'failure':
        prepared_data.iloc[-1, prepared_data.columns.get_loc('Reliability')] = 0

    return prepared_data


def visualization(KM_data, weibull_data, machine_name):
    # Ensuring the data is sorted by Duration for plotting
    KM_data = KM_data.sort_values(by='Duration')
    weibull_data = weibull_data.sort_values(by='t')

    # Plotting the Kaplan-Meier reliability function
    plt.figure(figsize=(10, 6))
    plt.step(KM_data['Duration'], KM_data['Reliability'], where='post', label='Kaplan-Meier')
    plt.step(weibull_data['t'], weibull_data['R_t'], where='post', label='Weibull')
    # Adding labels and title
    plt.xlabel('Time (t)')
    plt.ylabel('Reliability function R(t)')
    plt.title(f'Reliability function over time - Machine {machine_name}')

    # Adding a grid
    plt.grid(True)

    # Adding a legend
    plt.legend()

    # Display the plot
    plt.show()


def meantimebetweenfailures_KM(KM_data):
    # Filter for failures only
    failures_only = KM_data[KM_data['Event'] == 'failure']

    # Calculate the mean of the 'Duration' column for failures
    MTBF = failures_only['Duration'].mean()

    return MTBF


def fit_weibull_distribution(prepared_data):
    # Define the ranges for lambda and kappa
    lambda_range = np.linspace(1, 35, 35)
    kappa_range = np.linspace(0.1, 3.5, 35)

    # Create an empty DataFrame to store results
    results_df = pd.DataFrame(
        columns=['Lambda', 'Kappa'] + [f'Observation_{i}' for i in range(len(prepared_data))] + ['LogLikelihood_Sum'])

    for l in lambda_range:
        for k in kappa_range:
            log_likelihoods = []

            for index, row in prepared_data.iterrows():
                t = row['Duration']
                if row['Event'] == 'failure':
                    density = (k / l) * (t / l) ** (k - 1) * np.exp(-(t / l) ** k)
                    log_likelihood = np.log(density) if density > 0 else -np.inf
                elif row['Event'] == 'PM':
                    reliability = np.exp(-(t / l) ** k)
                    log_likelihood = np.log(reliability) if reliability > 0 else -np.inf
                else:
                    # Default to -inf for unexpected Event types
                    log_likelihood = -np.inf

                log_likelihoods.append(log_likelihood)

            log_likelihood_sum = np.sum(log_likelihoods)
            results_row = [l, k] + log_likelihoods + [log_likelihood_sum]
            results_df.loc[len(results_df)] = results_row

    # Find the best (lambda, kappa) pair based on the highest log-likelihood sum
    best_row = results_df.loc[results_df['LogLikelihood_Sum'].idxmax()]
    l = best_row['Lambda']
    k = best_row['Kappa']

    return l, k


def meantimebetweenfailures_weibull(l, k):
    # calculate mean time between failure
    MTBF_Weibull = l * math.gamma(1 + 1 / k)

    return MTBF_Weibull


def create_weibull_curve_data(prepared_data, l, k):
    # Determine the range of t values
    max_duration = prepared_data['Duration'].max()
    t_values = np.linspace(0, max_duration, num=int(max_duration * 100) + 1)

    # Calculate R(t) for each t value
    R_t = np.exp(-np.power(t_values / l, k))

    # Create DataFrame with 2 columns
    weibull_data = pd.DataFrame({
        't': t_values,
        'R_t': R_t
    })

    return weibull_data


def create_cost_data(prepared_data, l, k, PM_cost, CM_cost, machine_name):
    # Define F(t) and R(t) for Weibull distribution using 'lambda' for conciseness
    F = lambda t: 1 - np.exp(-(t / l) ** k)
    R = lambda t: np.exp(-(t / l) ** k)

    # Initialize DataFrame
    t_max = prepared_data['Duration'].max()
    ts = np.arange(0.01, t_max, 0.01)

    # Calculate R(t) and F(t) for all t values
    R_ts = R(ts)
    F_ts = F(ts)

    delta_t = 0.01
    mean_cycle_length = np.cumsum(R_ts) * delta_t

    # Create a dataframe to store all results
    df_cost = pd.DataFrame({'t': ts, 'R(t)': R_ts, 'F(t)': F_ts, 'Mean Cycle Length': mean_cycle_length})

    # Calculate Cost per Cycle and Cost rate
    df_cost['Cost per Cycle'] = CM_cost * df_cost['F(t)'] + PM_cost * df_cost['R(t)']
    df_cost['Cost rate(t)'] = df_cost['Cost per Cycle'] / df_cost['Mean Cycle Length']

    # Plotting
    plotting_sample_t = df_cost[df_cost['t'] > 0.1]
    plt.figure(figsize=(10, 6))
    plt.plot(plotting_sample_t['t'], plotting_sample_t['Cost rate(t)'], label='Cost Rate')
    plt.xlabel('Maintenance Age (t)')
    plt.ylabel('Cost Rate (C(t))')
    plt.title(f'Cost Rate for different maintenance ages - Machine {machine_name}')
    plt.grid(True)
    plt.legend()

    plt.show()

    # Calculate optimal age & cost rate using 'idxmin'
    optimal_row = df_cost.loc[df_cost['Cost rate(t)'].idxmin()]
    best_age = optimal_row['t']
    best_cost_rate = optimal_row['Cost rate(t)']

    return best_age, best_cost_rate


def CBM_data_preparation(condition_data):
    # sort by 'Time'.
    condition_data.sort_values('Time', inplace=True)

    # Calculate increments as the difference between subsequent 'Condition' values. first increment should be equal to "condition"
    condition_data['Increments'] = condition_data['Condition'].diff().abs()
    condition_data.loc[0, 'Increments'] = condition_data.loc[0, 'Condition']

    # filter out decreases
    condition_data = condition_data[condition_data['Increments'] >= 0]

    # Assume the initial 'Condition' is 0 or set by the first row if it's the starting condition after maintenance
    condition_data.iloc[0, condition_data.columns.get_loc('Increments')] = condition_data.iloc[0]['Condition']

    return condition_data


def CBM_create_simulations(condition_data, failure_level, threshold):
    # choose the number of simulations
    num_simulations = 5000
    # DataFrame to store simulation results
    simulation_data = pd.DataFrame(columns=['Duration', 'Event'])

    # Loop through each simulation iteration
    for simulation in range(num_simulations):
        # Shuffle the condition data and reset index to maintain consistency
        condition_data_shuffled = condition_data.sample(frac=1).reset_index(drop=True)
        # Initialize condition, time, and index for the simulation
        condition = 0
        time = 0
        index = 0

        # Iterate over the shuffled condition data until all increments have been used
        while True:
            # Check if all increments have been used
            if index >= len(condition_data_shuffled):  # Check if all increments have been used
                break
            # Retrieve the increment value from the shuffled dataset
            increment = condition_data_shuffled.loc[index, 'Increments']
            # Update the condition and time variables
            condition += increment
            time += 1
            index += 1

            # Check for failure or preventive maintenance
            if condition >= failure_level:
                # failure
                new_row = pd.DataFrame({'Duration': [time], 'Event': ['failure']})
                simulation_data = pd.concat([simulation_data, new_row], ignore_index=True)
                break
            elif condition >= threshold:
                # Preventive maintenance
                new_row = pd.DataFrame({'Duration': [time], 'Event': ['PM']})
                simulation_data = pd.concat([simulation_data, new_row], ignore_index=True)
                break

    return simulation_data


def CBM_analyze_costs(simulation_data, PM_cost, CM_cost):
    # Count the number of cycles ending in PM and failure
    num_PM = simulation_data[simulation_data['Event'] == 'PM'].shape[0]
    num_failure = simulation_data[simulation_data['Event'] == 'failure'].shape[0]
    total_cycles = num_PM + num_failure

    # Compute the mean cost per cycle
    mean_cost_per_cycle = (PM_cost * num_PM + CM_cost * num_failure) / total_cycles

    # Compute the average length of simulated cycles
    mean_cycle_length = simulation_data['Duration'].mean()

    # Calculate the cost rate under CBM
    cost_rate = mean_cost_per_cycle / mean_cycle_length

    return cost_rate


def CBM_create_cost_data(prepared_condition_data, PM_cost, CM_cost, failure_level, machine_name):
    # Set the range of thresholds to evaluate
    threshold_start = 0
    threshold_end = int(failure_level)
    threshold_step = 1
    thresholds = np.arange(threshold_start, threshold_end + threshold_step, threshold_step)

    # Initialize DataFrame to store the cost rates for each threshold
    cost_data = pd.DataFrame(columns=['Threshold', 'Cost Rate'])

    # Iterate over each threshold, create simulations, and evaluate the cost rate
    for threshold in thresholds:
        simulation_data = CBM_create_simulations(prepared_condition_data, failure_level, threshold)
        cost_rate = CBM_analyze_costs(simulation_data, PM_cost, CM_cost)
        new_row = {'Threshold': threshold, 'Cost Rate': cost_rate}
        cost_data = pd.concat([cost_data, pd.DataFrame([new_row])], ignore_index=True)

    # Plotting the cost rates for each threshold
    plt.figure(figsize=(10, 6))
    plt.plot(cost_data['Threshold'], cost_data['Cost Rate'], linestyle='-', color='blue')
    # Adding labels and title
    plt.xlabel('Maintenance Threshold (M)')
    plt.ylabel('Cost Rate (C(M))')
    plt.title(f'Cost Rate for different maintenance thresholds - Machine {machine_name}')
    # showing grid
    plt.grid(True)
    # showing the plot
    plt.show()

    # Finding the optimal maintenance threshold and the corresponding cost rate
    optimal_threshold = cost_data.loc[cost_data['Cost Rate'].idxmin(), 'Threshold']
    optimal_cost_rate = cost_data.loc[cost_data['Cost Rate'].idxmin(), 'Cost Rate']

    return optimal_threshold, optimal_cost_rate


def run_analysis():
    # make a list of all machines
    machine_names = [1, 2, 3]
    results_summary = []

    # looping through all machines
    for machine_name in machine_names:
        machine_data = pd.read_csv(f'{data_path}{student_nr}-Machine-{machine_name}.csv')
        cost_data = pd.read_csv(f'{data_path}{student_nr}-Costs.csv').loc[machine_name - 1]

        PM_cost, CM_cost = cost_data.iloc[1], cost_data.iloc[2]
        prepared_data = data_preparation(machine_data)

        KM_data = create_kaplanmeier_data(prepared_data)
        MTBF_KM = meantimebetweenfailures_KM(KM_data)

        l, k = fit_weibull_distribution(prepared_data)
        MTBF_weibull = meantimebetweenfailures_weibull(l, k)
        weibull_data = create_weibull_curve_data(prepared_data, l, k)

        # Visualization for Kaplan-Meier and Weibull data
        visualization(KM_data, weibull_data, machine_name)

        # Calculating cost rate for CM (Corrective Maintenance)
        CM_cost_rate = CM_cost / MTBF_KM

        # Policy evaluation for TBM
        best_age, best_cost_rate_TBM = create_cost_data(prepared_data, l, k, PM_cost, CM_cost, machine_name)
        savings_TBM_vs_CM = CM_cost_rate - best_cost_rate_TBM

        # For machine 3, add CBM analysis
        if machine_name == 3:
            condition_data = pd.read_csv(f'{data_path}{student_nr}-Machine-{machine_name}-condition-data.csv')
            prepared_condition_data = CBM_data_preparation(condition_data)
            failure_level = condition_data['Condition'].max()
            CBM_threshold, CBM_cost_rate = CBM_create_cost_data(prepared_condition_data, PM_cost, CM_cost,
                                                                failure_level, machine_name)
            savings_CBM_vs_CM = CM_cost_rate - CBM_cost_rate
            savings_CBM_vs_TBM = best_cost_rate_TBM - CBM_cost_rate
            optimal_policy = 'CBM'
        else:
            CBM_threshold, CBM_cost_rate, savings_CBM_vs_CM, savings_CBM_vs_TBM = (None, None, None, None)
            optimal_policy = 'TBM' if savings_TBM_vs_CM > 0 else 'CM'

        # saving results in a dictionary
        results_summary.append({
            'Machine': machine_name,
            'MTBF-KaplanMeier': round(MTBF_KM, 2),
            'MTBF-Weibull': round(MTBF_weibull, 2),
            'Optimal Policy': optimal_policy,
            'Optimal Age/CBM Threshold': round(best_age if optimal_policy != 'CBM' else CBM_threshold, 2),
            'Best Cost Rate': round(best_cost_rate_TBM if optimal_policy != 'CBM' else CBM_cost_rate, 2),
            'Savings vs CM': round(savings_TBM_vs_CM if optimal_policy != 'CBM' else savings_CBM_vs_CM, 2),
            'Savings vs TBM': round(savings_CBM_vs_TBM, 2) if optimal_policy == 'CBM' else None
        })
    # printing results for each machine
    for result in results_summary:
        print(f"Machine {result['Machine']}:")
        print(f"The MTBF-KaplanMeier for machine {result['Machine']} is: {result['MTBF-KaplanMeier']}")
        print(f"The MTBF-Weibull for machine {result['Machine']} is: {result['MTBF-Weibull']}")
        print(f"The optimal maintenance policy for machine {result['Machine']} is: {result['Optimal Policy']}")
        if result['Optimal Policy'] != 'CM':
            print(
                f"The optimal {'maintenance age' if result['Optimal Policy'] == 'TBM' else 'CBM threshold'} for machine {result['Machine']} is: {result['Optimal Age/CBM Threshold']}")
            print(f"The best cost rate for machine {result['Machine']} is: {result['Best Cost Rate']}")
            print(
                f"The savings compared to a pure corrective maintenance policy for machine {result['Machine']} are: {result['Savings vs CM']}")
            if result['Optimal Policy'] == 'CBM':
                print(
                    f"The savings compared to a time-based maintenance policy for machine {result['Machine']} are: {result['Savings vs TBM']}")


run_analysis()
