import numpy as np
import pandas as pd


def read_video_games():

    # Set variables
    num_time_periods = 1095

    # For the coefficients
    mu_1 = 1000
    mu_2 = -90
    mu_3 = 1000
    mu_4 = 50000
    sigma_1 = 100
    sigma_2 = 10
    sigma_3 = 100
    sigma_4 = 1000

    # For the events
    mu_5 = 100
    sigma_5 = 10

    # For the noise
    mu_6 = 0
    sigma_6 = 100

    # For pi
    alpha_1 = 1
    beta_1 = 250

    # For tau
    alpha_2 = 5
    beta_2 = 15

    # Draw coefficients
    theta_1 = np.random.normal(mu_1, sigma_1, 1).item()  # intercept
    theta_2 = np.random.normal(mu_2, sigma_2, 1).item()  # temp
    theta_3 = np.random.normal(mu_3, sigma_3, 1).item()  # holidays
    theta_4 = np.random.normal(mu_4, sigma_4, 1).item()  # trend

    event_params = []
    for t in range(3):
        theta = np.random.normal(mu_5, sigma_5, 1).item()  # coefficient for events
        rho = np.random.beta(alpha_1, beta_1, 1).item()  # event occurrence
        tau = np.random.beta(alpha_2, beta_2, 1).item()  # event carry over
        event_params.append({"theta": theta, "rho": rho, "tau": tau})

    # Create the X variables
    x = np.linspace(-np.pi, 5 * np.pi, num_time_periods)
    temp = (np.sin(x + 1) + np.random.normal(0, 0.5, num_time_periods)) * 10 + 10
    x = np.linspace(0, 0.2, num_time_periods)
    trend = x**2
    holidays = [1] * 21 + [0] * 77 + [1] * 21 + [0] * 77 + [1] * 84 + [0] * 77 + [1] * 8
    holidays = np.array(holidays + holidays + holidays)

    event_group = []
    event_group_nc = []

    for t in range(3):
        event_occurrence = np.random.binomial(
            1, event_params[t]["rho"], num_time_periods
        )
        carry_over_effect = np.zeros(num_time_periods)
        for i in range(num_time_periods):
            carry_over_effect[i] = (
                event_occurrence[i] + event_params[t]["tau"] * carry_over_effect[i - 1]
            )
        event_group.append(carry_over_effect)
        event_group_nc.append(event_occurrence)

    # Create the noise
    noise = np.random.normal(mu_6, sigma_6, num_time_periods)

    # Create the Y variable
    video_game_searches = (
        theta_1
        + theta_2 * temp
        + theta_3 * holidays
        + theta_4 * trend
        + event_params[0]["theta"] * event_group[0]
        + event_params[1]["theta"] * event_group[1]
        + event_params[2]["theta"] * event_group[2]
        + noise
    )

    # Create dataframe of searches and explanatory variables

    # Create the dates
    dates = pd.date_range(start="1/1/2018", periods=num_time_periods)

    video_games_df = pd.DataFrame(
        {
            "searches": video_game_searches,
            "date": dates,
            "temp": temp,
            "holidays": holidays,
            "big_game_release": event_group_nc[0],
            "comic_con": event_group_nc[1],
            "console_release": event_group_nc[2],
        },
    )

    # Convert to json
    video_game_searches_json = video_games_df.to_dict(orient="records")

    # Create a dictionary of the parameters
    params = {
        "theta_1": theta_1,
        "theta_2": theta_2,
        "theta_3": theta_3,
        "theta_4": theta_4,
        "event_params": event_params,
    }

    return {"data": video_game_searches_json, "params": params}
