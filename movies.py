import numpy as np
import pandas as pd
import random
import faker
from itertools import product


def read_movies(num_films=10, num_viewers=100):
    """
    This function responds to a request for /api/nodes
    with the prog node data

    :return:        node data
    """

    sci_fi_name_1 = ["evil", "metalic", "cosmic", "blood thirsty", "galactic", "mutant"]
    sci_fi_name_2 = ["droid", "zombie", "space zebra"]
    sci_fi_name_3 = ["from outer space", "invasion", "attack on Mars"]

    action_name_1 = ["ultimate", "bad", "fatal", "unstoppable", "crucial", "loaded"]
    action_name_2 = ["weapon", "cop", "tough guy"]
    action_name_3 = ["in the line of fire", "death wish", "mission"]
    period_drama_name_1 = [
        "pride",
        "obstinacy",
        "indolence",
        "luxury",
        "sense",
        "sensibility",
    ]
    period_drama_name_2 = ["and", "in", "over"]
    period_drama_name_3 = ["prejudice", "paradise", "ecstacy"]
    sequel = ["", "V", "II", "III", "IV"]

    # Generate all possible permutations for sci-fi
    all_sci_fi_permutations = list(
        product(sci_fi_name_1, sci_fi_name_2, sci_fi_name_3, sequel)
    )

    # Randomly select 10 sci-fi films
    sci_fi_films = random.sample(all_sci_fi_permutations, 10)
    sci_fi_films = [" ".join(movie) for movie in sci_fi_films]

    # Generate all possible permutations for period drama
    all_period_drama_permutations = list(
        product(period_drama_name_1, period_drama_name_2, period_drama_name_3, sequel)
    )

    # Randomly select 10 period drama films
    period_drama_films = random.sample(all_period_drama_permutations, 10)
    period_drama_films = [" ".join(movie) for movie in period_drama_films]

    # Generate all possible permutations for action films
    all_action_permutations = list(
        product(action_name_1, action_name_2, action_name_3, sequel)
    )

    # Randomly select 10 action films
    action_films = random.sample(all_action_permutations, 10)
    action_films = [" ".join(movie) for movie in action_films]

    # Generate viewer preferences
    action = np.random.beta(0.2, 0.1, num_viewers)  # preference for action
    sci_fi = np.random.beta(0.1, 0.1, num_viewers)  # preference for sci-fi
    period_drama = np.random.beta(2, 2, num_viewers)  # preference for period drama

    # Generate film reputation scores
    sci_fi_rep = np.random.uniform(0, 1, num_films)
    period_drama_rep = np.random.uniform(0, 1, num_films)
    action_rep = np.random.uniform(0, 1, num_films)

    # Generate viewing data
    action_viewing = np.transpose(
        np.array(
            [
                [np.random.binomial(1, x * y, 1).item() for x in action]
                for y in action_rep
            ]
        )
    )
    period_drama_viewing = np.transpose(
        np.array(
            [
                [np.random.binomial(1, x * y, 1).item() for x in period_drama]
                for y in period_drama_rep
            ]
        )
    )
    sci_fi_viewing = np.transpose(
        np.array(
            [
                [np.random.binomial(1, x * y, 1).item() for x in sci_fi]
                for y in sci_fi_rep
            ]
        )
    )

    # Combine viewing data
    viewing = np.concatenate(
        (action_viewing, period_drama_viewing, sci_fi_viewing), axis=1
    )

    # Generate viewing minutes
    action_viewing_mins = np.array(
        [np.random.poisson(x * 50, num_films).tolist() for x in action]
    )
    period_drama_viewing_mins = np.array(
        [np.random.poisson(x * 50, num_films).tolist() for x in period_drama]
    )
    sci_fi_viewing_mins = np.array(
        [np.random.poisson(x * 50, num_films).tolist() for x in sci_fi]
    )
    viewing_mins = np.concatenate(
        (action_viewing_mins, period_drama_viewing_mins, sci_fi_viewing_mins), axis=1
    )
    viewing = np.multiply(viewing, viewing_mins)

    # Create a dataframe
    viewing_df = pd.DataFrame(viewing)
    viewing_df.columns = sci_fi_films + action_films + period_drama_films

    # Create a random name for each viewer
    fake = faker.Faker()
    viewers = [fake.name() for _ in range(num_viewers)]

    # Add viewer names and IDs to the dataframe
    viewing_df["viewer"] = viewers
    viewing_df["viewer_id"] = range(num_viewers)

    # Put both viewer and viewer_id at the beginning of the dataframe
    viewing_df = viewing_df[
        ["viewer", "viewer_id"] + sci_fi_films + action_films + period_drama_films
    ]

    # Convert to an ordered dictionary

    # Convert to json
    viewing_json = viewing_df.to_dict(orient="list")

    return viewing_json
