import numpy as np
import json
from itertools import product
import pandas as pd


def get_simflix_viewing_data(num_viewers, num_movies):

    # Step 1: Generate a random matrix A
    A = np.random.rand(5, 5)

    # Step 2: Create a symmetric matrix B
    B = A + A.T

    # Step 3: Ensure positive semi-definiteness
    # Option 1: Multiply B by its transpose (guarantees positive semi-definiteness)
    # cov_matrix = np.dot(B, B.T)
    # Option 2: Add a scalar to the diagonal (if needed, to adjust eigenvalues)
    cov_matrix = B + np.eye(5) * 2
    # cov_matrix = np.eye(5

    cov_matrix = cov_matrix / 4

    # Print the correlation matrix
    corr_matrix = np.corrcoef(cov_matrix)

    # means = np.random.rand(5)
    # means[0] = -5 * means[0]
    alpha = np.random.normal(-6, 1, 1).item()
    beta_1 = np.random.normal(1, 1, 1).item()
    beta_2 = np.random.normal(1, 1, 1).item()
    beta_3 = np.random.normal(1, 1, 1).item()
    beta_4 = np.random.normal(1, 1, 1).item()

    means = np.array([alpha, beta_1, beta_2, beta_3, beta_4])
    viewer_coefs = np.random.multivariate_normal(means, cov_matrix, size=num_viewers)

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

    sci_fi_films = [" ".join(movie) for movie in all_sci_fi_permutations]
    sci_fi_films_df = pd.DataFrame({"movie_title": sci_fi_films, "genre": "sci-fi"})

    all_period_drama_permutations = list(
        product(period_drama_name_1, period_drama_name_2, period_drama_name_3, sequel)
    )
    period_drama_films = [" ".join(movie) for movie in all_period_drama_permutations]
    period_drama_films_df = pd.DataFrame(
        {"movie_title": period_drama_films, "genre": "period drama"}
    )

    all_action_permutations = list(
        product(action_name_1, action_name_2, action_name_3, sequel)
    )
    action_films = [" ".join(movie) for movie in all_action_permutations]
    action_films_df = pd.DataFrame({"movie_title": action_films, "genre": "action"})

    all_films = pd.concat([sci_fi_films_df, period_drama_films_df, action_films_df])

    # Sampe n_movies from the list of all possible films
    sampled_movies = all_films.sample(num_movies, replace=False)

    # Dummify the genre column
    sampled_movies = pd.get_dummies(sampled_movies, columns=["genre"])

    # Sample quality scores from a beta distribution
    sampled_movies["quality"] = np.random.beta(2, 5, num_movies)
    sampled_movies["intercept"] = 1

    sampled_movies = sampled_movies[
        [
            "movie_title",
            "intercept",
            "genre_action",
            "genre_period drama",
            "genre_sci-fi",
            "quality",
        ]
    ]

    lin_pred = np.dot(
        viewer_coefs,
        sampled_movies[
            [
                "intercept",
                "genre_action",
                "genre_period drama",
                "genre_sci-fi",
                "quality",
            ]
        ].values.T,
    )

    def sigmoid_and_sample(x):
        p = 1 / (1 + np.exp(-x))
        return np.random.binomial(1, p)

    # Apply the sigmoid function to the linear predictor
    vectorized_sigmoid = np.vectorize(sigmoid_and_sample)
    viewing = vectorized_sigmoid(lin_pred)

    # Create a dataframe with the viewing data
    viewing_df = pd.DataFrame(viewing, columns=sampled_movies["movie_title"])
    viewing_df["viewer_id"] = range(num_viewers)

    # Use faker to generate random names
    from faker import Faker

    fake = Faker()
    viewers = [fake.name() for _ in range(num_viewers)]
    viewing_df["viewer_name"] = viewers

    # Place viewer_id and viewer_name at the beginning of the dataframe
    cols = viewing_df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    viewing_df = viewing_df[cols]

    # Melt the dataframe
    # viewing_df = viewing_df.melt(id_vars=['viewer_id', 'viewer_name'], var_name='movie_title', value_name='viewed')

    # Print the correlation matrix
    corr_matrix = np.corrcoef(cov_matrix)

    # Build the latex model description

    cov_matrix_latex_desc = ""
    for i in range(5):
        for j in range(5):
            cov_matrix_latex_desc += f"{cov_matrix[i,j]:.2f} & "

        cov_matrix_latex_desc += "\\\\"

    cor_matrix_latex_desc = ""
    for i in range(5):
        for j in range(5):
            cor_matrix_latex_desc += f"{corr_matrix[i,j]:.2f} & "

        cor_matrix_latex_desc += "\\\\"

    model_latex = f"""
    The probability of viewer $i$ watching movie $j$ is given by:

    $$p_{{i, j}} = \\text{{logit}}(\\mu_{{i,j}})$$

    Where

    $$ \\mu_{{i,j}} = \\beta_0 + \\sum_{{k=1}}^3 \\beta_{{i, k}} g_{{k,j}} + \\beta_{{4, i}} q_j$$

    And

    <ul>
        <li>$\\beta_{{0,i}}$ the baseline log odds of watching a movie for viewer $i$</li>
        <li>$g_{{k, j}}$ is an indicator variable which is one if the movie is in the kth genre and zero
            otherwise. The genres are coded as 1=sci-fi, 2=action and 3=period costume drama.</li>
        <li>$\\beta_{{i, k}}$ is the log odds multiplier that is applied when viewer $i$ watches genre $k$.</li>
        <li>$q_j$ is a quality score for movie $j$</li>
        <li>$\\beta_{{4, i}}$ controls how viewer $i$ reacts to programme quality.</li>

    </ul>

    The betas are drawn from a multivariate normal distribution so that:

    $$\\mathbf{{\\beta}} \\sim \\mathcal{{N}}(\\boldsymbol{{\\mu}}, \\Sigma)$$

    where:

    $\\boldsymbol{{\\mu}} = [{means[0]:.2f}, {means[1]:.2f}, {means[2]:.2f}, {means[3]:.2f}, {means[4]:.2f}]^T$ is the mean vector and

    <br><br>

    $\\Sigma = \\begin{{bmatrix}}
    {cov_matrix_latex_desc}
    \\end{{bmatrix}}$ is the covariance matrix.

    <br><br>

    The corresponding correlation matrix is:

    <br><br>

    $P = \\begin{{bmatrix}}
    {cor_matrix_latex_desc}
    \\end{{bmatrix}}$ 

    <br><br>

    The $q_j$ are drawn from a Beta distribution with parameters $\\alpha=2$ and $\\beta=5$.
    """

    return viewing_df, corr_matrix, means, cov_matrix, model_latex
