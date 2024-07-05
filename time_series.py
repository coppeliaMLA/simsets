import numpy as np
import pandas as pd


# Create a class called ExplantoryVariable
class ExplanatoryVariable:
    def __init__(self, name, coefficient):
        self.name = name

        # If the coefficient is a tuple then sample from a normal distribution
        if isinstance(coefficient, tuple):
            self.coefficient = np.random.normal(coefficient[0], coefficient[1])
        else:
            self.coefficient = coefficient


class Intercept(ExplanatoryVariable):
    def __init__(self, name, coefficient):
        super().__init__(name, coefficient)

    def generate(self, length):
        self.contribution = np.ones(length) * self.coefficient
        self.observed = None

        return self.contribution

    def describe(self):
        # Create a dictionary to store the description
        desc = {
            "name": self.name,
            "coefficient": self.coefficient,
            "latex": f"{self.coefficient:.2f} ",
            "latex_context": "",
        }

        return desc


class Seasonality(ExplanatoryVariable):
    def __init__(self, name, coefficient, period):
        super().__init__(name, coefficient)
        self.period = period

        # If the coefficient is a tuple then sample from a normal distribution
        if isinstance(coefficient, tuple):
            self.coefficient = np.random.normal(coefficient[0], coefficient[1])
        else:
            self.coefficient = coefficient

    def generate(self, length):
        x = np.array([x for x in range(length)])
        self.observed = 5 * np.sin((x - 91.25) * 2 * np.pi / 365) + np.random.normal(
            0, 0.5, length
        )
        self.contribution = self.observed * self.coefficient

        return self.contribution

    def describe(self):

        sign = "+ " if self.coefficient > 0 else ""

        # Create a dictionary to store the description
        desc = {
            "name": self.name,
            "coefficient": self.coefficient,
            "period": self.period,
            "latex": sign + f"{self.coefficient:.2f} s_t ",
            "latex_context": "",
        }

        return desc


# Create a subclass called trend that inherits from ExplanatoryVariable
class Trend(ExplanatoryVariable):
    def __init__(self, name, coefficient, functional_form):
        super().__init__(name, coefficient)
        self.functional_form = functional_form

    def generate_functional_form(self, length):
        if self.functional_form == "slope":
            x = np.array([float(i) for i in range(length)])
            return x
        if self.functional_form == "quadratic":
            x = np.linspace(0, 0.2, length)
            return x**2
        elif self.functional_form == "log":
            x = np.linspace(0, 6, length)
            return np.log(x + 1)
        else:
            raise ValueError("Functional form not recognized")

    def generate_trend_shape(self, length):

        wo_breaks = self.generate_functional_form(length)

        # Scale so that the trend is between 0 and 1
        # w_breaks = (wo_breaks - np.min(wo_breaks)) / (
        #    np.max(wo_breaks) - np.min(wo_breaks)
        # )

        return wo_breaks

    def generate(self, length):
        self.trend_shape = self.generate_trend_shape(length)
        self.contribution = self.trend_shape * self.coefficient
        self.observed = None
        self.length = length

        return self.contribution

    def describe(self):
        # Create a dictionary to store the description

        if self.functional_form == "slope":
            latex = f"{self.coefficient:.2f}t"
        elif self.functional_form == "quadratic":
            latex = f"{10000*self.coefficient*(0.2/self.length)**2:.4f}t^2 \\times 10^{{-4}} "
        elif self.functional_form == "log":
            latex = f"{self.coefficient:.2f} \\ln({6/self.length:.2f}t + 1)"
        else:
            raise ValueError("Functional form not recognized")

        sign = "+ " if self.coefficient > 0 else ""

        desc = {
            "name": self.name,
            "coefficient": self.coefficient,
            "functional_form": self.functional_form,
            "trend_shape": self.trend_shape,
            "latex": sign + latex,
            "latex_context": "",
        }

        return desc


class Holidays(ExplanatoryVariable):
    def __init__(self, name, coefficient):
        super().__init__(name, coefficient)

    def generate(self, length):
        holiday_vector = (
            [1] * 21 + [0] * 77 + [1] * 21 + [0] * 77 + [1] * 84 + [0] * 77 + [1] * 8
        )
        # Use modulo to repeat the holidays vector
        holidays = np.array(holiday_vector * ((length // len(holiday_vector)) + 1))
        # Truncate the holidays vector to the length of the time series
        holidays = holidays[:length]

        self.contribution = holidays * self.coefficient
        self.observed = holidays

        return self.contribution

    def describe(self):
        # Create a dictionary to store the description

        sign = "+ " if self.coefficient > 0 else ""
        desc = {
            "name": self.name,
            "coefficient": self.coefficient,
            "latex": sign + f"{self.coefficient:.2f} h_t ",
            "latex_context": "",
        }

        return desc


class Noise(ExplanatoryVariable):
    def __init__(
        self, name, standard_dev, autocorrelation, heteroskedasticity, coefficient=1
    ):
        super().__init__(name, coefficient)

        if isinstance(standard_dev, tuple):
            self.standard_dev = np.random.gamma(standard_dev[0], standard_dev[1])
        else:
            self.standard_dev = standard_dev

        self.autocorrelation = autocorrelation
        self.heteroskedasticity = heteroskedasticity

    def generate(self, length):
        # Sample from a normal distribution
        self.contribution = np.random.normal(0, self.standard_dev, length)
        self.observed = None

        return self.contribution

    def describe(self):
        # Create a dictionary to store the description
        desc = {
            "name": self.name,
            "standard_dev": self.standard_dev,
            "autocorrelation": self.autocorrelation,
            "heteroskedasticity": self.heteroskedasticity,
            "latex": "+ \\epsilon_t ",
            "latex_context": f"\\epsilon_t \\sim N(0, {self.standard_dev:.2f}^2)",
        }

        return desc


class Event(ExplanatoryVariable):
    def __init__(self, name, index, coefficient, num_events, carry_over):
        super().__init__(name, coefficient)
        self.num_events = num_events
        self.carry_over = carry_over
        self.index = index

        if isinstance(num_events, float):
            self.num_events = np.random.poisson(num_events)
        else:
            self.num_events = num_events

        if isinstance(carry_over, tuple):
            self.carry_over = np.random.beta(carry_over[0], carry_over[1])
        else:
            self.carry_over = carry_over

    def generate(self, length):

        events = np.zeros(length)

        # Randomly select the time periods where the events occur
        event_periods = np.random.choice(
            range(length), size=self.num_events, replace=False
        )

        # Set the events to a random beta distribution
        for e in event_periods:
            events[e] = np.random.beta(5, 6, 1).item()

        carry_over_effect = np.zeros(length)
        for i in range(length):
            carry_over_effect[i] = (
                events[i] + self.carry_over * carry_over_effect[i - 1]
            )
        self.contribution = carry_over_effect * self.coefficient
        self.observed = events

        return self.contribution

    def describe(self):

        sign = "+ " if self.coefficient > 0 else ""

        # Create a dictionary to store the description
        desc = {
            "name": self.name,
            "index": self.index,
            "coefficient": self.coefficient,
            "carry_over": self.carry_over,
            "latex": sign + f"{self.coefficient:.2f} v_{{{self.index}, t}} ",
            "latex_context": f"v_{{{self.index}, t}} = e_{{{self.index}, t}} + {self.carry_over:.2f} v_{{{self.index}, t-1}}",
        }

        return desc


class SimulatedTimeSeries:
    def __init__(self, explanatory_variables):
        self.explanatory_variables = explanatory_variables

    def generate_time_series(self, num_time_periods, random_seed=None):

        if random_seed is not None:
            np.random.seed(random_seed)
        contributions_df = pd.DataFrame()
        observed_df = pd.DataFrame()
        latex_model_desc = ""
        latex_context = ""

        for i, e in enumerate(self.explanatory_variables):
            contributions_df[e.name] = e.generate(num_time_periods)
            if e.observed is not None:
                observed_df[e.name] = e.observed
            latex_model_desc += " " + e.describe()["latex"]
            if (i + 1) % 7 == 0 and i != len(self.explanatory_variables) - 1:
                latex_model_desc += "$ \n\n $"
            lc = e.describe()["latex_context"]
            if len(lc) > 0:
                latex_context += "\n\n $" + lc + "$ "

        # Sum the explanatory variables to create the Y variable
        contributions_df["y"] = contributions_df.sum(axis=1)

        observed_df["y"] = contributions_df["y"]

        observed_df["date"] = pd.date_range(start="1/1/2018", periods=num_time_periods)

        # Set date as the index
        observed_df.set_index("date", inplace=True)

        # Create latex description
        latex = f"$y_t = {latex_model_desc}$\n\n Where: \n\n $t = 0, 1, 2 \\ldots, n$ \n\n and $s, h, v_i$ are the seasonality, holiday and events variables in the generated data set. The effect of the events include a carry over effect and are modelled recursively as follows: {latex_context}\n\n The $e_i$ correspond to the events variables in the data set."

        return contributions_df, observed_df, latex


def simulate_all_params(num_time_periods):
    intercept = Intercept("intercept", (500, 50))
    # Randomly pick a type of trend from slope, quadratic, or log
    trend_type = np.random.choice(["slope", "quadratic", "log"])

    if trend_type == "slope":
        trend = Trend("trend", (0, 0.5), "slope")
    elif trend_type == "quadratic":
        trend = Trend("trend", (0, 50), "quadratic")
    else:
        trend = Trend("trend", (0, 50), "log")

    seasonality = Seasonality("seasonality", (0, 1), 365)

    # Randomly pick a number of events between 0 and 10
    num_events = np.random.randint(0, 10)
    events = []
    for i in range(num_events):
        event = Event(f"event_{i}", i, (20, 5), 15.0, (0.5, 0.5))
        events.append(event)

    holidays = Holidays("holidays", (0, 10))
    noise = Noise("noise", (9, 0.5), 1, 1)

    sim_ts = SimulatedTimeSeries(
        [intercept, trend, seasonality, holidays, *events, noise]
    )

    contributions, observed, latex = sim_ts.generate_time_series(num_time_periods)

    return contributions, observed, latex
