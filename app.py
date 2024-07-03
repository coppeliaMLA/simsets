from flask import Flask, request
from flask_swagger_ui import get_swaggerui_blueprint
from flasgger import Swagger
from movies import read_movies
from video_games import read_video_games
import pandas as pd
import time_series as ts

app = Flask(__name__)
Swagger(app)

SWAGGER_URL = "/api/docs"  # URL for exposing Swagger UI (without trailing '/')
API_URL = "/static/swagger.yml"  # Our API url (can of course be a local resource)

# Call factory function to create our blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
    API_URL,
    config={"app_name": "Test application"},  # Swagger UI config overrides
)


app.register_blueprint(swaggerui_blueprint)


@app.route("/")
def index():
    return "Hello World!"


@app.route("/test", methods=["GET"])
def say_hi():
    return "Hi, I'm test.py"


@app.route("/vod_movies", methods=["GET"])
def get_movies():
    try:
        # Extract num_viewers from query parameters and convert to integer
        num_viewers = int(request.args.get("num_viewers", 0))
    except ValueError:
        num_viewers = 100
    return read_movies(num_viewers=num_viewers)


@app.route("/timeseries", methods=["GET"])
def get_timeseries():
    intercept = ts.Intercept("intercept", (500, 50))
    trend = ts.Trend("trend", (0, 500), "quadratic")
    seasonality = ts.Seasonality("seasonality", (0, 1), 1)
    event_1 = ts.Event("event_1", 1, (20, 5), 15.0, (0.5, 0.5))
    event_2 = ts.Event("event_2", 2, (30, 5), 3.5, (5, 1))
    event_3 = ts.Event("event_3", 2, (10, 1), 40.5, (10, 1))
    holidays = ts.Holidays("holidays", (0, 10))
    noise = ts.Noise("noise", (9, 0.5), 1, 1)
    sim_ts = ts.SimulatedTimeSeries(
        [intercept, trend, seasonality, holidays, event_1, event_2, event_3, noise]
    )
    contributions, observed, latex = sim_ts.generate_time_series(1095)
    observed_json = observed.to_json(orient="records")
    return {"observed": observed_json, "latex": latex}
