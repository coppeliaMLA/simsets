from flask import Flask, request, send_file
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
    output_type = request.args.get("output_type", default="json")
    num_time_periods = request.args.get("num_time_periods", default=365, type=int)

    contributions, observed, latex = ts.simulate_all_params(num_time_periods)

    print(f"Output type: {output_type}")

    if output_type == "json":
        observed_json = observed.to_json(orient="records")
        return {"observed": observed_json, "latex": latex}
    else:
        observed.to_csv("observed.csv", index=False)
        return send_file("observed.csv")
