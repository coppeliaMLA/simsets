from flask import Flask, request, send_file, render_template, url_for
from flask_swagger_ui import get_swaggerui_blueprint
from flasgger import Swagger
from movies import read_movies
from video_games import read_video_games
import pandas as pd
import time_series as ts
import markdown
from mdx_math import MathExtension
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# Global variable to store CSV data
csv_data = None


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
    global csv_data
    output_type = request.args.get("output_type", default="json")
    num_time_periods = request.args.get("num_time_periods", default=365, type=int)

    contributions, observed, latex = ts.simulate_all_params(num_time_periods)

    if output_type == "json":
        observed_json = observed.to_json(orient="records")
        contributions_json = contributions.to_json(orient="records")
        return {
            "observed": observed_json,
            "contributions": contributions_json,
            "latex": latex,
        }
    else:
        csv_data = observed
        md = markdown.Markdown(
            extensions=["mdx_math"],
            extension_configs={"mdx_math": {"enable_dollar_delimiter": False}},
        )
        model_latex = md.convert(latex)

        # Use matplotlib to generate a plot of the observed data as a png
        plt.figure(figsize=(8, 3))
        _ = observed["y"].plot()
        plt.savefig("static/observed_plot.png")  # Save the plot in the static directory
        plt.close()  # Close the plot to free up memory

        # Return both files
        return render_template(
            "time_series_desc.html",
            model_latex=model_latex,
            plot_url=url_for("static", filename="observed_plot.png"),
        )


@app.route("/download_time_series", methods=["GET"])
def download_time_series():
    global csv_data
    if csv_data is None:
        return "No data to download."

    # Create a temporary CSV file and serve it for download
    with open("observed.csv", "w") as csv_file:
        csv_data.to_csv(csv_file, index=True)

    return send_file("observed.csv", as_attachment=True, download_name="observed.csv")
