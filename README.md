# Simsets

This repository contains the code that sits behind the simsets API. 

The simsets web API generates simulated data for some common scenarios. Why is this useful? Because if you generated the data then you know the answers to many questions that are unanswerable in real life. Some examples: What is the true contribution of advertising to a time series of product sales? Which latent factors really explain this viewing data? With real life data, modelling allows you to estimate such things but you can't know that you are right.

So having the details of the simulation is like having the answer sheet at the back of the book. This makes simulated perfect for use-cases like:

- Testing forecasting and prediction methods
- Testing methods which try to explain the data using other observable or latent variables
- Testing the claims of service providers who say they can do either of the above
- Creating interview questions
- Creating examples for teaching

Our API saves you the trouble of producing this data. It gives you the data and the answers.

Currently the following endpoints are available:

- [Explainable time series data](http://www.simsets.co.uk/timeseries)
- [Simflix](http://www.simsets.co.uk/simflix) (Simulated viewing data for video-on-demand)

Some notebooks demonstrating use of the endpoints are here and here
