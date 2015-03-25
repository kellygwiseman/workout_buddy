#Workout Buddy
=============

A webapp that turns your iPhone into a personal trainer. Check out the final product at http://kellygwiseman.pythonanywhere.com/. You can also view KellyWiseman_presentation.pdf for visualizations of the data and processing steps.

#### Inspiration
Often when I go for a long run, I like to stop midway and cross-train.  I already have a fitness device to track my run, but I have no way to record my other exercises.

I developed Workout Buddy to fill this gap by providing detailed tracking and assessment of basic exercises. The beta version starts by analyzing pushups repetitions, but product extensions could include situps, pullups, squats, etc.  Workout Buddy uses equipment that most fitness lovers already have, your smart phone and an armband.  
 
#### Sensor Data
Workout Buddy uses your phone's accelerometer and gyro sensor data, collected from a user wearing their phone in an armband. The data is recorded at 20 Hz from the application SensorLog (https://itunes.apple.com/us/app/sensorlog/id388014573?mt=8).

#### Signal Processing
Workout Buddy uses a series of filters to separate the pushup signal from the background noise.  First the data is bandpass filtered between 0.5 - 2.0 Hz, then the pushup duration window is extracted by picking the window where key feature pairs are correlated above a threshold (see filter.py and process_data.py for more details). The pushup duration detection algorithm is robust to a range of typical user behaviors.  It is resilient to the phone being worn on either the left or right arm. It is also robust to a variety of pushup stances, including: basic wide stance, elbows-in stance, hands-in stance, and knee pushups. 

#### Repetition Detection
Once the pushup duration window is calculated, Workout Buddy uses peak detection algorithms, on the unfiltered pitch data, to pick out the press-down and push-up times for each repetition in the set. The maximum press-down amplitude and the repetition duration are extracted for classifying the pushup form. In addition, the entire pitch and y-acceleration repetition time series are used during the classification process (See process_data.py and detect_peaks.py for more details).

#### Exercise Classification and Rating
Workout Buddy uses an ensemble of classifiers to provide detailed ratings of your latest workout. It uses Random Forest and Support Vector Machine classifiers to model the amplitude and repetition duration features. Dynamic Time Warping, a method of calculating the distance between time series, is used in combination with K-Nearest Neighbors to classify the repetition time series. Each of the models provides a probability that the pushup repetition is either 'ok' or 'good'. The ensemble of models are combined, with equal weights. Workout Buddy uses the binary classification, along with the probabilities, to provide detailed ratings of your set of pushup repetitions (see classify.py and dtw.py for more details).

#### Interactive Visualizations
Workout Buddy provides several interactive visualizations of your latest workout, and your workout history, on the webapp dashboard. Plotly is used to make the interactive plots (see plotly_graphs.py for more details). There are two visualizations of your latest set of repetitions. The first visualization plots the pitch time series for each repetition, so you can see the variability in your set. An optimal pushup repetition is also plotted, so you can see how close you are to performing an expert pushup. The next visualization is a bar chart of your latest set of repetitions, plotted sequentially. The size of the bar corresponds to the rating of the repetition, with 0% being poor and 100% being excellent form. The rating is the probability of your repetition being 'good' from the ensemble classification model. The bar is colored according to the binary classification of 'ok' (colored red) or 'good' (colored green). The last visualization is a stacked bar chart of your past 30 days of activity, showing the number of 'ok' and 'good' repetitions.

==============
#### Implementation Details ####
The code, and the data to train your own classifiers, is provided and can be run with the script training_pipeline.py. You can also use the provided models to classify your own data with the script user_prediction_pipeline.py (for data with user info) or anon_prediction_pipeline (for anonymous sensor data). Run setup.py first to set up the necessary directory structure.

#### Necessary Python Packages ####
1. cPickle
2. Flask - just for webapp
3. Matplotlib
4. Numpy
5. Pandas
6. Plotly
7. Scikit-Learn
8. Scipy
9. Werkzeug - just for webapp 