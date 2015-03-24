#Workout Buddy
=============

A webapp that turns your iPhone into a personal trainer. Check out the final product at http://kellygwiseman.pythonanywhere.com/.

#### Inspiration
Often when I go for a long run through Golden Gate Park, I like to stop midway and cross-train.  I already have a fitness device to track my run, but I have no way to record my other exercises.

I developed Workout Buddy to fill this gap by providing detailed tracking and assessment of basic exercises. The beta version starts by analyzing pushups repetitions, but product extensions could include situps, pullups, squats, etc.  Workout Buddy uses equipment that most fitness lovers already have, your smart phone and an armband.  
 
#### Sensor Data
Workout Buddy uses your phone's accelerometer and gyro sensor data, collected from a user wearing their phone in an armband during their workout. The data is recorded at 20 Hertz from the application SensorLog (https://itunes.apple.com/us/app/sensorlog/id388014573?mt=8).

#### Signal Processing
I use a series of filters to separate the pushup signal from the background noise.  First I bandpass filter the data between 0.5 - 2.0 Hertz, then I select the time window when key feature pairs are correlated. The pushup duration detection algorithm is robust to a range of typical user behaviors.  It can handle the phone being worn on either the left or right arm, and can detect pushups from users doing a variety of pushup stances: basic wide stance, elbows-in stance, hands-in stance, and knee pushups.

#### Repetition Detection
Once the pushup duration window is calculated, Workout Buddy uses peak detection algorithms, on the unfiltered pitch data, to pick out the press-down and push-up times for each repetition in the set. The maximum press-down amplitude and repetition duration are calculated to be used as features for classifying the pushup form. In addition, the entire repetition time series, for the pitch rotation and y-component of acceleration, are used during the classification process.

#### Exercise Classification and Rating
Workout Buddy uses an ensemble of classifiers to provide detailed ratings of your latest workout. It uses Random Forest and Support Vector Machine classifiers to model the pitch amplitude and repetition duration features. Dynamic Time Warping, a method of calculating the distance between time series, is used in combination with K-Nearest Neighbors to classify the repetition time series. Each of the models provides a probability that the pushup repetition is either 'ok' or 'good'. The ensemble of models are combined, with equal weights. Workout Buddy uses the binary classification, along with the probabilities, to provide detailed ratings of your set of pushup repetitions.

#### Interactive Visualizations
Workout Buddy provides several interactive visualizations of your latest workout and your recent workout history. Plotly is used to make the interactive plots (all the scripts are available in plotly_graphs.py). There are two visualizations of your latest set of repetitions. The first plots the pitch time series for each repetition, so you can see the variability in your set. An optimal pushup repetition is also plotted, so you can see how close you are to performing an expert pushup. The next visualization is a bar chart of your latest set of repetitions, plotted sequentially. The size of the bar corresponds to the rating of the repetition, with 0% being poor and 100% being excellent form. The rating is the probability of your repetition being 'good' from the ensemble classifier. The bar is colored according to the binary classification of 'ok' (colored red) or 'good' (colored green). The last visualization is a stacked bar chart of your past 30 days of activity, showing the number of 'ok' and 'good' repetitions.

==============
#### Implementation Details ####
The code and data to train your own classifiers is provided and can be run with training_pipeline.py. You can also use the provided models to classify your own data with user_prediction_pipeline.py (for data with user info) or anon_prediction_pipeline (for anonymous sensor data). Run setup.py first to set up the necessary directory structure.

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