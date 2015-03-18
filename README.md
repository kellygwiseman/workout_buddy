#Workout Buddy
=============

A webapp that turns your iPhone into a personal trainer.

#### Inspiration
Often when I go for a long run through Golden Gate Park, I like to stop midway and cross-train.  I already have a fitness device to track my run, but I have no way to record my other exercises.

I developed Workout Buddy to fill this gap by providing detailed tracking and assessment of basic exercises. The beta version starts by analyzing pushups repetitions, but product extensions could include situps, pullups, squats, etc.  Workout Buddy uses equipment that most fitness lovers already have, your smart phone and an armband.  
 
#### Sensor Data
Workout Buddy uses your phone's accelerometer and gyro sensor data, collected from a user wearing their phone in an armband during their workout. The data is recorded at 20 Hertz from the application SensorLog (https://itunes.apple.com/us/app/sensorlog/id388014573?mt=8).

#### Signal Processing
I use a series of filters to separate the pushup signal from the background noise.  First I bandpass filter the data between 0.5 - 2.0 Hertz, then I select the time window when key feature pairs are correlated. The pushup duration detection algorithm is robust to a range of typical user behaviors.  It can handle the phone being worn on either the left or right arm, and can detect pushups from users doing a variety of pushup stances: basic wide stance, elbows-in stance, hands-in stance, and knee pushups.

#### Repetition Detection

#### Exercise Classification and Rating

#### Interactive Visualizations