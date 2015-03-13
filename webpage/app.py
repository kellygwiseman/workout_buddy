import sys
sys.path.append(r'../code')
from user_prediction import UserPrediction
import pandas as pd
import plotly_graphs as pg
import os
# We'll render HTML templates and access data sent by POST
# using the request object from flask. Redirect and url_for
# will be used to redirect the user once the upload is done
# and send_from_directory will help us to send/show on the
# browser the file that the user just uploaded
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename, SharedDataMiddleware

# Initialize the Flask application
app = Flask(__name__)
user = 2

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['txt'])

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/')
def index():
    global user
    return render_template('index.html', user = user, daily_fig = daily_urls, ts_fig = ts_urls, monthly_fig = monthly_urls, tip_text = tips)


# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    global user
    # Get the name of the uploaded file
    file = request.files['file']
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        # Move the file form the temporal folder to
        # the upload folder we setup
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # run sample
        info = pd.read_table('uploads/'+filename, sep=',', skipinitialspace=True)
        p = UserPrediction(info, 100)
        prob_history, bin_history, tip, daily_url, ts_url = p.batch_process_user_samples()
        monthly_url = pg.monthly_reps(bin_history, 100)
        ts_urls.append(str(ts_url))
        daily_urls.append(str(daily_url))
        monthly_urls.append(str(monthly_url))
        tips.append(tip)
        user += 1
        return render_template('index.html', user = user, daily_fig = daily_urls, ts_fig = ts_urls, monthly_fig = monthly_urls, tip_text = tips)

# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Orderd by pro, good, novice
    ts_urls = ['https://plot.ly/~kellygwiseman/220','https://plot.ly/~kellygwiseman/207', 'https://plot.ly/~kellygwiseman/160']
    daily_urls = ['https://plot.ly/~kellygwiseman/219', 'https://plot.ly/~kellygwiseman/206', 'https://plot.ly/~kellygwiseman/159']
    monthly_urls = ['https://plot.ly/~kellygwiseman/221', 'https://plot.ly/~kellygwiseman/208', 'https://plot.ly/~kellygwiseman/161']
    tips = ["You're doing good. Next time try to keep an even pace throughout your set.","You're doing good. Try to switch to regular pushups next time.", "You're doing ok. Next time try to keep an even pace throughout your set."]

    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True)
