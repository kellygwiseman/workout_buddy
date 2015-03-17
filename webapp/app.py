import sys
sys.path.append(r'../code')
from anonymous_user_prediction import AnonPrediction
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename, SharedDataMiddleware

# Initialize the Flask application
app = Flask(__name__)

# Example user plots
# They are orderd: pro, good, novice
ts_urls = ['https://plot.ly/~kellygwiseman/220','https://plot.ly/~kellygwiseman/310', 'https://plot.ly/~kellygwiseman/160']
bar_urls = ['https://plot.ly/~kellygwiseman/287', 'https://plot.ly/~kellygwiseman/311', 'https://plot.ly/~kellygwiseman/290']
monthly_urls = ['https://plot.ly/~kellygwiseman/221', 'https://plot.ly/~kellygwiseman/208', 'https://plot.ly/~kellygwiseman/161']
tips = ["Great form! Next time add more reps or try a different pushup stance.", "You're doing good. Next time try to keep an even pace throughout your set.", "You're doing ok. Next time try to keep an even pace throughout your set."]
user = 2 # start off plotting novice user

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'

# These are the extensions that we are accepting to be uploaded
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
    return render_template('index.html', user=user, bar_fig=bar_urls, ts_fig=ts_urls, monthly_fig=monthly_urls, tip_text=tips)


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

        # Run anonymous sample
        data = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        p = AnonPrediction(data)
        tip, ts_url, bar_url, monthly_url = p.process_user_sample()
        ts_urls.append(str(ts_url))
        bar_urls.append(str(bar_url))
        monthly_urls.append(str(monthly_url))
        tips.append(tip)
        user += 1
        return render_template('index.html', user=user, bar_fig=bar_urls, ts_fig=ts_urls, monthly_fig=monthly_urls, tip_text=tips)

# This route is expecting a parameter containing the name
# of a file. 
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(
        host = "0.0.0.0",
        port = 8080,
        debug = True)
