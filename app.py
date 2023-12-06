from flask import Flask, render_template, request,send_from_directory
from src.gradio_demo import SadTalker
from werkzeug.utils import secure_filename
import os 
import shutil
import cloudinary

app = Flask(__name__)
          
cloudinary.config( 
  cloud_name = "name", 
  api_key = "key-no", 
  api_secret = "keys-secret" 
)

app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['STATIC_FOLDER'] = os.path.join(os.getcwd(), 'static')

# Initialize SadTalker outside the route to load the model only once
checkpoint_path = 'checkpoints'
config_path = 'src/config'
sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/process', methods=['POST'])
def process():
    # Get the uploaded files
    source_image = request.files['source_image']
    driven_audio = request.files['driven_audio']

    # Generate secure filenames for the uploaded files
    source_image_filename = secure_filename(source_image.filename)
    driven_audio_filename = secure_filename(driven_audio.filename)

    # Define the upload directory
    upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'input')
    os.makedirs(upload_dir, exist_ok=True)

    # Save the uploaded files with secure filenames
    source_image_path = os.path.join(upload_dir, source_image_filename)
    driven_audio_path = os.path.join(upload_dir, driven_audio_filename)

    source_image.save(source_image_path)
    if driven_audio:
        driven_audio.save(driven_audio_path)
    else:
        driven_audio_path = None

    # Get other form data
    pose_style = 0
    size_of_image = 512
    preprocess_type = 'full'
    is_still_mode = True
    enhancer = True

    # Define the input directory
    input_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'input')

    # Perform SadTalker processing
    result_video = sad_talker.test(source_image_path, driven_audio_path, preprocess_type, enhancer, is_still_mode, size_of_image)
    print('result_video_is_here :-', result_video)
    # Save the result video in the static folder
    static_result_video_path = os.path.join(app.config['STATIC_FOLDER'], 'result_video.mp4')
    shutil.move(result_video, static_result_video_path)
    return render_template('home.html', result_video='result_video.mp4')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)




