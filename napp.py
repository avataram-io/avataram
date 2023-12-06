from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from src.gradio_demo import SadTalker
import os
import shutil
import cloudinary
from cloudinary.uploader import upload


app = Flask(__name__)

cloudinary.config(
    cloud_name="drettcko3",
    api_key="734375513744988",
    api_secret="sKUIFHNcjBc1VUcv9rCuuk5-S1A"
)

app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['STATIC_FOLDER'] = os.path.join(os.getcwd(), 'static')

# Initialize SadTalker outside the route to load the model only once
checkpoint_path = 'checkpoints'
config_path = 'src/config'
sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)

def upload_to_cloudinary(file_path,resource_type='auto'):
    print(f"Uploading file: {file_path}")
    # response = upload(file_path)
    response = upload(file_path, resource_type=resource_type)
    return response['secure_url']

def remove_local_files(directory):
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

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

    # Upload files to Cloudinary
    cloud_source_image_url = upload_to_cloudinary(source_image_path)
    cloud_driven_audio_url = upload_to_cloudinary(driven_audio_path,resource_type='raw') if driven_audio_path else None

    # Get other form data
    pose_style = 0
    size_of_image = 512
    preprocess_type = 'full'
    is_still_mode = True
    enhancer = True

    # Perform SadTalker processing
    result_video = sad_talker.test(source_image_path, driven_audio_path, preprocess_type, enhancer, is_still_mode, size_of_image)
    print('result_video_is_here :-', result_video)
    cloud_driven_result_url = upload_to_cloudinary(result_video,resource_type='video')
    print('cloud_driven_result_url :',cloud_driven_result_url)

    # Save the result video in the static folder
    static_result_video_path = os.path.join(app.config['STATIC_FOLDER'], 'result_video.mp4')
    shutil.move(result_video, static_result_video_path)

    # Remove local files
    remove_local_files(upload_dir)

    return render_template('home.html', result_video=cloud_driven_result_url)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
