from flask import Flask, redirect, url_for, session, jsonify
from authlib.integrations.flask_client import OAuth
from authlib.common.security import generate_token
import os
import json

app = Flask(__name__)
app.secret_key = os.urandom(24)


oauth = OAuth(app)

nonce = generate_token()


oauth.register(
    name=os.getenv('OIDC_CLIENT_NAME'),
    client_id=os.getenv('OIDC_CLIENT_ID'),
    client_secret=os.getenv('OIDC_CLIENT_SECRET'),
    #server_metadata_url='http://dex:5556/.well-known/openid-configuration',
    authorization_endpoint="http://localhost:5556/auth",
    token_endpoint="http://dex:5556/token",
    jwks_uri="http://dex:5556/keys",
    userinfo_endpoint="http://dex:5556/userinfo",
    device_authorization_endpoint="http://dex:5556/device/code",
    client_kwargs={'scope': 'openid email profile'}
)

@app.route('/')
def home():
    user = session.get('user')
    if user:
        return f"<h2>Logged in as {user['email']}</h2><a href='/logout'>Logout</a>"
    return '<a href="/login">Login with Dex</a>'

@app.route('/login')
def login():
    session['nonce'] = nonce
    redirect_uri = 'http://localhost:8000/authorize'
    return oauth.flask_app.authorize_redirect(redirect_uri, nonce=nonce)

@app.route('/authorize')
def authorize():
    token = oauth.flask_app.authorize_access_token()
    nonce = session.get('nonce')

    user_info = oauth.flask_app.parse_id_token(token, nonce=nonce)  # or use .get('userinfo').json()
    session['user'] = user_info
    return redirect('/')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

# Get a list of all course codes in our system
@app.route('/api/courses_list', methods=["GET"])
def get_courses_list():
  curr_dir = os.path.dirname(os.path.abspath(__file__))
  courses_path = os.path.join(curr_dir, 'data', 'test_course_info.json')
  
  with open(courses_path, "r") as file:
    courses = json.load(file)
    
  courses_list = list(courses.keys())
  
  return jsonify(courses_list)
  
# Get videos for a course code
@app.route('/api/videos/<course_code>', methods=["GET"])
def get_videos(course_code):
  curr_dir = os.path.dirname(os.path.abspath(__file__))
  courses_path = os.path.join(curr_dir, 'data', 'test_course_info.json')
  
  with open(courses_path, "r") as file:
    courses = json.load(file)
    
  videos = courses[course_code]["videos"]
  
  return jsonify(videos)

# Get course information for a course code
@app.route('/api/courses/<course_code>', methods=["GET"])
def get_course_info(course_code):
  curr_dir = os.path.dirname(os.path.abspath(__file__))
  courses_path = os.path.join(curr_dir, 'data', 'test_course_info.json')
  
  with open(courses_path, "r") as file:
    courses = json.load(file)
    
  course_info = {
    "title": courses[course_code]["title"],
    "description": courses[course_code]["description"]
  } 
  
  return jsonify(course_info)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
