from flask import Flask, render_template
from flask import request
from util import initModel, generateMusic
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/generate', methods=['POST'])
def play_song():
  q = request.form.get('q')
  initModel()
  generateMusic(q)
  timestamp = datetime.now().timestamp()
  song_path = 'static/music/output.wav'
  
  if song_path:
    return render_template('index.html', song_path=song_path, timestamp=timestamp)
  else:
    return "No song selected or invalid option.", 400

if __name__ == '__main__':
  app.run(debug=True)