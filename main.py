from flask import Flask, render_template
from flask import request
from util import initModel, generateMusic
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def home():
  return render_template('index.html', song_path='', timestamp='', q='')

@app.route('/generate', methods=['POST'])
def play_song():
  q = request.form.get('q')

  if (not q is None):
    initModel()
    generateMusic(q)
    timestamp = str(datetime.now().timestamp())
    song_path = 'static/music/output.wav'
    if song_path:
      return render_template('index.html', song_path=song_path, timestamp=timestamp, quarter=q)
    else:
      return "No song selected or invalid option.", 400
    
  return render_template('index.html', song_path='', timestamp='', quarter='')
  
if __name__ == '__main__':
  app.run(debug=True)