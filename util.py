import subprocess
import pretty_midi
import numpy as np
import torch
import os
from cvae import CVAE


TIME_STEP = 1/16
MAX_TIME = 16
TEMPO = 2 # beat/sec
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initModel():
  if CVAE.instance is None:
    model = CVAE().to(device)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))
    CVAE.instance = model

def midi_to_audio(midi_file_path, audio_file_path, sound_font_path="font.sf2"):
  # current_dir = os.path.dirname(os.path.abspath(__file__))
  # fluidsynth_path = os.path.join(current_dir, 'fluidsynth')
  subprocess.call(['fluidsynth', '-ni', sound_font_path, midi_file_path, '-F', audio_file_path, '-r', '44100'])
  print(f"Conversion complete: {audio_file_path}")

def vector_to_midi(vector, bps=2, time_step=TIME_STEP, max_time=MAX_TIME, ):
  num_pitches = 128
  beats = 16 // bps
  print(beats)
  num_time_steps = int(max_time / time_step)
  note_matrix = vector.reshape((num_time_steps, num_pitches))
  
  midi_data = pretty_midi.PrettyMIDI()
  instrument = pretty_midi.Instrument(program=0)  # Default to a piano instrument

  for pitch in range(num_pitches):
    # Get 1D list of time indices for each pitch where velocity > 0
    active_time_steps = np.where(note_matrix[:, pitch] > 0)[0]
    if len(active_time_steps) == 0: continue

    start_idx = min(round(active_time_steps[0] / beats) * beats, 255)
    for i in range(len(active_time_steps)):
      # Check if picth is not continous at this time index meaning the note turns off
      if active_time_steps[i-1] != active_time_steps[i] - 1:
        # Add note to notes array
        start_time = start_idx * time_step
        end_time = active_time_steps[i-1] * time_step
        if active_time_steps[i-1] - start_idx < 4: continue 
        velocity = int(note_matrix[start_idx, pitch])
        note = pretty_midi.Note(velocity=velocity,pitch=pitch,start=start_time,end=end_time)
        instrument.notes.append(note)
        # Find next note
        start_idx = min(round(active_time_steps[i] / beats) * beats, 255)

    # Add final note
    start_time = start_idx * time_step
    end_time = active_time_steps[-1] * time_step
    velocity = int(note_matrix[start_idx, pitch])
    note = pretty_midi.Note(velocity=velocity,pitch=pitch,start=start_time,end=end_time)
    instrument.notes.append(note)

  midi_data.instruments.append(instrument)
  return midi_data

def generate(model, emotion):
  latent_dim = 4
  z = torch.randn(1, latent_dim - 2).to(device)
  z_with_emotion = torch.cat((z, emotion), dim=1).to(device)
  return torch.round(model.decode(z_with_emotion) * 5)

def generateMusic(q: str):
  if CVAE.instance is None:
    print('Model in unitialized')
    return
  # Generate vector with Convolutional Variational Auto-encoder
  print('Generating vector')
  emotion_mapping = {
    'Q1': [1, 1],
    'Q2': [0, 1],
    'Q3': [0, 0],
    'Q4': [1, 0]
  }
  emotion = emotion_mapping[q]
  vec = generate(
    CVAE.instance,
    torch.tensor(emotion, device=device).to(torch.float32).view(1, 2).to(device)
  ).cpu().detach().numpy()[0][0]
  # Generate midi file
  print('Generating midi')
  tempo = 4 if q == 'Q1' or q == 'Q2' else 2
  vec = vec / np.max(vec) * 127
  print(np.max(vec))
  midi = vector_to_midi(vec.flatten(), bps=tempo)
  current_dir = os.path.dirname(os.path.abspath(__file__))
  midi_path = os.path.join(current_dir, 'static/music/output.mid')
  midi.write(midi_path)
  # Generate wav music file
  print('Generating wave')
  music_path = os.path.join(current_dir, 'static/music/output.wav')
  sound_font_path = os.path.join(current_dir, 'font.sf2')
  midi_to_audio(midi_path, music_path, sound_font_path=sound_font_path)


