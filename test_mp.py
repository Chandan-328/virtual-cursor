import mediapipe as mp
print(f"MediaPipe version: {mp.__version__}")
try:
    print(f"Solutions: {mp.solutions}")
except AttributeError as e:
    print(f"Error accessing solutions: {e}")
    print(f"Dir(mp): {dir(mp)}")
