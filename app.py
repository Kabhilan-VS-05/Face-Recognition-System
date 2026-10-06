import subprocess
from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html.txt')

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        subprocess.Popen(['python', 'recognizer.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return jsonify({"message": "Recognition Process Started!"}), 202
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)  # Disable auto-reloading loop
