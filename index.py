from flask import Flask, request
import metrics

app = Flask(__name__)


@app.route('/getParameters', methods=['POST'])
def getParameters():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'there is no file'
        file = request.files['file']
        file.save(file.filename)
        return metrics.getMetrics(file.filename)


if __name__ == "__main__":
    app.run()
