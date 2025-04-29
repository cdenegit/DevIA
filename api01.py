from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "API de suma funcionando correctamente."

@app.route('/sumar', methods=['POST'])
def sumar():
    data = request.get_json()

    try:
        val1 = float(data.get('a', 0))
        val2 = float(data.get('b', 0))
        val3 = float(data.get('c', 0))
        total = val1 + val2 + val3

        return jsonify({
            'a': val1,
            'b': val2,
            'c': val3,
            'suma_total': total
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)