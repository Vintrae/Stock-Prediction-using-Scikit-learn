from flask import Flask, render_template, request, redirect
import sklearn as sk
import pandas as pd
import yfinance as yf
import matplotlib

app = Flask(__name__)

def predictPrice(data):
    pass

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route("/add_stock", methods=['GET'])
def add_stock():
    print(request.args.get('stock'))
    return render_template('index.html', var='hello')
