# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 04:35:49 2018

@author: Yiyi.Zhang
"""

from flask import Flask
 
app = Flask(__name__)
 
@app.route("/")
def home():
    return "Hello World!"
 
app.run(debug=True)