# -*- coding: utf-8 -*-
"""
Created on Tue May  8 20:57:32 2018

@author: Yiyi.Zhang
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash()

app.layout = html.Div(
        children=[html.H1(children='Deblur Image'),
                  html.Div(children='''
                           A web application for debluring images'''),
                  dcc.Upload(
                          id='upload-image',
                          children=html.Div([
                                  'Drag and Drop or ',
                                  html.A('Select Files')]),
                          style={'width': '100%',
                                 'height': '60px',
                                 'lineHeight': '60px',
                                 'borderWidth': '1px',
                                 'borderStyle': 'dashed',
                                 'borderRadius': '5px',
                                 'textAlign': 'center',
                                 'margin': '10px'
                                  },
                          multiple=True),
                  html.Div(id='output-image-upload'),
])

def parse_contents(contents, filename):
    return html.Div([
        html.H5(filename),
        html.Img(src=contents),
        html.Hr(),
    ])


@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents'),
               Input('upload-image', 'filename'),
#               Input('upload-image', 'last_modified')
               ])
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n) for c, n in
            zip(list_of_contents, list_of_names)]
        return children


#from flask import Flask, render_template
#app = Flask(__name__)

#@app.route('/')
#def home():
#    return render_template('home.html')
#
#@app.route('/about/')
#def about():
#    return render_template('about.html')
#
if __name__=='__main__':
#    app.run(debug=True)
    app.run_server(debug=True)