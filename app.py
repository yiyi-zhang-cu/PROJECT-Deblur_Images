# -*- coding: utf-8 -*-

# Import modules
#from __future__ import print_function
#import tensorflow as tf
#import numpy as np
#from model_tf import deblur_model
#import argparse
#from utils import load_images, load_own_images, deprocess_image, preprocess_image
#import os
#import h5py
#import matplotlib.pyplot as plt
#import plotly.plotly as py
#
#parser = argparse.ArgumentParser(description="deblur train")
#parser.add_argument("--g_input_size", help="Generator input size of the image", default=256,type=int)
#parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
#parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
#parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
#parser.add_argument('--n_downsampling', type=int, default=2, help='# of downsampling in generator')
#parser.add_argument('--n_blocks_gen', type=int, default=9, help='# of res block in generator')
#parser.add_argument('--d_input_size', type=int, default=256, help='Generator input size')
#parser.add_argument('--kernel_size', type=int, default=4, help='kernel size factor in discriminator')
#parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
#parser.add_argument('--LAMBDA_A', default=100000, type=int, help='The lambda for preceptual loss')
#parser.add_argument('--g_train_num', default=0, type=int, help='Train the generator for x epoch before adding discriminator')
#
#param = parser.parse_args(args='')
#tf.reset_default_graph()
#model = deblur_model(param)

#def parse_contents(contents, filename):
#    return html.Div([
#        html.H5(filename),
#        html.Img(src=contents),
#        html.Hr(),
#    ])

from flask import Flask, render_template, redirect
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

server = Flask(__name__)
app = dash.Dash(__name__, server=server, 
               url_base_pathname='/Home'
)
app.css.append_css({"external_url":"./static/css/main.css"})

app.config.supress_callback_exceptions = True
app.layout = html.Div(children=[
        html.Div([html.H1(children='Deblur Image',
                style={'background-color': '#DFB887',
                       'height': '35px',
                       'width': '100%',
                       'opacity': '.9',
                       'margin-bottom': '10px',
                       'margin': '0',
                       'font-size': '1.7em',
                       'color': '#fff',
                       'text-transform': 'uppercase',
                       'float': 'left'}),]),
        html.Div(children='''A web application for debluring images'''),
        html.Div(children=[
            dcc.Upload(id='upload-image',
                   children=html.Div(['Drag and Drop or ',html.A('Select a File')]),
                   style={'width': '100%',
                          'height': '40px',
                          'lineHeight': '40px',
                          'borderWidth': '1px',
                          'borderStyle': 'dashed',
                          'borderRadius': '5px',
                          'textAlign': 'center',
                          'margin': '10px'},
#                   multiple=True
                 ),
         ]),
        html.Div(id='output-image-upload'),
        html.Div(id='output-image-deblur'),
])


@app.callback(
        Output('output-image-upload', 'children'),
        [Input('upload-image', 'contents'),
         Input('upload-image', 'filename'),]
)
def input_image(content, filename):
#    if list_of_contents is not None:
#        children = [
#            parse_contents(list_of_contents, list_of_names) for c, n in
#            zip(list_of_contents, list_of_names)]
#        return children
    return html.Div([
           html.H5(filename),
           html.Img(src=content, style={'width': '30%','float': 'center'}),
           html.Hr(),
           html.Button('Click to Deblur', id='deblur-button', n_clicks=0),
    ])
    
@app.callback(Output('output-image-deblur', 'children'),
              [Input('deblur-button','n_clicks')])
def deblur_image(n_clicks):
     image_filename = './static/images/DebluredImage.png'
     if n_clicks > 0:
         return html.Img(src=image_filename,style={'width': '30%'})

#@app.callback(Output('output-image-deblur', 'children'),
#              [Input('upload-image', 'filename'),
#               ])
#def deblur_image():
#    data = load_own_images('./deblur_4990/images/own', n_images=1)
#    children=[model.generate(test_data=data, batch_size=1, trained_model='Deblur_1525567719',
#                   customized=True,save=False)]
#    return children

@server.route('/')
def home():
    return redirect('/Home')

@server.route('/ContactUs/')
def about():
    return render_template('about.html')
#
if __name__=='__main__':
    server.run(debug=True)
#    app.run_server(debug=True)