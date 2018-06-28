"""
Rien de très intéressant à modifier ici. Va plutôt voir transforms.py
"""
import os

from kivy.properties import ObjectProperty, StringProperty, ListProperty

from kivy.uix.floatlayout import FloatLayout


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    path = StringProperty(os.path.dirname(os.path.dirname(__file__)))


class ColorDialog(FloatLayout):
    select = ObjectProperty(None)
    cancel = ObjectProperty(None)
    color = ListProperty(None)