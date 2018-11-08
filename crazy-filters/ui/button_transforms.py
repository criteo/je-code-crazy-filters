"""
Rien de très intéressant à modifier ici. Va plutôt voir transforms.py.
"""
import os
import time

from kivy.properties import StringProperty
from kivy.uix.tabbedpanel import TabbedPanelItem
from kivy.uix.scatter import Scatter
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.label import Label

from ui.dialogs import ColorDialog

import transforms as tf


class TransformerBehaviourMixin:
    """
    Inherit from this class to add the transform behaviour to it.
    The transform behaviour can be augmented by
     * defining new transform functions in transforms.py
     * adding buttons to the widget in the kv file
     * adding cases in the on_click_transform function.
    """

    def __init__(self, **kwargs):
        super(TransformerBehaviourMixin, self).__init__(**kwargs)
        self.tf_buttons = {}
        self.buttons = {}
        self.last_color = [1, 1, 1, 1]
        self._tick = [0,]

    def add_button(self, name, func, is_tf=True, is_toggle=True, **kwargs):
        """
        Create buttons handling different options
        and attaching to the given function.
        By default, the function is an image transforms
        and is applied through self.on_click_transform.
        For more specific functions, you need to define them
        in th class and use is_tf=False.

        :param name: Button label
        :param func: function to run on click
        :param is_tf: True if the function is a direct image transform.
        """
        if is_tf:
            self.tf_buttons[name] = (func, kwargs)
            btn = ToggleButton(text=name, on_press=self.on_click_transform)
        else:
            self.buttons[name] = func
            if is_toggle:
                btn = ToggleButton(text=name, on_press=func)
            else:
                btn = Button(text=name, on_press=func)

        btns = self.ids['buttons']
        btns.add_widget(btn)

    def on_click_transform(self, instance):
        image = self.ids['image']
        button_type = instance.text

        if button_type in self.tf_buttons:
            if 'ticking' in self.tf_buttons[button_type][1]:

                if instance.state != 'down':
                    self._tick[0] = 0

            image.on_click_transform(self.tf_buttons[button_type][0], **self.tf_buttons[button_type][1])


    def on_color_click(self, instance):
        if instance.state == 'down':
            clr_picker = ColorDialog(select=self.on_color_transform,
                                     cancel=self.dismiss_color,
                                     color=self.last_color)
            self._popup = Popup(title="Choose color", content=clr_picker,
                                size_hint=(0.9, 0.9))
            self._popup.open()
        else:  # remove from transform list
            self.ids['image'].on_click_transform(tf.colorize, color=self.last_color)

    def on_color_transform(self, value):
        image = self.ids['image']
        image.on_click_transform(tf.colorize, color=value)
        self.last_color = value
        self.dismiss_color()

    def dismiss_color(self):
        self._popup.dismiss()


class CameraTransformerMixin(TransformerBehaviourMixin):
    """
    Inherit from this class and a widget to add the camera
    and buttons to the widget.
    Button corresponds to transforms that can be applied
    to the camera input.
    """

    def __init__(self, **kwargs):
        super(CameraTransformerMixin, self).__init__(**kwargs)

    def build(self):
        # ESSAIE ! Ajouter les boutons pour la caméra ici
        self.add_button('Négatif', tf.invert_image)
        self.add_button('Sepia', tf.to_sepia)
        self.add_button('Carrousel', tf.carrousel_transfo, ticking=self._tick)
        self.add_button('Custom', tf.custom)
        self.add_button('Couleur', self.on_color_click, is_tf=False)
        self.add_button('Capture', self.capture, is_tf=False, is_toggle=False)
        self.source = self.ids['image']

    def capture(self, *args):
        """
        Exemple de fonction qui ne fonctionne que pour la caméra
        et qui n'est pas une transformation d'image.
        """
        camera = self.ids['image']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs('captures', exist_ok=True)
        img_path = "captures/IMG_{}.png".format(timestr)
        camera.export_to_png(img_path)
        popup = Popup(title='You just saved an image!',
                      content=Label(text='it was saved to {}'.format(img_path)),
                      size_hint=(None, None), size=(400, 200))
        popup.open()


class ImageTransformerMixin(TransformerBehaviourMixin):
    """
    Inherit from this class and a widget to add an image
    and transform buttons to the widget.
    The buttons correspond to transforms that can be applied
    to an image input.
    The image file is given with the `source` parameter.
    """
    source = StringProperty(None)

    def __init__(self, **kwargs):
        super(ImageTransformerMixin, self).__init__(**kwargs)

    def build(self):
        # Ajouter les boutons pour les images fixes ici
        self.add_button('Négatif', tf.invert_image)
        self.add_button('Sepia', tf.to_sepia)
        self.add_button('Répéter', tf.repeat)
        self.add_button('Custom', tf.custom)
        self.add_button('Cut', self.on_cut, is_tf=False)
        self.add_button('Couleur', self.on_color_click, is_tf=False)

    def on_cut(self, *args):
        """
        Exemple de fonction qui ne fonctionne que pour les images fixes.
        Elle ne fonctionne pas comme les autres transformations d'image car
        elle nécessite une entrée souris.
        """
        image = self.ids['image']
        image.on_cut()


"""
Example of widgets that implement the transform buttons.
Layout definition is in the .kv file
"""


class ScatterCamera(Scatter, CameraTransformerMixin):
    def __init__(self, **kwargs):
        super(ScatterCamera, self).__init__(**kwargs)
        self.build()


class ScatterPicture(Scatter, ImageTransformerMixin):
    def __init__(self, **kwargs):
        super(ScatterPicture, self).__init__(**kwargs)
        self.build()


class TabPicture(TabbedPanelItem, ImageTransformerMixin):
    def __init__(self, **kwargs):
        super(TabPicture, self).__init__(**kwargs)
        self.build()


class TabCamera(TabbedPanelItem, CameraTransformerMixin):
    def __init__(self, **kwargs):
        super(TabCamera, self).__init__(**kwargs)
        self.build()