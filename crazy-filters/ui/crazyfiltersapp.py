import os
from glob import glob
from os.path import join, dirname
from random import randint

import kivy

kivy.require('1.0.6')

from kivy.app import App
from kivy.logger import Logger
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.popup import Popup

from ui.button_transforms import ScatterCamera, ScatterPicture, TabPicture, TabCamera
from ui.dialogs import LoadDialog


class CrazyFiltersApp(App):
    def __init__(self, **kwargs):
        super(CrazyFiltersApp, self).__init__(**kwargs)
        self._camera = None
        self.widget_container = None
        # ESSAIE ! Paramètres modifiables ci-dessous
        self.widget_type = 'tab'  # essaie de changer ça :-)
        self._use_pictures = True  # True: charge les images dans le dossier images. Sinon camera seule.
        self._use_camera = True

    def open_image(self, filename):
        if self.widget_type == 'tab':
            tab_name = os.path.splitext(os.path.basename(filename))[0]
            pic = TabPicture(text=tab_name,
                             source=filename)
        else:
            pic = ScatterPicture(source=filename,
                                 rotation=randint(-30, 30),
                                 pos=(randint(10, self.widget_container.width - 10),
                                      randint(10, self.widget_container.height - 54)))
        self.widget_container.add_widget(pic)
        self.widget_container.switch_to(pic)

    def build(self):
        if self.widget_type == 'tab':
            self.widget_container = TabbedPanel(size_hint=(1, 0.9), do_default_tab=False)
            self.build_tabs()
            self.root.add_widget(self.widget_container)
        else:
            self.widget_container = self.root
            self.build_scatter()

    def build_scatter(self):
        if self._use_camera:
            try:
                self._camera = ScatterCamera(rotation=randint(-20, 20))
                self.widget_container.add_widget(self._camera)
            except AttributeError:
                print("Warning: could not start Camera. Trying to load pictures.")
                self._use_pictures = True
        # get any files into images directory
        if self._use_pictures:
            self.open_images_from_dir()

    def build_tabs(self):
        if self._use_camera:
            try:
                self._camera = TabCamera(text='Camera')
                self.widget_container.add_widget(self._camera)
            except kivy.lang.builder.BuilderException as e:
                print("Warning: could not start Camera.  Trying to load pictures instead. "
                      "Check that another app is not using your webcam.")
                self._camera = None
                self._use_pictures = True
        # get any files into images directory
        if self._use_pictures:
            self.open_images_from_dir()

    def open_images_from_dir(self):
        curdir = dirname(__file__)
        for filename in glob(join(curdir, '..', 'images', '*.[jpg][png]*')):
            try:
                self.open_image(filename)
            except Exception as e:
                Logger.exception('Open images: Unable to load <%s>' % filename)

    def on_stop(self):
        if self._camera is not None:
            self._camera.source._camera.stop()
            del self._camera.source._camera
        return True

    def show_load(self):
        content = LoadDialog(load=self.load,
                             cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path, filename):
        self.open_image(os.path.join(path, filename[0]))
        self.dismiss_popup()

    def dismiss_popup(self):
        self._popup.dismiss()

    def on_pause(self):
        return True
