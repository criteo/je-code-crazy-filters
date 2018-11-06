"""
Rien de très intéressant à modifier ici. Va plutôt voir transforms.py
"""
import cv2
import numpy as np

from kivy.graphics.texture import Texture
from kivy.properties import BooleanProperty, ObjectProperty
from kivy.core.image import ImageData
from kivy.uix.camera import Camera
from kivy.uix.image import Image

from ui.touchtracer import Touchtracer
import transforms as tf


class CutMixin(Touchtracer):
    is_cut_state = BooleanProperty(False)

    def on_cut(self):
        self.is_cut_state = not self.is_cut_state

    def check_transform_touch(self, touch):
        """
        We need to update the touch to get to scatter-coordinates for
        touch tracer to work correctly.
        :param touch:
        :return:
        """
        x, y = touch.x, touch.y

        # if the touch isnt on the image we do nothing
        if not self.collide_point(*self.to_local(x, y)):
            return False

        # let the child widgets handle the event if they want
        touch.push()
        touch.apply_transform_2d(self.to_local)
        return True

    def on_touch_move(self, touch):
        if self.is_cut_state:
            if self.check_transform_touch(touch):
                Touchtracer.on_touch_move(self, touch)
                touch.pop()
                return True
        return False

    def on_touch_down(self, touch):
        if self.is_cut_state:
            if self.check_transform_touch(touch):
                Touchtracer.on_touch_down(self, touch)
                touch.pop()
                return True
        return False

    def on_touch_up(self, touch):
        if self.is_cut_state:
            if self.check_transform_touch(touch):
                Touchtracer.on_touch_up(self, touch)
                touch.pop()
                return True
        return False


class TextureTransformMixin:
    tf_texture = ObjectProperty(None)

    def __init__(self, **kwargs):
        self.texwidth = 0
        self.texheight = 0
        self.texfmt = 'rgba'
        self.transforms = {}

    def on_click_transform(self, transform_func, **kwargs):
        if transform_func in self.transforms:
            self.transforms.pop(transform_func)
        else:
            self.transforms[transform_func] = kwargs
        self.init_texture()

    def init_texture(self):
        pass

    def apply_image_filter(self, image_filter_func, **kwargs):
        """

        :param image_filter_func: a function that takes an image and returns an image of the exact same size and shape.
        :param kwargs: parameters of the filter (optionally)
        :return:
        """
        pass

    def set_frame(self, buf):
        buf = cv2.flip(buf, 0)
        im = ImageData(width=self.texwidth,
                       height=self.texheight,
                       fmt=self.texfmt,
                       data=buf.reshape(self.texheight * self.texwidth * len(self.texfmt)),
                       flip_vertical=False)
        # now blit the array
        self.tf_texture.blit_data(im)

        def get_frame(self):
            pass

    def apply_all_transforms(self):
        buf = self.get_frame()
        for image_transform, kwargs in self.transforms.items():
            if 'ticking' in kwargs:
                kwargs['ticking'][0] += 1
            if 'use_alpha' in kwargs:
                buf = image_transform(buf, **kwargs).astype(np.uint8)
            else:
                buf = image_transform(buf[:, :, :3], **kwargs).astype(np.uint8)
            if len(buf.shape) == 2:
                buf = buf.reshape(buf.shape[0], buf.shape[1], 1)
                buf = np.repeat(buf, 4, axis=2)
                buf[:, :, 3] = 255
            elif buf.shape[2] == 3:
                buf = np.dstack((buf, np.zeros((self.texheight, self.texwidth, 1), np.uint8)))
                buf[:, :, 3] = 255
        self.set_frame(buf)


class CameraTransform(Camera, TextureTransformMixin):
    def __init__(self, **kwargs):
        super(CameraTransform, self).__init__(**kwargs)
        self.tf_texture = self.texture

    def init_texture(self):
        if len(self.transforms) == 0:
            self.tf_texture = self._camera.texture
        else:
            self.tf_texture = Texture.create(size=self.texture.size, colorfmt=self.texture.colorfmt)
        texture = self._camera.texture
        self.texwidth = texture.size[0]
        self.texheight = texture.size[1]
        self.texfmt = texture.colorfmt

    def get_frame(self):
        texture = self._camera.texture
        # initialize the array with the buffer values
        frame = np.fromstring(texture.pixels,
                              dtype=np.uint8,
                              count=self.texwidth * self.texheight * len(self.texfmt))
        buf = frame.reshape(self.texheight, self.texwidth, len(self.texfmt))
        return buf

    def on_tex(self, *args):
        if len(self.transforms) > 0:
            self.apply_all_transforms()
        self.canvas.ask_update()


class ImageTransform(CutMixin, Image, TextureTransformMixin):
    def __init__(self, **kwargs):
        super(ImageTransform, self).__init__(**kwargs)
        self.tf_texture = self.texture
        self.contours = []

    def init_texture(self):
        self.reload()
        if len(self.transforms) > 0:
            texture = self.texture
            self.texwidth = texture.size[0]
            self.texheight = texture.size[1]
            self.tf_texture = Texture.create(size=self.texture.size, colorfmt=self.texfmt)

            self.apply_all_transforms()
            self.texture = self.tf_texture
            self.canvas.ask_update()

    def get_frame(self):
        texture = self.texture
        # initialize the array with the buffer values
        frame = np.fromstring(texture.pixels,
                              dtype=np.uint8,
                              count=self.texwidth * self.texheight * len(self.texfmt))
        buf = frame.reshape(self.texheight, self.texwidth, len(self.texfmt))
        return buf

    def update_area(self, lines):
        # somehow contours are not in the right image reference
        img_resize_ratio = self.texture_size[0] / self.norm_image_size[0]
        x_offset = (self.width - self.norm_image_size[0]) // 2
        contours = [[((l.points[idx] - self.x - x_offset) * img_resize_ratio,
                      (self.top - l.points[idx + 1]) * img_resize_ratio)
                     for idx in range(0, len(l.points), 2)]
                    for l in lines]
        # append to list of transforms
        self.on_click_transform(tf.extract_contour, contours=contours, use_alpha=True)

