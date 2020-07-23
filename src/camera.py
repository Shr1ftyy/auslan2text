#!C:/Users/Syeam/AppData/Local/Programs/Python/Python37/python.exe
# @author Syeam_Bin_Abdullah 

from kivy.app import App
from kivy.core.window import Window as Win
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.gridlayout import GridLayout 
from kivy.uix.button import Button
from kivy.uix.label import Label
# from model import MobileNet
from tensorflow.keras.models import load_model
import numpy as np
import time
import cv2


class Camera(Image):
    """
    Gathers imagery using OpenCV as a backend
    """
    def __init__(self, capture, fps, **kwargs):
        super(Camera, self).__init__(**kwargs)
        self.capture = capture

        # initialization of rectangle
        self.RECTX = 100 # x position of rectangle
        self.RECTY = 100 # y position of rectangle
        self.RECTW = 224 # width of rectangle
        self.RECTH = 224 # height of rectangle
        self.RECTT = 2 # thickness of rectangle 
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        self.ret, self.frame = self.capture.read()
        if self.ret:
            # convert it to texture
            self.output = self.frame.copy()
            cv2.rectangle(self.output, (self.RECTX, self.RECTY), (self.RECTX+self.RECTW, self.RECTY+self.RECTH), (0,255,0), self.RECTT)
            buf1 = cv2.flip(self.output, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')

            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture


class CamMenu(GridLayout):
    """
    Camera Menu, contains UI elements such as buttons and label
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rows = 3 
        self.capture = cv2.VideoCapture(0)
        print(Win.size[1])
        self.cam = Camera(capture=self.capture, fps=30)
        self.add_widget(self.cam)

        self.label = Label(text="testing", size_hint=(.5,.3))
        self.add_widget(self.label)

        self.button = Button(text="Take Photo", size_hint=(.5,.5))
        self.button.bind(on_press=self.predict)
        self.add_widget(self.button)
        self.index = 2

    def save(self, instance):
        time.sleep(3)
        self.cam.update(None)
        self.img = self.cam.frame[self.cam.RECTY:self.cam.RECTH+self.cam.RECTY, 
                             self.cam.RECTX:self.cam.RECTW+self.cam.RECTX]
        print(self.img.shape)
        self.label.text = "LOL"
        cv2.imshow("_", self.img)
        cv2.waitKey(0)
        key = input('')
        if key.lower().strip() == 's':
            cv2.imwrite(f"./{self.index}.png", self.img)
            self.index += 1
        else:
            pass

    def predict(self, instance):
        self.img = self.cam.frame[self.cam.RECTY:self.cam.RECTH+self.cam.RECTY, 
                             self.cam.RECTX:self.cam.RECTW+self.cam.RECTX]

        pred = np.argmax(model.predict(np.expand_dims(np.array([cv2.resize(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), (28, 28))]), axis=3)/255.0))
        cv2.imshow('-', cv2.resize(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), (28, 28))) # change for finalized release
        cv2.waitKey(0)
        self.label.text = alphabet[pred]
 


class CamApp(App):
    def build(self):
        self.menu = CamMenu()
        return self.menu

    def on_stop(self):
        #without this, app will not exit even if the window is closed
        self.menu.capture.release()

if __name__ == '__main__':
    global model
    global alphabet
    alphabet="ABCDEFGHIKLMNOPQRSTUVWXYZ"
    # model = MobileNet(classes=26, idx=channels_last")
    model = load_model('../utils/test.h5')
    # model.compile(optimizer='adam', loss="categorical_crossentropy")
    CamApp().run()
