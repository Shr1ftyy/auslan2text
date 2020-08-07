#!C:/Users/Syeam/AppData/Local/Programs/Python/Python37/python.exe
# @author Syeam_Bin_Abdullah 
import numpy as np
import cv2
import time
from kivy.app import App
from kivy.core.window import Window as Win
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.gridlayout import GridLayout 
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label

images = []
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "

for i in range(0,26):
    img = cv2.imread(f'../testing/letters/{i}.png') 
    ret = cv2.resize(img, (200, 150))
    images.append(ret)

print(np.shape(alphabet))
print(np.shape(alphabet[0]))

class LetterImg(Image):
    """
    Displays translated Auslan (fingerspelling)
    """
    def __init__(self, fps=2, **kwargs):
        super(Camera, self).__init__(**kwargs)
        Clock.schedule_interval(self.update, 1.0 / fps)

    # def update(self, dt):
        # self.ret, self.frame = self.capture.read()
    def update(self, img=None, dt):
        if img is not None:
            self.frame = img
            # convert it to texture
            self.output = self.frame.
            buf = self.output.tostring()
            image_texture = Texture.create(
                size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')

            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture
            time.sleep(0.5)

#    def readimg():


class TranslatorWin(GridLayout):

    def __init__(self, **kwargs):
        self.rows = 4

        self.textbox = TextInput()
        self.button.bind(on_press=self.translate)
        self.add_widget(self.button)

        self.start = start(text="Start", size_hint=(.5,.5))
        self.add_widget(self.start)
        self.rest = rest(text="Reset", size_hint=(.5,.5))
        self.add_widget(self.rest)

        self.letter_display = LetterImg()

    def translate(self, sent): # function for translating sentence input into Auslan fingerspelling
        for i in sent:
            if i.upper() not in alphabet:
                print(f'Invalid input: {i}')

            else: 
                self.letter_display.update(images[alphabet.index(i.upper())])






# for i in range (0,26):
#     length = 103
#     width = 84
#     gap = i+16
#     try:
#         if i > 0:
#             alphabet.append(img[i*width+gap:i*width+gap+width, i*length+gap:i*length+gap+length,:])
#         else:
#             alphabet.append(img[:width, i:length,:])

#     except:
#         print('failed to add image %d'%(i))

# cv2.imshow('_', alphabet[3])
# print(np.shape(alphabet))
# cv2.waitKey(0)
