#!C:/Users/Syeam/AppData/Local/Programs/Python/Python37/python.exe
# @author Syeam_Bin_Abdullah 
import numpy as np
import cv2
from kivy.app import App
from kivy.core.window import Window as Win
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.gridlayout import GridLayout 
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label

W = 200
H = 150
images = []
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "

for i in range(0,26):
    img = cv2.imread(f'../letters/{i}.png') 
    ret = cv2.resize(img, (W, H))
    images.append(ret)

cv2.imshow('Preview', images[0])
cv2.waitKey(0)

print(np.shape(alphabet))
print(np.shape(alphabet[0]))

class LetterImg(Image):
  """
  Displays translated Auslan (fingerspelling)
  """
  def __init__(self, imgs=None, **kwargs):
      super(LetterImg, self).__init__(**kwargs)
      self.index = 0
      self.rst = False
      self.frame = np.zeros((H, W))
      self.imgs = imgs
      self.words = None
      Clock.schedule_interval(self.update, 0.75)

  # def update(self, dt):
      # self.ret, self.frame = self.capture.read()
  def insert(self, keywords):
    self.words = keywords

  # def update(self, dt=None, imgs=None, sent=None):
  def update(self, dt=None):
    if self.rst:
      # sent = None
      self.words = None
      self.index = 0
      self.frame = np.zeros((H, W))
      self.canvas.clear()
      self.rst = False

    if self.words is not None:
      letter = ''.join(self.words[self.index].upper().strip().split(' '))
      print(letter)
      img = self.imgs[alphabet.index(letter)]
      print(img)
    else: 
      print('no words...')
      img = np.zeros((H, W))
      self.index -=1

    if self.imgs is not None:
      self.frame = cv2.flip(img, 0)
      # img = np.reshape(img, (H, W))
      # print(f'Shape: {img.shape}')
    else:
      self.frame = np.zeros((H, W))

    # convert it to texture
    self.output = self.frame
    buf = self.output.tostring()
    image_texture = Texture.create(
        size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr')

    image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    # display image from the texture
    self.texture = image_texture
    print('done updating texture')
    self.index += 1

  def reset(self, instance):
    self.rst = True

#    def readimg():


class TranslateWin(GridLayout):

  def __init__(self, **kwargs):
      super().__init__(**kwargs)
      self.rows = 4

      self.letter_display = LetterImg(imgs=images)
      self.add_widget(self.letter_display)

      self.textbox = TextInput()
      self.add_widget(self.textbox)

      self.start = Button(text="Start", size_hint=(.5,.5))
      self.start.bind(on_press=self.translate)
      self.add_widget(self.start)

      self.resetbutton = Button(text="Reset", size_hint=(.5,.5))
      self.add_widget(self.resetbutton)
      self.resetbutton.bind(on_press=self.letter_display.reset)

  def translate(self, instance): # function for translating sentence input into Auslan fingerspelling
      print('called translate')
      sent = ''.join(self.textbox.text.strip().split(' '))
      print(sent)

      for i in sent:
        if i.upper() not in alphabet:
            print(f'Invalid input: {i}')

      if sent is not None:
        print('sent images')
        self.letter_display.insert(sent)
        print('called insert')


class TranslateApp(App):
  def build(self):
    self.menu = TranslateWin()
    return self.menu


if __name__ == '__main__':
  TranslateApp().run()
