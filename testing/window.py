#!C:/Users/Syeam/AppData/Local/Programs/Python/Python37/python.exe
# @author Syeam_Bin_Abdullah 
import kivy
import os
from kivy.app import App
from kivy.uix.label import Label 
from kivy.uix.gridlayout import GridLayout 
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.uix.video import Video

class MainMenu(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols = 2

        if os.path.isfile("details.txt"):
            with open("details.txt", "r") as f:
                d = f.read().split(",")
                prev_ip = d[0]
                prev_port = d[1]
                prev_usr = d[2]

        else:
            prev_ip = ''
            prev_port = ''
            prev_usr = ''

        self.add_widget(Label(text='IP:'))

        self.ip = TextInput(text=prev_ip, multiline=False)
        self.add_widget(self.ip)

        self.add_widget(Label(text='Port:'))

        self.port = TextInput(text=prev_port, multiline=False)
        self.add_widget(self.port)

        self.add_widget(Label(text='Username:'))

        self.usr = TextInput(text=prev_usr, multiline=False)
        self.add_widget(self.usr)

        self.join = Button(text="Join")
        self.join.bind(on_press=self.join_button) 
        self.add_widget(Label())
        self.add_widget(self.join)

    def join_button(self, instance):
       port = self.port.text
       ip = self.ip.text
       usr = self.usr.text

       print(f"Attempting to join {ip}:{port} as {usr}")

       with open("details.txt", "w") as f:
           f.write(f"{ip},{port},{usr}")

class TestApp(App):
    def build(self):
        return MainMenu()

if __name__ == "__main__":
    TestApp().run()


