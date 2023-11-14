from tkinter import Tk, Label, Button, filedialog

class UserInterface:
    def __init__(self):
        self.search_space = {}
        self.selected_dataset = None

    def load_dataset(self):
        root = Tk()
        root.withdraw()

        filepath = filedialog.askopenfilename()
        self.selected_dataset = filepath

    def define_search_space(self):
        # Code for defining the search space
        pass

    def display_gui(self):
        # Code for creating and displaying the GUI
        pass

    def get_selected_dataset(self):
        return self.selected_dataset