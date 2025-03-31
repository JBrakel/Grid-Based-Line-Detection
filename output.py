import tkinter as tk

class Output:

    def __init__(self):
        root = tk.Tk()
        self.screen_width = int(root.winfo_screenwidth()) if int(root.winfo_screenwidth()) < 3840 else int(root.winfo_screenwidth()/2)
        self.screen_height = int(root.winfo_screenheight())
        root.withdraw()
        self.factor = self.screen_width / (1024 * 2)
        self.window_width  = int(1024 * self.factor)
        self.window_height = int(576 * self.factor)

    def get_window_size(self):
        return [self.window_width, self.window_height]

    def center_output_one_window(self):
        self.x1 = ((self.screen_width - self.window_width)  // 2) - (self.window_width // 2)
        self.y1 = (self.screen_height - self.window_height) // 2
        self.x2 = ((self.screen_width - self.window_width)  // 2) + (self.window_width // 2)
        self.y2 = (self.screen_height - self.window_height) // 2
        self.x = int((self.x1 + self.x2)/2)
        self.y = int((self.y1 + self.y2)/2-self.screen_height/5)
        return [self.x, self.y]

    def center_output_two_windows(self):
        self.x1 = ((self.screen_width - self.window_width)  // 2) - (self.window_width // 2)
        self.y1 = (self.screen_height - self.window_height) // 2
        self.x2 = ((self.screen_width - self.window_width)  // 2) + (self.window_width // 2)
        self.y2 = (self.screen_height - self.window_height) // 2
        return [self.x1, self.y1, self.x2, self.y2]

