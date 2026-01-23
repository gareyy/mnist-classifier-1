from tkinter import ttk
import tkinter as tk
from typing import Union
from frontend_components import DrawGrid, ButtonSuite

class Application(ttk.Frame):
    def __init__(
        self,
        master: Union[tk.Tk, tk.Widget],
        **kwargs
    ) -> None:
        super().__init__(
            master,
            **kwargs
        )
        self.grid()
        self.drawgrid = DrawGrid(self, (28, 28), (500, 500))
        self.drawgrid.grid(column=0, row=0)
        self.drawgrid.reset()
        self.button_frame = ButtonSuite(self, self.drawgrid.reset, self.drawgrid.reset, padding=0)
        self.button_frame.grid(column=1, row=0)

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root, padding=5)
    root.mainloop()
