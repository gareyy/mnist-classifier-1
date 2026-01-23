from tkinter import ttk
import tkinter as tk
from typing import Union

import torch
from frontend_components import DrawGrid, ButtonSuite
import frontend_components
from torch_train import NeuralNetwork

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class TorchApplication(ttk.Frame):
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
        self.button_frame = ButtonSuite(self, reset_func=self.reset, guess_func=self.guess, padding=0)
        self.button_frame.grid(column=1, row=0)
        self.model = NeuralNetwork()
        self.model.load_state_dict(torch.load("weights/model.dat", weights_only=True))
        self.model.eval()
        self.model.to(device)

    def reset(self) -> None:
        self.drawgrid.reset()
        self.button_frame.update_text(frontend_components.INIT_LABEL)

    def guess(self) -> None:
        griddata_as_torch = torch.from_numpy(self.drawgrid.get_griddata().reshape(1, 1, 28, 28))
        # educational note: model requires inputs of shape (N, 1, 28, 28), where N is the batch size
        # by reshaping to (1, 1, 28, 28), the input works with the model, otherwise it rejects it
        prediction = self.model(griddata_as_torch.to(device))
        print(prediction.argmax(1))

if __name__ == "__main__":
    root = tk.Tk()
    app = TorchApplication(root, padding=5)
    root.mainloop()
