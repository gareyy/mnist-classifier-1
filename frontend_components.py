import tkinter as tk
from tkinter import ttk
from typing import Callable, Union
import numpy as np

BLACK = "#000000"
WHITE = "#FFFFFF"
GREYMID = "#888888"
INIT_LABEL = "Press 'Guess' to guess the drawn number."

class AbstractGrid(tk.Canvas):
    """A type of tkinter Canvas that provides support for using the canvas as a
    grid (i.e. a collection of rows and columns)."""

    def __init__(
        self,
        master: Union[tk.Tk, tk.Widget],
        dimensions: tuple[int, int],
        size: tuple[int, int],
        **kwargs
    ) -> None:
        """Constructor for AbstractGrid.

        Parameters:
            master: The master frame for this Canvas.
            dimensions: (#rows, #columns)
            size: (width in pixels, height in pixels)
        """
        super().__init__(
            master,
            width=size[0] + 1,
            height=size[1] + 1,
            highlightthickness=0,
            **kwargs
        )
        self._size = size
        self.set_dimensions(dimensions)

    def set_dimensions(self, dimensions: tuple[int, int]) -> None:
        """Sets the dimensions of the grid.

        Parameters:
            dimensions: Dimensions of this grid as (#rows, #columns)
        """
        self._dimensions = dimensions

    def _get_cell_size(self) -> tuple[int, int]:
        """Returns the size of the cells (width, height) in pixels."""
        rows, cols = self._dimensions
        width, height = self._size
        return width // cols, height // rows

    def pixel_to_cell(self, x: int, y: int) -> tuple[int, int]:
        """Converts a pixel position to a cell position.

        Parameters:
            x: The x pixel position.
            y: The y pixel position.

        Returns:
            The (row, col) cell position.
        """
        cell_width, cell_height = self._get_cell_size()
        return y // cell_height, x // cell_width

    def _get_bbox(self, position: tuple[int, int]) -> tuple[int, int, int, int]:
        """Returns the bounding box of the given (row, col) position.

        Parameters:
            position: The (row, col) cell position.

        Returns:
            Bounding box for this position as (x_min, y_min, x_max, y_max).
        """
        row, col = position
        cell_width, cell_height = self._get_cell_size()
        x_min, y_min = col * cell_width, row * cell_height
        x_max, y_max = x_min + cell_width, y_min + cell_height
        return x_min, y_min, x_max, y_max

    def _get_midpoint(self, position: tuple[int, int]) -> tuple[int, int]:
        """Gets the graphics coordinates for the center of the cell at the
            given (row, col) position.

        Parameters:
            position: The (row, col) cell position.

        Returns:
            The x, y pixel position of the center of the cell.
        """
        row, col = position
        cell_width, cell_height = self._get_cell_size()
        x_pos = col * cell_width + cell_width // 2
        y_pos = row * cell_height + cell_height // 2
        return x_pos, y_pos

    def color_cell(self, position: tuple[int, int], color: str) -> None:
        """
        Colors the cell at the given (row, col) position with the specified
        color

        Parameters:
            position: The (row, col) cell position.
            color: The tkInter string corresponding to the desired color
        """
        self.create_rectangle(*self._get_bbox(position), fill=color, outline="gray10")

    def reset(self):
        """Clears all child widgets off the canvas."""
        self.delete("all")

    def bind_heldm1_callback(
        self, click_callback: Callable[[tuple[int, int]], None]
    ) -> None:
        """Binds click buttons 1 and 2 to a callable function that resides in
        an external location

        Parameters:
            click_callback (Callable[tuple[int, int], None]): Function to bind
                                                              To
        """

        #First define an intermediate function that converts mouse location to
        #canvas coordinates, so that it satisfies the condition that the 
        #_handle_click method uses row and column coordinates
        def _inner(event: tk.Event) -> None:
            """Intermediate function between the click event and the 
            _handle_click event.

            Parameters:
                event (tk.Event): The event to detect for and extract attributes
                                  from.
            
            Returns:
                None
            """
            coords = self.pixel_to_cell(event.x, event.y)
            click_callback(coords)

        #Then bind to intermediate function
        self.bind('<B1-Motion>', _inner)

class DrawGrid(AbstractGrid):
    def __init__(
        self,
        master: Union[tk.Tk, tk.Widget],
        dimensions: tuple[int, int],
        size: tuple[int, int],
    ) -> None:
        """Constructor for DrawGrid

        Parameters:
            master: The master frame for this Canvas.
            dimensions: (#rows, #columns)
            size: (width in pixels, height in pixels)
        """
        super().__init__(
            master,
            dimensions,
            size
        )
        self.griddata = np.zeros((size[0], size[1]), dtype=np.float32)
        self.bind_heldm1_callback(self.paint)

    def reset(self) -> None:
        super().reset()
        for i in range(self._dimensions[0]):
            for j in range(self._dimensions[1]):
                self.color_cell((i, j), BLACK)
        self.griddata = np.zeros((self._dimensions[0], self._dimensions[1]), np.float32)

    def paint(self, pos: tuple[int, int]) -> None:
        self.single_point_paint(pos)
        self.single_point_paint((pos[0] + 1, pos[1]))
        self.single_point_paint((pos[0], pos[1] + 1))
        self.single_point_paint((pos[0] + 1, pos[1] + 1))

    def single_point_paint(self, pos: tuple[int, int], add: float = 1.0, color: str = WHITE) -> None:
        if pos[0] < 0 or pos[0] >= self._dimensions[0] or pos[1] < 0 or pos[1] >= self._dimensions[1]:
            return
        self.color_cell(pos, color)
        self.griddata[pos[0], pos[1]] += add
        self.griddata[pos[0], pos[1]] = min(np.float32(1.0), self.griddata[pos[0]][pos[1]])

    def get_griddata(self) -> np.ndarray:
        return self.griddata

class ButtonSuite(ttk.Frame):
    def __init__(
        self,
        master: Union[tk.Tk, tk.Widget],
        guess_func: Callable,
        reset_func: Callable,
        **kwargs
    ) -> None:
        super().__init__(
            master,
            **kwargs
        )
        self.grid()
        self.guess_func = guess_func
        self.reset_func = reset_func
        ttk.Button(self, text="Clear", command=self.reset_func).grid(column=0, row=0)
        ttk.Button(self, text="Guess", command=self.guess_func).grid(column=0, row=1, pady=10)
        self.label = ttk.Label(self, text=INIT_LABEL, width=35)
        self.label.grid(column=0, row=2, pady=10)

    def update_text(self, new_text: str) -> None:
        self.label.configure(text=new_text)

