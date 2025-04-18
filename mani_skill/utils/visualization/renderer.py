import sys

from matplotlib import pyplot as plt


class ImageRenderer:
    def __init__(self, wait_for_button_press=True):
        """
        Create a very light-weight image renderer.

        Args:
            wait_for_button_press (bool): If True, each call to this renderer will pause the process until the user presses any key.
            event_handler: Code to run given an event / button press. If None the default is mapping 'escape' and 'q' to sys.exit(0)
        """
        self._image = None
        self.last_event = None
        self.wait_for_button_press = wait_for_button_press
        self.pressed_keys = set()

    def key_press_handler(self, event):
        self.last_event = event
        self.pressed_keys.add(event.key)
        if event.key in ["q", "escape"]:
            sys.exit(0)

    def key_release_handler(self, event):
        if event.key in self.pressed_keys:
            self.pressed_keys.remove(event.key)

    def __call__(self, buffer):
        if not self._image:
            plt.ion()
            self.fig = plt.figure()
            self._image = plt.imshow(buffer, animated=True)
            self.fig.canvas.mpl_connect("key_press_event", self.key_press_handler)
            self.fig.canvas.mpl_connect("key_release_event", self.key_release_handler)
        else:
            self._image.set_data(buffer)
        if self.wait_for_button_press:
            plt.waitforbuttonpress()
        else:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        plt.draw()

    def __del__(self):
        self.close()

    def close(self):
        plt.ioff()
        plt.close()
