from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt 

import cv2
import statistics
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

import helpers
import config

from flask import Flask
from flask_caching import Cache
cache_config = {
    "DEBUG": True,                    # some Flask specific configs
    "CACHE_TYPE": "FileSystemCache",  # Flask-Caching related configs
    "CACHE_DIR": "cache",             # the path where the cache will be stored
}
app = Flask(__name__)
# tell Flask to use the above defined config
app.config.from_mapping(cache_config)
cache = Cache(app)


fast_sam = FastSAM()
fast_sam.to(device=config.DEVICE)  

@cache.memoize(0)
def get_mask_results(image):
    results = fast_sam(
        source=image,
        device=config.DEVICE,
        retina_masks=True,
        imgsz=1024,
        conf=0.5,
        iou=0.6
    )
    return results

class SampleAnnotator:
    def __init__(self, c, labels, names):
        cache.clear() #fix strange bug with masks being carried over
        self.c      = c
        self.msk    = []
        self.gp     = []
        self.rp     = []
        self.label  = labels[c]
        self.rmv    = False
        self.mask   = 0
        self.co     = 0
        self.bs     = 0
        self.score  = []
        self.attempt  = [0, 0]
        self.stdx   = []
        self.stdy   = []
        self.ng     = []
        self.nr     = []
        self.green  = []
        self.red    = []
        self.greenx = []
        self.redx   = []
        self.greeny = []
        self.redy   = []
        self.label  = self.label == 1
        self.image  = names[c]
        self.fig, self.ax = plt.subplots(1, 3, figsize=(15, 7))

        if len(self.image.shape) == 2:
            self.image = cv2.cvtColor((np.array(((self.image + 1) / 2) * 255, dtype='uint8')), cv2.COLOR_GRAY2RGB)
        if config.DOWNSAMPLE:
            self.image = helpers.downsample(self.image)
            self.label = helpers.downsample(self.label, True)

    def run(self):
        while True:
            self.annotate_sample()
            break

    def perform_sam(self):
        print("green:", self.green)
        print("red:", self.red)

        input_point = np.concatenate((self.green, self.red))
        input_label = np.concatenate(([1] * len(self.green), [0] * len(self.red)))


        FastSAM_input_point = input_point.tolist()
        FastSAM_input_label = input_label.tolist()


        results = get_mask_results(self.image)
        prompt_process = FastSAMPrompt(self.image, results, device=config.DEVICE)
        masks = prompt_process.point_prompt(points=FastSAM_input_point, pointlabel=FastSAM_input_label)
        mask = masks[0].masks.data
        mask = mask.numpy()

        self.ax[2].clear()
        self.ax[2].imshow(self.image)
        helpers.show_mask(mask, self.ax[2])
        intersection = (mask & self.label).sum()
        union = (mask | self.label).sum()
        if intersection == 0:
            s = 0
        else:
            s = intersection / union

        helpers.show_points(input_point, input_label, self.ax[2], marker_size = self.current_star_size)
        msg = ""

        if len(self.score[self.attempt[0]:]) == 0:
            maxx = 0
        else:
            maxx = max(self.score[self.attempt[0]:])
            print("maxx",maxx)
        self.score.append(s)
        self.gp.append(np.multiply(self.green, 1))

        self.rp.append(np.multiply(self.red, 1))
        self.ng.append(len(self.greenx))
        self.nr.append(len(self.redx))
        grx = np.concatenate([self.greenx, self.redx])
        gry = np.concatenate([self.greeny, self.redy])

        self.stdx.append(statistics.pstdev(grx.astype(int).tolist()))
        self.stdy.append(statistics.pstdev(gry.astype(int).tolist()))
        print("up count", self.count)
        if maxx >= s:
            print("inside",self.count)
            if self.count >= 10:
                self.lessfive += 1
            else:
                self.count += 1
        elif maxx < s:

            self.count = 1
        if self.lessfive == 1:
            maxx = 0
            self.count=1
            self.attempt[0] = len(np.array(self.score))
            msg = " (attempt 2) "
        plt.title(f"Score: {(intersection / union):.3f}" + msg, fontsize=13)
        ## saving masks, scores, points and other stats: 
        self.msk.append(np.multiply(mask, 5))
        print("less than best score", self.lessfive)
        print("scores:", self.score)
        if self.lessfive == 1:
            self.lessfive += 1
            for line in self.ax[0].lines:
                line.set_data([], [])
            for line in self.ax[1].lines:
                line.set_data([], [])
            self.green = []
            self.red = []
            self.greenx = []
            self.redx = []
            self.greeny = []
            self.redy = []
            plt.draw()
            self.ax[2].clear()
            self.ax[2].imshow(self.image)
            helpers.show_mask(mask, self.ax[2])
            self.count = 1
            print("below count", self.count)
            plt.title("No better score is achieved in the last 5 attempts. Start attempt 2 from scratch")
        elif self.lessfive == 3:
            self.attempt[1]=len(self.score)-self.attempt[0]
            print("The window closed because you did not achieve a better score after 5 consecutive clicks in the 2nd attempt")
            plt.close()

    def onclose(self, event):
        self.fig.canvas.stop_event_loop()
        self.fig.canvas.mpl_disconnect(self.cid)

    def onclick(self, event):

        if event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)

        if event.button is MouseButton.LEFT:
            if self.current_color == 'green':

                self.green.append((x, y))
                self.greenx.append(x)

                self.greeny.append(y)
                self.ax[0].plot(x, y, 'go', markersize=self.current_green_red_dot_size)
                self.ax[1].plot(x, y, 'go', markersize=self.current_green_red_dot_size)
                plt.draw()

            else:
                self.red.append((x, y))
                self.redx.append(x)

                self.redy.append(y)
                self.ax[0].plot(x, y, 'ro', markersize=self.current_green_red_dot_size)
                self.ax[1].plot(x, y, 'ro', markersize=self.current_green_red_dot_size)
                plt.draw()

        elif event.button is MouseButton.RIGHT:
            self.delete_point(x, y)

        if self.green and self.red:
            #plt.pause(0.1) #give time for the plot to update
            self.perform_sam()

    def toggle_color(self, event):
        if event.key == 'g':
            self.current_color = 'green'
            print("Switched to GREEN dot mode.")

        elif event.key == 'r':
            self.current_color = 'red'
            print("Switched to RED dot mode.")
        elif event.key == ' ':
            for line in self.ax[0].lines:
                line.set_data([], [])
            for line in self.ax[1].lines:
                line.set_data([], [])
            self.green = []
            self.red = []
            self.greenx = []
            self.redx = []
            self.greeny = []
            self.redy = []
            plt.draw()
            self.ax[2].clear()
            self.ax[2].imshow(self.image)
            helpers.show_mask(self.mask, self.ax[2])
            self.count = 1
            print("below count", self.count)
        elif event.key == 'z':
            self.dot_size_toggle = not self.dot_size_toggle
            
            if self.dot_size_toggle == config.SMALL_DOT_SIZE_MODE:
                # true => smaller dot size
                self.current_star_size = config.SMALL_STAR_SIZE
                self.current_green_red_dot_size = config.SMALL_GREEN_RED_DOT_SIZE
                print("Switched to SMALL DOT SIZE mode.")
            else:
                # false => default dot size
                self.current_star_size = config.MEDIUM_STAR_SIZE
                self.current_green_red_dot_size = config.MEDIUM_GREEN_RED_DOT_SIZE
                print("Switched to MEDIUM DOT SIZE mode.")

    def delete_point(self,x, y):
        if not self.green and not self.red:
            print("no points to delete")
        elif self.green:
            print(self.current_color)
            if self.current_color == 'green':
                indx = helpers.closetn((x, y), self.green)
                print(indx)
                for line in self.ax[0].lines:
                    if len(line.get_xdata()) > 0:
                        if line.get_xdata()[0] == self.green[indx][0] and line.get_ydata()[0] == self.green[indx][1]:

                            line.set_data([], [])
                            break
                for line in self.ax[1].lines:
                    if len(line.get_xdata()) > 0:
                        if line.get_xdata()[0] == self.green[indx][0] and line.get_ydata()[0] == self.green[indx][1]:
                            line.set_data([], [])
                            break
                del self.green[indx]
                del self.greenx[indx]
                del self.greeny[indx]
                plt.draw()
            elif self.red:
                print("delete red")
                print(self.current_color)
                indx = helpers.closetn((x, y), self.red)
                print(indx)

                for line in self.ax[0].lines:
                    if len(line.get_xdata()) > 0:
                        print()
                        if line.get_xdata()[0] == self.red[indx][0] and line.get_ydata()[0] == self.red[indx][1]:
                            line.set_data([], [])
                            break
                for line in self.ax[1].lines:
                    if len(line.get_xdata()) > 0:
                        if line.get_xdata()[0] == self.red[indx][0] and line.get_ydata()[0] == self.red[indx][1]:
                            line.set_data([], [])
                            break
                del self.red[indx]
                del self.redx[indx]
                del self.redy[indx]
                plt.draw()

    def annotate_sample(self):
        self.s = 0
        self.count = 1
        self.lessfive = 0
        self.current_color = 'green'
        self.dot_size_toggle = config.SMALL_DOT_SIZE_MODE
        self.current_star_size = config.SMALL_STAR_SIZE
        self.current_green_red_dot_size = config.SMALL_GREEN_RED_DOT_SIZE

        
        if self.green and self.red:
            self.ax[0].plot(self.greenx, self.greeny, 'go', markersize=5)
            self.ax[1].plot(self.greenx, self.greeny, 'go', markersize=5)
            self.ax[0].plot(self.redx, self.redy, 'ro', markersize=5)
            self.ax[1].plot(self.redx, self.redy, 'ro', markersize=5)
            plt.draw()

        # Create a figure and display the image
        a = self.ax[0].plot()
        b = self.ax[1].plot()
        self.ax[0].imshow(self.image)
        self.ax[1].imshow(self.label)
        # Connect mouse click and keyboard key events
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.toggle_color)
        self.cid = self.fig.canvas.mpl_connect('close_event', self.onclose)
        self.fig.show()
        self.fig.canvas.start_event_loop()