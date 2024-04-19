#prictor.py

from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt 

import sys
import cv2
import statistics
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from segment_anything import sam_model_registry, SamPredictor

from config import *
from helpers import *
import workbook


sys.path.append("..")

try:
    matplotlib.use('Qt5Agg')
except:
    matplotlib.use('TkAgg')

plt.rcParams['keymap.grid'].remove('g')
plt.rcParams['keymap.home'].remove('r')


fast_sam = FastSAM()
fast_sam.to(device=DEVICE)  

                  



names  = np.load("samples.npy", allow_pickle=True)
labels = np.load("labels.npy", allow_pickle=True)

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process inputs for continuing work and providing a name.")
    parser.add_argument("--continue", dest="continue_previous", action="store_true", help="Continue previous work")
    parser.add_argument("name", type=str, default = None, nargs="?", help="The name to associate with the work")
    args = parser.parse_args()

    # Call the function with the parsed arguments.
    c = workbook.open_workbook(args.continue_previous, args.name)
else:
    c = workbook.open_workbook()


## start looping through samples: 
while c < MAX_SAMPLES:
    workbook.update_sample(c)
    msk      = []        # masks for each samples
    gp       = []        # green points
    rp       = []        # red points
    label    = labels[c] # GT for sample c
    rmv      = False
    mask     = 0     
    co       = 0
    bs       = 0
    score    = []
    round    = [0,0]
    stdx     = []
    stdy     = []
    ng       = []
    nr       = []
    green    = []
    red      = []
    greenx   = []
    redx     = []
    greeny   = []
    redy     = []
    label    = label == 1 #
    image    = names[c]   # samples c

    if len(image.shape) == 2:
        image = cv2.cvtColor((np.array(((image + 1) / 2) * 255, dtype='uint8')), cv2.COLOR_GRAY2RGB)
       
    

    while True:
        s                          = 0  # this is for the score
        count                      = 1  # to count the score max
        lessfive                   = 0
        current_color              = 'green'
        dot_size_toggle            = SMALL_DOT_SIZE_MODE # default will be small dot, not medium
        current_star_size          = SMALL_STAR_SIZE
        current_green_red_dot_size = SMALL_GREEN_RED_DOT_SIZE

        fig, ax = plt.subplots(1, 3, figsize=(15, 7))
        if green and red:
            ax[0].plot(greenx, greeny, 'go', markersize=5)
            ax[1].plot(greenx, greeny, 'go', markersize=5)
            ax[0].plot(redx,   redy,   'ro', markersize=5)
            ax[1].plot(redx,   redy,   'ro', markersize=5)
            plt.draw()


        def onclose(event):
            fig.canvas.stop_event_loop()
            fig.canvas.mpl_disconnect(cid)


        def onclick(event):
            global count
            global green
            global red
            global greenx
            global redx
            global greeny
            global redy
            global label
            global mask
            global lessfive
            global s

            if event.xdata is not None and event.ydata is not None:

                x, y = int(event.xdata), int(event.ydata)
                print(not x)
                print(not y)

                if event.button is MouseButton.LEFT:
                    if current_color == 'green':

                        green.append((x, y))
                        greenx.append(x)

                        greeny.append(y)
                        ax[0].plot(x, y, 'go', markersize=current_green_red_dot_size, color = GREEN_COLOR)
                        ax[1].plot(x, y, 'go', markersize=current_green_red_dot_size, color = GREEN_COLOR)
                        plt.draw()

                    else:
                        red.append((x, y))
                        redx.append(x)

                        redy.append(y)
                        ax[0].plot(x, y, 'ro', markersize=current_green_red_dot_size, color = RED_COLOR)
                        ax[1].plot(x, y, 'ro', markersize=current_green_red_dot_size, color = RED_COLOR)
                        plt.draw()

                elif event.button is MouseButton.RIGHT:
                    green, red, greenx, greeny, redx, redy = delete_point(green, red, current_color, x, y, ax, greenx, greeny, redx, redy)

                if green and red:
                    print("green:", green)
                    print("red:", red)

                    input_point = np.concatenate((green, red))
                    input_label = np.concatenate(([1] * len(green), [0] * len(red)))


                    FastSAM_input_point = input_point.tolist()
                    FastSAM_input_label = input_label.tolist()
                    results = fast_sam(
                                source=image,
                                device=DEVICE,
                                retina_masks=True,
                                imgsz=1024,
                                conf=0.5,
                                iou=0.6)
                    prompt_process = FastSAMPrompt(image, results, device=DEVICE)
                    masks = prompt_process.point_prompt(points=FastSAM_input_point, pointlabel=FastSAM_input_label)
                    mask = masks[0].masks.data

                    ax[2].clear()
                    ax[2].imshow(image)
                    show_mask(mask, ax[2])
                    intersection = (mask & label).sum()
                    union = (mask | label).sum()
                    if intersection == 0:
                        s = 0
                    else:
                        s = intersection / union

                    show_points(input_point, input_label, ax[2], marker_size = current_star_size)
                    msg = ""

                    if len(score[round[0]:]) == 0:
                        maxx = 0
                    else:
                        maxx = max(score[round[0]:])
                        print("maxx",maxx)
                    score.append(s)
                    gp.append(np.multiply(green, 1))

                    rp.append(np.multiply(red, 1))
                    ng.append(len(greenx))
                    nr.append(len(redx))
                    grx = np.concatenate([greenx, redx])
                    gry = np.concatenate([greeny, redy])

                    stdx.append(statistics.pstdev(grx.astype(int).tolist()))
                    stdy.append(statistics.pstdev(gry.astype(int).tolist()))
                    print("up count", count)
                    if maxx >= s:
                        print("inside",count)
                        if count >= 10:
                            lessfive += 1
                        else:
                            count += 1
                    elif maxx < s:

                        count = 1
                    if lessfive == 1:
                        maxx = 0
                        count=1
                        round[0] = len(np.array(score))
                        msg = " (round 2) "
                    plt.title(f"Score: {(intersection / union):.3f}" + msg, fontsize=13)
                    ## saving masks, scores, points and other stats: 
                    msk.append(np.multiply(mask, 5))
                    print("less than best score", lessfive)
                    print("scores:", score)
                    if lessfive == 1:
                        lessfive += 1
                        for line in ax[0].lines:
                            line.set_data([], [])
                        for line in ax[1].lines:
                            line.set_data([], [])
                        green = []
                        red = []
                        greenx = []
                        redx = []
                        greeny = []
                        redy = []
                        plt.draw()
                        ax[2].clear()
                        ax[2].imshow(image)
                        show_mask(mask, ax[2])
                        count = 1
                        print("below count", count)
                        plt.title("No better score is achieved in the last 5 attempts. Start round 2 from scratch")
                    elif lessfive == 3:
                        round[1]=len(score)-round[0]
                        print("The window closed because you did not achieve a better score after 5 consecutive clicks in the 2nd round")
                        plt.close()


        # Create a function to toggle between green and red dots
        def toggle_color(event):
            global green
            global red
            global greenx
            global redx
            global greeny
            global redy
            global current_color
            global count
            global current_star_size
            global current_green_red_dot_size
            global dot_size_toggle
            
            if event.key == 'g':
                current_color = 'green'
                print("Switched to GREEN dot mode.")

            elif event.key == 'r':
                current_color = 'red'
                print("Switched to RED dot mode.")
            elif event.key == ' ':
                for line in ax[0].lines:
                    line.set_data([], [])
                for line in ax[1].lines:
                    line.set_data([], [])
                green = []
                red = []
                greenx = []
                redx = []
                greeny = []
                redy = []
                plt.draw()
                ax[2].clear()
                ax[2].imshow(image)
                show_mask(mask, ax[2])
                count = 1
                print("below count", count)
            elif event.key == 'z':
                dot_size_toggle = not dot_size_toggle
                
                if dot_size_toggle == SMALL_DOT_SIZE_MODE:
                    # true => smaller dot size
                    current_star_size = SMALL_STAR_SIZE
                    current_green_red_dot_size = SMALL_GREEN_RED_DOT_SIZE
                    print("Switched to SMALL DOT SIZE mode.")
                else:
                    # false => default dot size
                    current_star_size = MEDIUM_STAR_SIZE
                    current_green_red_dot_size = MEDIUM_GREEN_RED_DOT_SIZE
                    print("Switched to MEDIUM DOT SIZE mode.")
                
                

        # Create a figure and display the image

        a = ax[0].plot()
        b = ax[1].plot()
        ax[0].imshow(image)
        ax[1].imshow(label)
        # Connect mouse click and keyboard key events
        fig.canvas.mpl_connect('button_press_event', onclick)

        fig.canvas.mpl_connect('key_press_event', toggle_color)
        fig.canvas.mpl_connect('key_press_event', toggle_color)


        cid = fig.canvas.mpl_connect('close_event', onclose)
        fig.show()  # this call does not block on my system
        fig.canvas.start_event_loop()  # block here until window closed
        break
 

    workbook.save_metrics(c, score, ng, nr, stdx, stdy, gp, rp, msk, round)
    workbook.update_survey()

    contin = input("do u want to continue? press y if you want to continue or anyting otherwise ")
    if not contin == 'y':
        workbook.save_workbook()
        break

    c += 1
    print("Sample:", c)
