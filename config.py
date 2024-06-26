import torch

DOWNSAMPLE  = False
FASTSAM     = False #USE Ultralitics FastSAM instead of SAM
CACHE_MASKS = False #Do not open interactive prompt, just initialize the masks and cache them  #WARNING: EACH MASK IS ~2.4GB
MAX_SAMPLES = 400
MEDIUM_STAR_SIZE = 50 
MEDIUM_GREEN_RED_DOT_SIZE = 5
SMALL_STAR_SIZE = 10
SMALL_GREEN_RED_DOT_SIZE = 2

MEDIUM_DOT_SIZE_MODE = False
SMALL_DOT_SIZE_MODE = True
DOT_SIZE_TOGGLE = SMALL_DOT_SIZE_MODE # small dot size by default
GREEN_COLOR = '#00f700'
RED_COLOR = '#ff1919'

SAM_CHECKPOINT = 'sam_vit_h_4b8939.pth'
MODEL_TYPE     = "vit_h"
DEVICE         = "cuda" if torch.cuda.is_available() else 'cpu'

if DOT_SIZE_TOGGLE == MEDIUM_DOT_SIZE_MODE:
    LINEWIDTH = 1.25
else:
    LINEWIDTH = 0.5