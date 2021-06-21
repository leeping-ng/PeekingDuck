# main palatte of peekingduck
CHAMPAGNE = [156, 223, 244]
BLIZZARD = [241, 232, 164]
VIOLET_BLUE = [188, 118, 119]
TOMATO = [77, 103, 255]
RED = [46, 68, 244]
PRIMARY_PALETTE = [CHAMPAGNE, BLIZZARD, TOMATO, VIOLET_BLUE, RED]

# constants for thin look
THIN = {
    'thickness': 1,
    'caps': True,
    'txt_bg': False
}

# constants for thick look
THICK = {
    'thickness': 2,
    'caps': False,
    'txt_bg': True
}

# constant to fill shapes in cv2. To be replace line thickness
FILLED = -1

# constants for font scale
SMALL_FONTSCALE = 0.5
NORMAL_FONTSCALE = 1
BIG_FONTSCALE = 2

# constants used for image manipulation
LOWER_SATURATION = 0.5

# constants for legend creation
LEGEND_BOX = {
    'alpha': 0.5,
    'beta': 0.5,
    'gamma': 0
}