import numpy as np

def get_stock(rgb):
    r = int(rgb[0])
    g = int(rgb[1])
    b = int(rgb[2])

    # Red flash: self stock lost
    if r >= 210 and g <= 50 and b <= 50:
        return -1

    # Cyan at the player's calibrated HUD point is game-credited stock gain.
    # Do not reinterpret it as every opponent death or impose a guessed hit timer.
    if r <= 80 and g >= 210 and b >= 210:
        return 1

    return 0
