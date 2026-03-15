import numpy as np

def get_dmg(rgb: np.ndarray) -> float:
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]

    # White to Yellow (0-50)
    if g == 255 and r == 255 and b < 255:
        dmg = 50 * (255 - b) / 255
        return max(0, min(50, dmg))
    
    # Yellow to Orange (50-100)
    elif r == 255 and b == 0 and g >= 153:
        dmg = 50 + 50 * (255 - g) / 102
        return max(50, min(100, dmg))
    
    # Orange to Red (100-150)
    elif r == 255 and b == 0 and g < 153:
        dmg = 100 + 50 * (153 - g) / 153
        return max(100, min(150, dmg))
    
    # Red variants (150-200)
    elif g == 0 and b == 0 and r >= 191:
        dmg = 150 + 50 * (255 - r) / 64
        return max(150, min(200, dmg))
    
    # Darker Red (200-250)
    elif g == 0 and b == 0 and r >= 140 and r < 191:
        dmg = 200 + 50 * (191 - r) / 51
        return max(200, min(250, dmg))

    # Very Dark Red (250-300)
    elif g == 0 and b == 0 and r > 74 and r <= 140:
        dmg = 250 + 50 * (140 - r) / 66
        return max(250, min(300, dmg))
    
    # near dark (300+)
    elif r <= 74 and g == 0 and b == 0:
        dmg = 300 + (74 - r) / 74 * 51
        return max(300, min(351, dmg))
    
    # Default case - no damage detected
    else:
        return 0.0


