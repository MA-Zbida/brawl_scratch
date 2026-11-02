from typing import Optional, Tuple

def get_rgb(pixel) -> Optional[Tuple]:
    return (int(pixel[2]), int(pixel[1]), int(pixel[0]))