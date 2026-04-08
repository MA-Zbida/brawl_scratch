UI_REGIONS = {
    "stock": (1863, 45),
    "op": (1795, 80),
    "agent": (1886, 80),
}

# Normalized platform bounds (red blocked rectangle in your sketch).
PLATFORM_BOUNDS = {
    "x_min": 0.315,
    "x_max": 0.683,
    "y_min": 0.5527,
    "y_max": 0.8149,
}

# Outer green rectangle in your sketch. Target sampling uses this area,
# then explicitly excludes the red PLATFORM_BOUNDS hole.
LOCOMOTION_GREEN_BOUNDS = {
    "x_min": 0.160,
    "x_max": 0.780,
    "y_min": 0.200,
    "y_max": 0.800,
}

# Safety margin that expands the red blocked area during sampling so targets
# do not land on walls/edges where movement is ambiguous.
LOCOMOTION_RED_EXCLUSION_MARGIN = 0.01
