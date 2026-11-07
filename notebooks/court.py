# court.py ~ thank you ChatGPT
"""Half-court diagram drawn in NBA tracking coordinates.

The hoop sits at the origin and **10 plot units == 1 foot** (NBA's loc_x/loc_y
are returned in tenths of a foot). Convert a shot before plotting it:

    x = dist_ft * 10 * sin(angle)
    y = dist_ft * 10 * cos(angle)
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc

def draw_half_court(ax=None, color='black', lw=2):
    if ax is None:
        ax = plt.gca()

    # Create the basketball hoop
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color, fill=False)
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color, fill=False)

    # Free throw top/bottom arcs
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180, linewidth=lw, color=color)
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0, linestyle='dashed', linewidth=lw, color=color)

    # Restricted zone
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw, color=color)

    # Three point line: straight corner lines at 22 ft, joined by a 23.75 ft
    # arc centered on the hoop. The arc meets the corner lines where
    # x = 220 -> y = sqrt(237.5^2 - 220^2) = 89.48 (22.13 degrees).
    corner_three_a = Rectangle((-220, -47.5), 0, 136.98, linewidth=lw, color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 136.98, linewidth=lw, color=color)
    three_arc = Arc((0, 0), 475, 475, theta1=22.13, theta2=157.87, linewidth=lw, color=color)

    # Half-court line (47 ft from the baseline, hoop 5.25 ft from the baseline)
    # with the center circle drawn as a semicircle on this side of the line.
    half_court = Rectangle((-250, 417.5), 500, 0, linewidth=lw, color=color)
    center_circle = Arc((0, 417.5), 120, 120, theta1=180, theta2=360,
                        linewidth=lw, color=color)

    # Add to plot
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a, corner_three_b,
                      three_arc, half_court, center_circle]

    for element in court_elements:
        ax.add_patch(element)

    # Set axis limits and hide axis
    ax.set_xlim(-320, 320)
    ax.set_ylim(-47.5, 422.5)

    # Plot every unit equally in both directions, otherwise circles render as
    # ellipses and equal distances look different vertically vs horizontally.
    ax.set_aspect('equal')

    ax.axis('off')

    return ax

