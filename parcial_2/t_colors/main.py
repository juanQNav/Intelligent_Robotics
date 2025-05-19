#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (ColorSensor,)
from pybricks.parameters import Port
from pybricks.tools import wait

# This program requires LEGO EV3 MicroPython v2.0 or higher.
# Click "Open user guide" on the EV3 extension tab for more information.


# Create your objects here.
ev3 = EV3Brick()
sensor = ColorSensor(Port.S3)


def show_rgb(color_nombre):
    # Leer los valores RGB
    rgb = sensor.rgb()
    # show screen
    ev3.screen.print("Color: {}".format(color_nombre))
    ev3.screen.print("R={}, G={}, B={}".format(rgb[0], rgb[1], rgb[2]))
    wait(2000)
    ev3.screen.clear()


# Write your program here.
ev3.speaker.beep()

# instructions
ev3.screen.print("Place the Green sheet")
wait(3000)
show_rgb("Green")

ev3.screen.print("Place the Yellow sheet")
wait(3000)
show_rgb("Yellow")

ev3.screen.print("Place the Pink sheet")
wait(3000)
show_rgb("Pink")

ev3.screen.print("Place the White sheet")
wait(3000)
show_rgb("White")

ev3.screen.print("Sensing complete")
