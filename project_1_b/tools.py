def fridge_management(temperature, door_open):
    """
    Decide the refrigerator power usage level from 0 to 3.

    Assumptions:
    The function uses a temperature inside the fridge provided as input (as a list of different temperatures)
    The function uses fridge door condition provided as input. 
    The door condition is represented as a Boolean, where True means "open" and False means "closed".
    The threshold is 3 degrees Celsius inside the fridge.
    The fridge uses more power when the temperature is higher than 3, because it needs to cool more.
    The fridge uses less power when the temperature is already cold. 
    When the door is open, power usage increases, because warm air enters the fridge.
    Power usage levels are assumed to be instantaneous power values, not cumulative.

    Level 0 (no/very low power): temperature is below 3 degrees and the door is closed.
    Level 1 (low power): temperature is below 3 degrees and the door is open.
    Level 2 (medium power): temperature is 3 degrees or higher and the door is closed.
    Level 3 (high power): temperature is 3 degrees or higher and the door is open.
    """

# Based on input temperature and door status, return the corresponding power usage level
    if temperature < 3 and door_open == False:
        return 0

    elif temperature < 3 and door_open == True:
        return 1

    elif temperature >= 3 and door_open == False:
        return 2

    else:
        return 3


def total_power_usage(power_values):
    """
    Calculate the total power usage from a list of instantaneous power values.
    """

    total = 0

    for x in power_values:
        total = total + x

    return total
