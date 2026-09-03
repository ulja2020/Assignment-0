from tools import fridge_management
from tools import total_power_usage


# Test fridge management function (using arbitrary values for temperature and door status)

assert fridge_management(2, False) == 0
assert fridge_management(2, True) == 1
assert fridge_management(5, False) == 2
assert fridge_management(5, True) == 3

print("Fridge management tests passed!")


# Test total power usage function (using arbitrary values of power)

power_values = [0, 1, 2, 3, 2, 1]

assert total_power_usage(power_values) == 9

print("Total power usage test passed!")
