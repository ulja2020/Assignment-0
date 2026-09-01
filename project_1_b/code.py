from tools import fridge_management
from tools import total_power_usage
import matplotlib.pyplot as plt

print("Code is running")

# Input
temperatures = [2, 4, 6, 4, 2, 6]

door_open = [False, False, True, False, True, False]

time = [0, 1, 2, 3, 4, 5]

power = []

# Call the fridge management function and store the results in the power list (instantaneous power values)
for i in range(len(temperatures)):
    result = fridge_management(temperatures[i], door_open[i])
    power.append(result)

# Calculate cumulative power usage
cumulative_power = []

total = 0

for x in power:
    total = total + x
    cumulative_power.append(total)

# Calculate total power usage
total_power = total_power_usage(power)

# Print total power used
print("Total power used:", total_power)

# Plot the instantaneous and cumulative power usage versus time
# assuming the task wants both graphs on the same plot

plt.plot(time, power, label="Instantaneous power")
plt.plot(time, cumulative_power, label="Cumulative power")

plt.xlabel("Time")
plt.ylabel("Power usage")
plt.legend()
plt.show()