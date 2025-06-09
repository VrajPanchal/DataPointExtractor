import numpy as np
import matplotlib.pyplot as plt

# Generate x-axis values (density) in log space
density = np.logspace(0, 1, 100)  # 10^0 = 1 to 10^1 = 10

# Generate example curves with similar shapes
gap_black = 20 * (1 - (density / 10)**1.5)
gap_black[gap_black < 0] = 0

gap_blue = 10 * (1 - (density / 8)**1.5)
gap_blue[gap_blue < 0] = 0

gap_green = 7 * (1 - (density / 7)**1.5)
gap_green[gap_green < 0] = 0

gap_red = 5 * (1 - (density / 6)**1.5)
gap_red[gap_red < 0] = 0

# Setup the plot
plt.figure(figsize=(6, 6))
plt.xscale('log')

# Plot colored curves with open circle markers
plt.plot(density, gap_black, 'ko-', markerfacecolor='none')
plt.plot(density, gap_blue, 'bo-', markerfacecolor='none')
plt.plot(density, gap_green, 'go-', markerfacecolor='none')
plt.plot(density, gap_red, 'ro-', markerfacecolor='none')

# Plot the gray line (approx from 10 eV at 1 g/cm³ to 0 eV at 6 g/cm³)
plt.plot([1, 6], [10, 0], color='gray')

# Set axis labels
plt.xlabel("Density (g cm$^{-3}$)")
plt.ylabel("Electronic energy gap (eV)")

# Set axis limits
plt.xlim(1, 10)
plt.ylim(0, 20)

# Remove top and right spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set custom ticks for log scale
ax.set_xticks([1, 2, 4, 6, 10])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.tick_params(which='minor', bottom=False)

plt.grid(False)
plt.tight_layout()
plt.show()
