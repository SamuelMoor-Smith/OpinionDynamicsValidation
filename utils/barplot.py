import matplotlib.pyplot as plt
import numpy as np

# Example data
labels = ['Zero', 'Deffuant', 'HK', 'Carpentras', 'Duggins']
group1 = [10, 15, 20, 25, 30, 35]
group2 = [12, 18, 22, 28, 30, 35]
group3 = [8, 14, 19, 23, 30, 35]
group4 = [8, 14, 20, 23, 30, 35]
group5 = [8, 14, 21, 23, 30, 35]

# X locations for groups
x = np.arange(len(labels))  # label locations
width = 0.15  # width of each bar

# Plotting each group with an offset
plt.bar(x-2*width, group1, width, color="#000000", label='Zero Model')
plt.bar(x-width,   group2, width, color="C0", label='Deffuant')
plt.bar(x,         group3, width, color="C1", label='HK Averaging')
plt.bar(x+width,   group4, width, color="C4", label='Carpentras')
plt.bar(x+2*width, group5, width, color="C2", label='Duggins')

# Labels and title
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Grouped Column Plot')
plt.xticks(x, labels)  # Set the x-axis tick labels
plt.legend()

plt.tight_layout()
plt.show()
