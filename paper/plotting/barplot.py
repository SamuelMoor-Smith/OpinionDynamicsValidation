import matplotlib.pyplot as plt
import numpy as np

# Example data
# labels = ['lrscale-UK', 'imwbcnt-UK', 'trstun-Uk', 'pplfair-UK', 'rlgdgr-UK', 'happy-UK']
# zero = [0.1496744186, 0.3483786152, 0.1332103321, 0.1320836966, 0.1674235808, 0.1991095625]
# deffuant = [0.1497353331, 0.3484568562, 0.1332295447, 0.1322690788, 0.1674256273, 0.1992528748]
# hk = [0.1545587749, 0.3526395883, 0.1334262612, 0.1331143561, 0.1681395097, 0.2032689033]
# carpentras = [0.1483535458, 0.3498427735, 0.1333989384, 0.1333013181, 0.1724873833, 0.1983740521]
# duggins = [0.1494604811, 0.3531527799, 0.1335430046, 0.1327644841, 0.1664511313, 0.1818405876]

labels = ['imwbcnt-UK', 'stfdem-FI', 'trstun-SI', 'ppltrst-HU', 'happy-PI', 'rlgdgr-PI']
zero = [0.3483786152, 0.4771633051, 0.3058933583, 0.3265771812, 0.3006647673, 0.3013358779]
deffuant = [0.3492498927, 0.4772413793, 0.3097099199, 0.3274307812, 0.30735652, 0.3032951404]
hk = [0.3485735618, 0.4776997471, 0.3183971259, 0.3298483951, 0.3016968279, 0.3014374477]
carpentras = [0.3516377365, 0.4771478658, 0.3045434178, 0.3197907786, 0.3047125542, 0.2951609512]
duggins = [0.3501342154, 0.4750169645, 0.3073133609, 0.3275180421, 0.3007340417, 0.3119763147]

# X locations for groups
x = np.arange(len(labels))  # label locations
width = 0.15  # width of each bar

# Plotting each group with an offset
plt.bar(x-2*width, zero, width, color="#000000", label='Zero Model')
plt.bar(x-width,   deffuant, width, color="C0", label='Deffuant')
plt.bar(x,         hk, width, color="C1", label='HK Averaging')
plt.bar(x+width,   carpentras, width, color="C4", label='Carpentras')
plt.bar(x+2*width, duggins, width, color="C2", label='Duggins')

# Labels and title
plt.xlabel('ESS Dataset')
plt.ylabel('Scaled Loss')
plt.title('Model Performance Comparison')
plt.xticks(x, labels)  # Set the x-axis tick labels
plt.legend()

plt.tight_layout()
# plt.show()
plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")

