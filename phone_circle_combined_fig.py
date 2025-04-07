from tools import MicrophoneCircle
from tools import Phone
import matplotlib.pyplot as plt

# Create plots
phone = Phone(position=[0.0, 0.0, 0.0], orientation=[0, -90, 0])
fig, ax = phone.plot_phone(plot=False, ax_lim=600)

mic_circle = MicrophoneCircle(center=[0, 0, 0], radius=500.0, n_mics=30, sphere=False)
fig, ax = mic_circle.plot_microphone_positions(plot=False, fig=fig, ax=ax)

# Show the combined plot
plt.show()
