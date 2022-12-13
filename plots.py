import pandas as pd
import matplotlib.pyplot as plt

attention_table = pd.read_csv("result/attention_training_reward.txt", sep =',', header = None)
performer_table = pd.read_csv("result/performer_training_reward.txt", sep =',', header = None)
cnn_table = pd.read_csv("result/CNN_training_reward.txt", sep =',', header = None)

print(cnn_table)

f = plt.figure()
f.set_figwidth(5)
f.set_figheight(4)


# Reward comparison between performers and regular attention
# plt.plot(performer_table[0], performer_table[2], label = "Performers")
# plt.plot(attention_table[0], attention_table[2], label = "Regular Attention")
# plt.plot(range(32500), [1] * 32500, label = "Threshold=1")
# plt.legend()
# plt.xlabel("episode")
# plt.ylabel("training reward")
# plt.title("Reward for performers and regular attention")
# plt.show()

plt.plot(performer_table[0], performer_table[2], label = "ViT with Performers")
plt.plot(cnn_table[0], cnn_table[2], label = "CNN")
plt.plot(range(32500), [1] * 32500, label = "Threshold=1")
plt.legend()
plt.xlabel("episode")
plt.ylabel("training reward")
plt.title("Reward for ViT-performers and CNN")
plt.show()