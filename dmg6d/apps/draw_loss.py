import matplotlib.pyplot as plt

# 读取文件中的loss_all值
file_path = '../logs/train_ycb_trainset_20240111.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()

# 提取loss_all值
loss_all_values = []
for line in lines:
    if 'loss_all' in line:
        loss_all_values.append(float(line.split('loss_all ')[1].strip()))

# 生成epoch的序列
epochs = list(range(1, len(loss_all_values) + 1))

# 画图
plt.plot(epochs, loss_all_values, label='loss_all')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()
