import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 定义超参数
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# 图像处理相关参数
IMG_SIZE = 64
CHANNELS = 3  # RGB
CLASSES = 2  # 猫和狗
# 替换为实际训练数据集路径
TRAIN_DIR = "datasets/cats_and_dogs_filtered/train"
# 替换为实际测试数据集路径
TEST_DIR = "datasets/cats_and_dogs_filtered/validation"


# 卷积神经网络结构
class ConvolutionalNeuralNetwork:
    def __init__(self):
        # 初始化卷积层权重和偏置
        self.W1 = np.random.randn(3, 3, 3, 16) * np.sqrt(2. / (3 * 3 * 3))
        self.b1 = np.zeros((1, 16))
        self.W2 = np.random.randn(3, 3, 16, 32) * np.sqrt(2. / (3 * 3 * 16))
        self.b2 = np.zeros((1, 32))
        # 初始化全连接层权重和偏置
        self.W3 = np.random.randn(16 * 16 * 32, 128) * np.sqrt(2. / (16 * 16 * 32))
        self.b3 = np.zeros((1, 128))
        self.W4 = np.random.randn(128, CLASSES) * np.sqrt(2. / 128)
        self.b4 = np.zeros((1, CLASSES))

    def conv_forward(self, X, W, b, stride=1, padding=1):
        N, H, W_in, C = X.shape
        F, F, C, K = W.shape
        H_out = 1 + (H + 2 * padding - F) // stride
        W_out = 1 + (W_in + 2 * padding - F) // stride
        out = np.zeros((N, H_out, W_out, K))
        X_pad = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
        for n in range(N):
            for h in range(H_out):
                for w in range(W_out):
                    for k in range(K):
                        h_start = h * stride
                        h_end = h_start + F
                        w_start = w * stride
                        w_end = w_start + F
                        x_slice = X_pad[n, h_start:h_end, w_start:w_end, :]
                        out[n, h, w, k] = np.sum(x_slice * W[:, :, :, k]) + b[0, k]
        return out

    def leaky_relu_forward(self, X, alpha=0.01):
        return np.where(X > 0, X, alpha * X)

    def max_pool_forward(self, X, pool_size=2, stride=2):
        N, H, W, C = X.shape
        H_out = (H - pool_size) // stride + 1
        W_out = (W - pool_size) // stride + 1
        out = np.zeros((N, H_out, W_out, C))
        for n in range(N):
            for h in range(H_out):
                for w in range(W_out):
                    for c in range(C):
                        h_start = h * stride
                        h_end = h_start + pool_size
                        w_start = w * stride
                        w_end = w_start + pool_size
                        x_slice = X[n, h_start:h_end, w_start:w_end, c]
                        out[n, h, w, c] = np.max(x_slice)
        return out

    def forward(self, X):
        self.conv1 = self.conv_forward(X, self.W1, self.b1)
        self.relu1 = self.leaky_relu_forward(self.conv1)
        self.pool1 = self.max_pool_forward(self.relu1)
        self.conv2 = self.conv_forward(self.pool1, self.W2, self.b2)
        self.relu2 = self.leaky_relu_forward(self.conv2)
        self.pool2 = self.max_pool_forward(self.relu2)
        self.flatten = self.pool2.reshape(self.pool2.shape[0], -1)
        self.fc1 = np.dot(self.flatten, self.W3) + self.b3
        self.relu3 = self.leaky_relu_forward(self.fc1)
        self.fc2 = np.dot(self.relu3, self.W4) + self.b4
        exp_scores = np.exp(self.fc2)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def conv_backward(self, d_out, X, W, b, stride=1, padding=1):
        N, H, W_in, C = X.shape
        F, F, C, K = W.shape
        H_out = 1 + (H + 2 * padding - F) // stride
        W_out = 1 + (W_in + 2 * padding - F) // stride
        X_pad = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
        d_X_pad = np.zeros_like(X_pad)
        d_W = np.zeros_like(W)
        d_b = np.zeros_like(b)
        for n in range(N):
            for h in range(H_out):
                for w in range(W_out):
                    for k in range(K):
                        h_start = h * stride
                        h_end = h_start + F
                        w_start = w * stride
                        w_end = w_start + F
                        x_slice = X_pad[n, h_start:h_end, w_start:w_end, :]
                        d_X_pad[n, h_start:h_end, w_start:w_end, :] += d_out[n, h, w, k] * W[:, :, :, k]
                        d_W[:, :, :, k] += d_out[n, h, w, k] * x_slice
                        d_b[0, k] += d_out[n, h, w, k]
        d_X = d_X_pad[:, padding:-padding, padding:-padding, :]
        return d_X, d_W, d_b

    def leaky_relu_backward(self, d_out, X, alpha=0.01):
        d_X = np.where(X > 0, d_out, alpha * d_out)
        return d_X

    def max_pool_backward(self, d_out, X, pool_size=2, stride=2):
        N, H, W, C = X.shape
        H_out = (H - pool_size) // stride + 1
        W_out = (W - pool_size) // stride + 1
        d_X = np.zeros_like(X)
        for n in range(N):
            for h in range(H_out):
                for w in range(W_out):
                    for c in range(C):
                        h_start = h * stride
                        h_end = h_start + pool_size
                        w_start = w * stride
                        w_end = w_start + pool_size
                        x_slice = X[n, h_start:h_end, w_start:w_end, c]
                        max_idx = np.unravel_index(np.argmax(x_slice), x_slice.shape)
                        d_X[n, h_start + max_idx[0], w_start + max_idx[1], c] = d_out[n, h, w, c]
        return d_X

    def backward(self, X, y, probs):
        num_examples = X.shape[0]
        delta4 = probs.copy()
        delta4[range(num_examples), y] -= 1
        d_W4 = np.dot(self.relu3.T, delta4)
        d_b4 = np.sum(delta4, axis=0, keepdims=True)
        delta3 = np.dot(delta4, self.W4.T) * (self.relu3 > 0)
        d_W3 = np.dot(self.flatten.T, delta3)
        d_b3 = np.sum(delta3, axis=0, keepdims=True)
        delta_flatten = np.dot(delta3, self.W3.T)
        delta_pool2 = delta_flatten.reshape(self.pool2.shape)
        delta_relu2 = self.max_pool_backward(delta_pool2, self.relu2)
        delta_conv2 = self.leaky_relu_backward(delta_relu2, self.conv2)
        d_X2, d_W2, d_b2 = self.conv_backward(delta_conv2, self.pool1, self.W2, self.b2)
        delta_pool1 = d_X2
        delta_relu1 = self.max_pool_backward(delta_pool1, self.relu1)
        delta_conv1 = self.leaky_relu_backward(delta_relu1, self.conv1)
        _, d_W1, d_b1 = self.conv_backward(delta_conv1, X, self.W1, self.b1)
        # 更新参数
        self.W4 -= LEARNING_RATE * d_W4
        self.b4 -= LEARNING_RATE * d_b4
        self.W3 -= LEARNING_RATE * d_W3
        self.b3 -= LEARNING_RATE * d_b3
        self.W2 -= LEARNING_RATE * d_W2
        self.b2 -= LEARNING_RATE * d_b2
        self.W1 -= LEARNING_RATE * d_W1
        self.b1 -= LEARNING_RATE * d_b1

    def compute_loss(self, probs, y):
        num_examples = probs.shape[0]
        correct_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(correct_logprobs)
        return 1. / num_examples * data_loss

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def save_model(self, filename):
        np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3, W4=self.W4,
                 b4=self.b4)


# 数据处理函数
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0  # 归一化
    return img


def load_dataset(data_dir):
    images = []
    labels = []

    for label, category in enumerate(['cats', 'dogs']):
        category_path = os.path.join(data_dir, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            try:
                img = preprocess_image(img_path)
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    # 转换为NumPy数组并打乱顺序
    images = np.array(images)
    labels = np.array(labels)

    combined = list(zip(images, labels))
    random.shuffle(combined)
    images[:], labels[:] = zip(*combined)

    return images, labels


# 计算评价指标
def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


# 主程序
if __name__ == "__main__":
    # 加载数据集
    print("Loading datasets...")
    X_train, y_train = load_dataset(TRAIN_DIR)
    X_test, y_test = load_dataset(TEST_DIR)

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # 创建卷积神经网络模型
    model = ConvolutionalNeuralNetwork()

    # 训练模型
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    best_loss = np.inf

    start_time = time.time()

    for epoch in range(EPOCHS):
        # 按批次训练
        epoch_loss = []
        for i in range(0, X_train.shape[0], BATCH_SIZE):
            X_batch = X_train[i:i + BATCH_SIZE]
            y_batch = y_train[i:i + BATCH_SIZE]

            probs = model.forward(X_batch)
            loss = model.compute_loss(probs, y_batch)
            epoch_loss.append(loss)
            model.backward(X_batch, y_batch, probs)

        # 计算测试损失
        test_probs = model.forward(X_test)
        test_loss = model.compute_loss(test_probs, y_test)

        # 计算训练集和测试集的准确率
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        train_losses.append(np.mean(epoch_loss))
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_losses[-1]:.4f} - Test Loss: {test_losses[-1]:.4f} - Train Acc: {train_accuracy:.4f} - Test Acc: {test_accuracy:.4f}")

        # 保存最佳模型
        if test_loss < best_loss:
            best_loss = test_loss
            best_weights = {
                'W1': model.W1.copy(),
                'b1': model.b1.copy(),
                'W2': model.W2.copy(),
                'b2': model.b2.copy(),
                'W3': model.W3.copy(),
                'b3': model.b3.copy(),
                'W4': model.W4.copy(),
                'b4': model.b4.copy()
            }

    end_time = time.time()
    print(f"Training completed in {(end_time - start_time):.2f} seconds")

    # 加载最佳权重
    model.W1 = best_weights['W1']
    model.b1 = best_weights['b1']
    model.W2 = best_weights['W2']
    model.b2 = best_weights['b2']
    model.W3 = best_weights['W3']
    model.b3 = best_weights['b3']
    model.W4 = best_weights['W4']
    model.b4 = best_weights['b4']

    # 保存模型
    model.save_model('best_model_cnn.npz')
    print("Model saved as best_model_cnn.npz")

    # 评估模型
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.show()

    # 绘制准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.show()
