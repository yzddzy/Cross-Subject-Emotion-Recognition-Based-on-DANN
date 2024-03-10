import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 域泛化DANN模型
class DANNModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(DANNModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        class_logits = self.class_classifier(features)

        return class_logits

# 训练函数
def train_dann(model, source_loader, target_loader, num_epochs, lr):
    criterion_class = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for (source_data, source_labels), (target_data, _) in zip(source_loader, target_loader):
            source_data, source_labels, target_data = source_data.to(device), source_labels.to(device), target_data.to(device)

            class_logits = model(source_data)

            class_loss = criterion_class(class_logits, source_labels)

            total_loss = class_loss 

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

# 测试函数
def test_dann(model, target_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in target_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data) 

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

def load_data(file_path):
    data = np.load(file_path)
    return data

# 参数
input_size = 310  
hidden_size1 = 100
hidden_size2 = 50
num_classes = 3
num_epochs = 10
lr = 0.001

dann_results = []

# GPU加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 留一交叉验证
for test_subject in tqdm(range(1, 13), desc="Testing Subjects"):
    test_data = load_data(f'dataset/{test_subject}/data.npy')
    test_labels = load_data(f'dataset/{test_subject}/label.npy')

    train_data = []
    train_labels = []

    for train_subject in range(1, 13):
        if train_subject != test_subject:
            train_data.append(load_data(f'dataset/{train_subject}/data.npy'))
            train_labels.append(load_data(f'dataset/{train_subject}/label.npy'))

    train_data = np.vstack(train_data)
    train_labels = np.hstack(train_labels)

    train_data = train_data.reshape(train_data.shape[0], -1)
    test_data = test_data.reshape(test_data.shape[0], -1)

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    train_data_tensor = torch.FloatTensor(train_data)
    train_labels_tensor = torch.LongTensor(train_labels)
    test_data_tensor = torch.FloatTensor(test_data)
    test_labels_tensor = torch.LongTensor(test_labels)

    source_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    target_dataset = TensorDataset(test_data_tensor, test_labels_tensor)

    source_loader = DataLoader(source_dataset, batch_size=32, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True)

    dann_model = DANNModel(input_size=input_size, hidden_size1=hidden_size1, hidden_size2=hidden_size2, num_classes=num_classes).to(device)

    train_dann(dann_model, source_loader, target_loader, num_epochs, lr)

    dann_accuracy = test_dann(dann_model, target_loader)
    dann_results.append(dann_accuracy)

    #t-SNE降维可视化
    if test_subject == 1:
        dann_model.eval()
        dann_features = []
        dann_labels = []

        with torch.no_grad():
            for data, labels in source_loader:
                data = data.to(device)
                features = dann_model.feature_extractor(data)
                dann_features.append(features.cpu().numpy())
                dann_labels.append(labels.cpu().numpy())

        dann_features = np.concatenate(dann_features, axis=0)
        dann_labels = np.concatenate(dann_labels, axis=0)

        tsne = TSNE(n_components=2, random_state=42)
        dann_features_tsne = tsne.fit_transform(dann_features)

        train_data_tsne = tsne.fit_transform(train_data)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(train_data_tsne[:, 0], train_data_tsne[:, 1], c=train_labels)
        plt.title('Original Features (t-SNE)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')

        plt.subplot(1, 2, 2)
        plt.scatter(dann_features_tsne[:, 0], dann_features_tsne[:, 1], c=dann_labels)
        plt.title('DANN Features (t-SNE)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')

        plt.tight_layout()
        plt.show()

        dann_model.eval()
        dann_features = []
        dann_labels = []

        label_offset = 0
        with torch.no_grad():
            slabel_counter = 0
            for data, labels in source_loader:
                data = data.to(device)
                features = dann_model.feature_extractor(data)
                dann_features.append(features.cpu().numpy())

                l = (slabel_counter // 2526) + 1
                sdomain_labels = torch.ones(labels.shape[0],).to(device) * l
                slabel_counter += source_loader.batch_size
                dann_labels.append(sdomain_labels.cpu().numpy())

        dann_features = np.concatenate(dann_features, axis=0)
        print(dann_features.shape)
        dann_labels = np.concatenate(dann_labels, axis=0)
        print(dann_labels.shape)

        tsne = TSNE(n_components=2, random_state=42)
        dann_features_tsne = tsne.fit_transform(dann_features)

        train_data_tsne = tsne.fit_transform(train_data)

        unique_labels = np.unique(dann_labels)

        color_list = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'royalblue']

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        for i, label in enumerate(unique_labels):
            plt.scatter(train_data_tsne[dann_labels == label, 0], train_data_tsne[dann_labels == label, 1], label=f"Domain {label}", color=color_list[i % len(color_list)])
        plt.title('Original Features (t-SNE)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1), ncol=1)

        plt.subplot(1, 2, 2)
        for i, label in enumerate(unique_labels):
            plt.scatter(dann_features_tsne[dann_labels == label, 0], dann_features_tsne[dann_labels == label, 1], label=f"Domain {label}", color=color_list[i % len(color_list)])
        plt.title('DANN Features (t-SNE)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1), ncol=1)

        plt.tight_layout()
        plt.show()

# 平均准确率和标准差
mean_accuracy_dann = np.mean(dann_results)
std_accuracy_dann = np.std(dann_results)

print(f"平均准确率: {mean_accuracy_dann}")
print(f"标准差: {std_accuracy_dann}")

