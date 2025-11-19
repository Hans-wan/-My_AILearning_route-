# m4_cat_dog_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import time
import json
from tqdm import tqdm

class M4CatDogTrainer:
    def __init__(self):
        # 检查并设置MPS设备
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🎉 M4 GPU (MPS) 可用，使用GPU加速训练")
        else:
            self.device = torch.device("cpu")
            print("⚠️  MPS不可用，使用CPU训练")
        
        # M4优化参数
        self.batch_size = 32  # M4内存较大，可以使用更大的batch size
        self.num_epochs = 50
        self.learning_rate = 0.001
        
        # 创建数据目录
        self.setup_directories()
        
        # 数据变换（针对M4优化）
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def setup_directories(self):
        """创建数据目录结构"""
        directories = [
            'data/train/cats',
            'data/train/dogs', 
            'data/val/cats',
            'data/val/dogs'
        ]
        
        for dir_path in directories:
            os.makedirs(dir_path, exist_ok=True)
        
        print("📁 数据目录结构已创建")
    def create_model(self, num_classes=2):
        """创建针对M4优化的模型"""
        # 使用预训练的ResNet50，更大的模型在M4上也能很好运行
        model = models.resnet50(pretrained=True)
        
        # 冻结前面层，只训练最后几层
        for param in model.parameters():
            param.requires_grad = False
        
        # 解冻最后两个层
        for param in model.layer4.parameters():
            param.requires_grad = True
        
        # 替换分类器
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        return model.to(self.device)

    def load_data(self):
        """加载数据"""
        print("📊 加载数据...")
        
        try:
            train_dataset = ImageFolder('data/train', transform=self.train_transform)
            val_dataset = ImageFolder('data/val', transform=self.val_transform)
            
            # 使用M4优化的数据加载器
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=2,  # M4可以处理更多worker
                pin_memory=True  # 加速GPU数据传输
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            
            print(f"✅ 训练样本: {len(train_dataset)}")
            print(f"✅ 验证样本: {len(val_dataset)}")
            print(f"✅ 类别: {train_dataset.classes}")
            
            return train_loader, val_loader, train_dataset.classes
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return None, None, None

    def train(self):
        """训练模型"""
        print("🚀 开始训练...")
        
        # 加载数据
        train_loader, val_loader, class_names = self.load_data()
        if train_loader is None:
            return
        
        # 创建模型
        model = self.create_model(len(class_names))
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        
        # 记录训练过程
        train_losses = []
        val_accuracies = []
        best_accuracy = 0.0
        
        # 开始训练计时
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            # 训练阶段
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [训练]')
            
            for images, labels in train_bar:
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 统计
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                # 更新进度条
                train_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * correct_train / total_train:.2f}%'
                })
            
            # 验证阶段
            model.eval()
            correct_val = 0
            total_val = 0
            
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{self.num_epochs} [验证]')
            
            with torch.no_grad():
                for images, labels in val_bar:
                    images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
                    
                    val_bar.set_postfix({
                        'Acc': f'{100 * correct_val / total_val:.2f}%'
                    })
            
            # 计算指标
            train_accuracy = 100 * correct_train / total_train
            val_accuracy = 100 * correct_val / total_val
            avg_loss = running_loss / len(train_loader)
            
            train_losses.append(avg_loss)
            val_accuracies.append(val_accuracy)
            
            # 学习率调度
            scheduler.step()
            
            print(f'\n📊 Epoch {epoch+1} 结果:')
            print(f'   训练损失: {avg_loss:.4f}')
            print(f'   训练准确率: {train_accuracy:.2f}%')
            print(f'   验证准确率: {val_accuracy:.2f}%')
            print(f'   学习率: {scheduler.get_last_lr()[0]:.2e}')
            
            # 保存最佳模型
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_accuracy,
                    'class_names': class_names
                }, 'best_m4_model.pth')
                print(f'   💾 保存最佳模型，准确率: {best_accuracy:.2f}%')
            
            print('-' * 50)
        
        # 训练完成
        training_time = time.time() - start_time
        print(f'✅ 训练完成! 总时间: {training_time/60:.2f} 分钟')
        print(f'🏆 最佳验证准确率: {best_accuracy:.2f}%')
        
        return model, train_losses, val_accuracies

    def plot_results(self, train_losses, val_accuracies):
        """绘制训练结果"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(train_losses)
        plt.title('训练损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(val_accuracies)
        plt.title('验证准确率')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        
        # 添加训练信息
        plt.subplot(1, 3, 3)
        plt.axis('off')
        info_text = f"""训练信息:
设备: {self.device}
Batch Size: {self.batch_size}
最佳准确率: {max(val_accuracies):.2f}%
最终损失: {train_losses[-1]:.4f}"""
        plt.text(0.1, 0.9, info_text, fontsize=12, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig('m4_training_results.png', dpi=300, bbox_inches='tight')
        plt.show()

class M4Predictor:
    def __init__(self, model_path='best_m4_model.pth'):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        self.class_names = checkpoint['class_names']
        
        self.model = models.resnet50(pretrained=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, len(self.class_names))
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 数据变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ 模型加载完成，使用设备: {self.device}")

    def predict(self, image_path):
        """预测单张图片"""
        # 加载图片
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        
        # 预处理
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class_idx].item()
        
        predicted_class = self.class_names[predicted_class_idx]
        
        # 显示结果
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title(f'输入图片\n预测: {predicted_class} ({confidence*100:.2f}%)', fontsize=14)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        # 绘制置信度
        colors = ['#ff9999' if i == predicted_class_idx else '#66b3ff' for i in range(len(self.class_names))]
        y_pos = np.arange(len(self.class_names))
        confidences = [probabilities[i].item() for i in range(len(self.class_names))]
        
        bars = plt.barh(y_pos, confidences, color=colors)
        plt.xlabel('置信度', fontsize=12)
        plt.title('分类置信度', fontsize=14)
        plt.yticks(y_pos, self.class_names)
        plt.xlim(0, 1)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=11)
        
        plt.tight_layout()
        plt.show()
        
        print(f"🎯 预测结果: {predicted_class}")
        print(f"📊 置信度: {confidence:.4f}")
        
        return predicted_class, confidence

def main():
    """主函数"""
    print("=" * 60)
    print("           M4 GPU 猫狗分类器")
    print("=" * 60)
    
    # 创建训练器并开始训练
    trainer = M4CatDogTrainer()
    model, train_losses, val_accuracies = trainer.train()
    
    # 绘制结果
    if model is not None:
        trainer.plot_results(train_losses, val_accuracies)
        
        print("\n🎉 训练完成！您可以使用以下代码进行预测：")
        print("predictor = M4Predictor('best_m4_model.pth')")
        print("result, confidence = predictor.predict('您的图片路径.jpg')")

if __name__ == "__main__":
    main()