# predict_with_m4_model.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

class M4Predictor:
    def __init__(self, model_path='best_m4_model.pth'):
        # 设置设备
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🎉 使用M4 GPU进行预测")
        else:
            self.device = torch.device("cpu")
            print("⚡ 使用CPU进行预测")
        
        # 加载模型
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.class_names = checkpoint['class_names']
            self.best_accuracy = checkpoint.get('accuracy', 0)
            
            # 创建模型结构（使用ResNet50，与训练时一致）
            self.model = models.resnet50(pretrained=False)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, len(self.class_names))
            )
            
            # 加载权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # 数据预处理
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            print(f"✅ 模型加载成功！")
            print(f"📊 训练准确率: {self.best_accuracy:.2f}%")
            print(f"🎯 识别类别: {self.class_names}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

    def predict(self, image_path):
        """预测单张图片"""
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                print(f"❌ 文件不存在: {image_path}")
                return None, None
            
            # 加载图片
            image = Image.open(image_path).convert('RGB')
            original_image = np.array(image)
            
            print(f"📷 正在分析图片: {os.path.basename(image_path)}")
            print(f"📐 图片尺寸: {image.size}")
            
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
            self._display_results(original_image, predicted_class, confidence, probabilities)
            
            print(f"🎯 预测结果: {predicted_class}")
            print(f"📊 置信度: {confidence:.4f} ({confidence*100:.2f}%)")
            
            # 判断是否高置信度
            if confidence > 0.8:
                print("💪 高置信度预测！")
            elif confidence > 0.6:
                print("👍 中等置信度预测")
            else:
                print("🤔 低置信度预测，可能需要更多训练数据")
            
            return predicted_class, confidence
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return None, None

    def _display_results(self, image, predicted_class, confidence, probabilities):
        """显示预测结果"""
        plt.figure(figsize=(13, 6))
        
        # 显示原图
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f'M4模型预测结果\n预测: {predicted_class}\n置信度: {confidence*100:.2f}%', 
                 fontsize=14, pad=15)
        plt.axis('off')
        
        # 显示置信度条形图
        plt.subplot(1, 2, 2)
        colors = ['#ff6b6b' if i == predicted_class else '#4ecdc4' 
                 for i in range(len(self.class_names))]
        
        confidences = [probabilities[i].item() for i in range(len(self.class_names))]
        bars = plt.bar(self.class_names, confidences, color=colors, alpha=0.8, width=0.6)
        
        plt.ylim(0, 1.1)
        plt.title('分类置信度分布', fontsize=14, pad=15)
        plt.ylabel('置信度', fontsize=12)
        plt.grid(True, axis='y', alpha=0.3)
        
        # 在条形上添加数值
        for bar, conf in zip(bars, confidences):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                    f'{conf:.3f}', ha='center', va='bottom', fontsize=12, 
                    fontweight='bold', color='black')
        
        # 添加阈值线
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='阈值 (0.5)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def predict_multiple(self, image_folder):
        """预测文件夹中的所有图片"""
        if not os.path.exists(image_folder):
            print(f"❌ 文件夹不存在: {image_folder}")
            return
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print(f"❌ 在 {image_folder} 中没有找到图片文件")
            return
        
        print(f"📁 在 {image_folder} 中找到 {len(image_files)} 张图片")
        print("开始批量预测...")
        
        results = []
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] 处理: {image_file}")
            image_path = os.path.join(image_folder, image_file)
            result, confidence = self.predict(image_path)
            if result is not None:
                results.append((image_file, result, confidence))
        
        # 显示统计结果
        if results:
            self._show_statistics(results)

    def _show_statistics(self, results):
        """显示批量预测的统计结果"""
        print("\n" + "="*60)
        print("                 M4模型批量预测统计结果")
        print("="*60)
        
        # 按类别统计
        from collections import Counter
        class_counter = Counter([result[1] for result in results])
        
        total_images = len(results)
        print(f"📊 总计处理图片: {total_images}张")
        print("\n分类结果统计:")
        for class_name in self.class_names:
            count = class_counter.get(class_name, 0)
            percentage = (count / total_images) * 100 if total_images > 0 else 0
            print(f"  {class_name}: {count}张 ({percentage:.1f}%)")
        
        # 置信度统计
        confidences = [result[2] for result in results]
        avg_confidence = np.mean(confidences)
        max_confidence = np.max(confidences)
        min_confidence = np.min(confidences)
        
        print(f"\n📈 置信度统计:")
        print(f"  平均置信度: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
        print(f"  最高置信度: {max_confidence:.4f} ({max_confidence*100:.2f}%)")
        print(f"  最低置信度: {min_confidence:.4f} ({min_confidence*100:.2f}%)")
        
        # 高置信度图片统计
        high_conf = len([c for c in confidences if c > 0.8])
        medium_conf = len([c for c in confidences if 0.6 < c <= 0.8])
        low_conf = len([c for c in confidences if c <= 0.6])
        
        print(f"\n🎯 置信度分布:")
        print(f"  高置信度 (>0.8): {high_conf}张 ({(high_conf/total_images)*100:.1f}%)")
        print(f"  中置信度 (0.6-0.8): {medium_conf}张 ({(medium_conf/total_images)*100:.1f}%)")
        print(f"  低置信度 (≤0.6): {low_conf}张 ({(low_conf/total_images)*100:.1f}%)")

def test_with_sample_images():
    """使用示例图片测试模型"""
    print("🐱🐶 测试M4猫狗分类模型")
    
    # 检查模型文件
    if not os.path.exists('best_m4_model.pth'):
        print("❌ 模型文件 'best_m4_model.pth' 不存在")
        print("请确保模型文件在当前目录")
        return
    
    # 创建预测器
    try:
        predictor = M4Predictor('best_m4_model.pth')
    except Exception as e:
        print(f"❌ 无法加载模型: {e}")
        return
    
    # 检查是否有测试图片
    test_images = []
    for ext in ['.jpg', '.jpeg', '.png']:
        test_images.extend([f for f in os.listdir('.') if f.lower().endswith(ext)])
    
    if test_images:
        print(f"\n📷 发现 {len(test_images)} 张测试图片:")
        for img in test_images:
            print(f"  - {img}")
        
        choice = input("\n是否测试这些图片? (y/n): ").strip().lower()
        if choice == 'y':
            for img in test_images:
                print(f"\n{'='*50}")
                predictor.predict(img)
    else:
        print("\n📝 使用说明:")
        print("1. 将猫狗图片放在当前目录")
        print("2. 运行: python predict_with_m4_model.py")
        print("3. 或者运行下面的交互模式")

def interactive_mode():
    """交互式预测模式"""
    print("\n🔍 M4猫狗分类器 - 交互模式")
    
    try:
        predictor = M4Predictor('best_m4_model.pth')
    except Exception as e:
        print(f"❌ 无法加载模型: {e}")
        return
    
    while True:
        print("\n" + "="*50)
        print("请选择操作:")
        print("1. 预测单张图片")
        print("2. 预测文件夹中的所有图片") 
        print("3. 退出")
        
        choice = input("\n请输入选择 (1/2/3): ").strip()
        
        if choice == '1':
            image_path = input("请输入图片路径: ").strip()
            if image_path and os.path.exists(image_path):
                predictor.predict(image_path)
            else:
                print("❌ 图片路径无效或文件不存在")
        
        elif choice == '2':
            folder_path = input("请输入图片文件夹路径: ").strip()
            if folder_path and os.path.exists(folder_path):
                predictor.predict_multiple(folder_path)
            else:
                print("❌ 文件夹路径无效或不存在")
        
        elif choice == '3':
            print("👋 感谢使用M4猫狗分类器！")
            break
        
        else:
            print("❌ 无效选择，请重新输入")

if __name__ == "__main__":
    print("=" * 60)
    print("           M4猫狗分类预测器")
    print("=" * 60)
    
    # 首先测试示例图片
    test_with_sample_images()
    
    # 然后进入交互模式
    interactive_mode()