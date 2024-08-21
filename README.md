#本项目主要使用Image Color Summarizer提供的接口，通过应用k-meanings算法对图像中的颜色进行识别，筛选出占比最大的五种颜色，并根据相似度从色卡中匹配颜色，提取深浅对比和冷暖对比最为显著的两个颜色输出


import os
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import requests
import pandas as pd
from tqdm import tqdm  # 引入 tqdm 库

# 定义计算颜色相似度的函数
def calculate_similarity(color1, color2):
    d = np.sqrt(np.sum((np.array(color1) - np.array(color2)) ** 2))
    similarity = 1 - d / np.sqrt(255**2 + 255**2 + 255**2)
    return similarity

# 从色卡图片中提取颜色
def extract_palette_colors(palette_image_path, num_colors=8):
    image = Image.open(palette_image_path)
    image = image.convert("RGB")
    image = np.array(image).reshape((-1, 3))

    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(image)
    colors = kmeans.cluster_centers_
    return [tuple(map(int, color)) for color in colors]

# 使用API分析图片的基本颜色
def analyze_image_colors(image_path, api_url_base):
    image_name = os.path.basename(image_path)
    image_url = f"{api_url_base}/{image_name}"
    api_url = f"http://mkweb.bcgsc.ca/color-summarizer/?url={image_url}&precision=medium&json=1&num_clusters=5"
    
    for attempt in range(3):
        try:
            response = requests.get(api_url)
            if response.status_code == 200:
                data = response.json()
                return [tuple(map(int, color['rgb'])) for color in data['clusters'].values()]
        except (requests.exceptions.RequestException, ValueError):
            print(f"第 {attempt+1} 次请求失败，正在重试...")
    
    print("多次请求失败，跳过该图像")
    return None

# 匹配颜色到色卡颜色
def match_colors_to_palette(image_colors, palette_colors):
    matched_colors = []
    for color in image_colors:
        similarities = [calculate_similarity(color, palette_color) for palette_color in palette_colors]
        best_match = palette_colors[np.argmax(similarities)]
        matched_colors.append(best_match)
    return matched_colors

# 图片路径和色卡图片路径
image_folder = r"C:\Users\13982\Desktop\测试"
palette_image_path = r"C:\Users\13982\Desktop\测试色卡.png"
ngrok_url = "https://81b9-14-145-160-32.ngrok-free.app"

# 提取色卡颜色并打印
palette_colors = extract_palette_colors(palette_image_path, num_colors=8)
print("提取到的色卡颜色RGB值：")
for i, color in enumerate(palette_colors):
    print(f"色卡颜色 {i+1}: {color}")

# 初始化结果列表
results = []

# 使用 tqdm 添加进度条，遍历文件夹中的所有图片文件
for filename in tqdm(os.listdir(image_folder), desc="处理进度"):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image_path = os.path.join(image_folder, filename)
        image_colors = analyze_image_colors(image_path, ngrok_url)
        
        if image_colors:
            matched_colors = match_colors_to_palette(image_colors, palette_colors)

            # 计算深浅对比
            contrasts = []
            for i in range(len(matched_colors)):
                for j in range(i + 1, len(matched_colors)):
                    d = np.sqrt(np.sum((np.array(matched_colors[i]) - np.array(matched_colors[j])) ** 2))
                    contrasts.append((d, matched_colors[i], matched_colors[j]))

            # 找出深浅对比最大的两个颜色
            max_contrast = max(contrasts, key=lambda x: x[0])
            dark_color, light_color = max_contrast[1], max_contrast[2]
            
            # 保存结果
            results.append({
                "图片名称": os.path.splitext(filename)[0],
                "深色": dark_color,
                "浅色": light_color
            })

# 保存为Excel文件
output_file = os.path.join(image_folder, "色卡匹配结果.xlsx")
df = pd.DataFrame(results)
df.to_excel(output_file, index=False)

print(f"\n色卡匹配分析完成，结果已保存为 {output_file}")
