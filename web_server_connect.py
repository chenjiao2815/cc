import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify, send_from_directory
from light_dlinknet import LightDLinkNet


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] Device: {device}')

model = None
model_path = 'weights/light_dlinknet_final.pth'

try:
    model = LightDLinkNet(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False), strict=False)
    model.eval()
    print(f'[INFO] Model loaded: {model_path}')
except Exception as e:
    print(f'[WARN] Model load error: {e}')
    print(f'[INFO] Trying fallback model...')
    from web_server import UNetWithAttention
    model = UNetWithAttention(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load('weights/road_best_optimized.pth', map_location=device, weights_only=False), strict=False)
    model.eval()
    print(f'[INFO] Fallback model loaded: weights/road_best_optimized.pth')


def post_process(mask, min_area=20, connect_distance=250):
    """让断断续续的道路变连贯"""
    processed = mask.copy()
    
    # 1. 形态学闭运算：填补小空洞、连接断点
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # 2. 膨胀：保证道路中心线连贯
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    processed = cv2.dilate(processed, kernel_dilate, iterations=1)
    
    # 3. 连通域分析：移除小噪点
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed, connectivity=8)
    
    valid_labels = []
    road_centroids = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            valid_labels.append(i)
            road_centroids.append(centroids[i])
    
    result_mask = np.zeros_like(processed)
    for lbl in valid_labels:
        result_mask[labels == lbl] = 255
    
    # 4. 道路片段连接：距离近的片段自动连接
    if len(road_centroids) >= 2:
        for i in range(len(road_centroids)):
            for j in range(i + 1, len(road_centroids)):
                c1, c2 = road_centroids[i], road_centroids[j]
                dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                if dist < connect_distance:
                    cv2.line(result_mask, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), 255, 12)
    
    # 5. 最终平滑
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, kernel_smooth)
    
    return result_mask


def segment_road(image_path):
    if model is None:
        return None, None, None
    
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    img_resized = cv2.resize(img_rgb, (256, 256))
    img_tensor = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(img_tensor)
        pred_mask = (pred > 0.5).float().squeeze().cpu().numpy()
    
    pred_mask = (pred_mask * 255).astype(np.uint8)
    pred_mask = cv2.resize(pred_mask, (w, h))
    pred_mask_processed = post_process(pred_mask)
    
    mask_colored = cv2.applyColorMap(pred_mask_processed, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.7, mask_colored, 0.3, 0)
    
    return img, pred_mask_processed, overlay


@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>乡村道路分割系统 - 道路连通版</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { text-align: center; color: white; margin-bottom: 30px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .upload-area { background: white; border-radius: 15px; padding: 30px; margin-bottom: 20px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }
        .upload-box { border: 3px dashed #ddd; border-radius: 10px; padding: 50px; text-align: center; cursor: pointer; transition: all 0.3s; }
        .upload-box:hover { border-color: #667eea; background: #f8f9ff; }
        #fileInput { display: none; }
        .btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 12px 30px; border-radius: 8px; cursor: pointer; font-size: 16px; transition: transform 0.2s; }
        .btn:hover { transform: translateY(-2px); }
        .results { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .result-card { background: white; border-radius: 15px; padding: 20px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); }
        .result-card h3 { color: #333; margin-bottom: 15px; font-size: 16px; }
        .result-card img { width: 100%; border-radius: 10px; }
        .model-info { background: #f0f4ff; padding: 15px; border-radius: 10px; margin-bottom: 20px; }
        .model-info p { color: #333; font-size: 14px; }
        .features { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
        .feature-tag { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛤️ 乡村道路分割系统 - 道路连通版</h1>
        
        <div class="model-info">
            <p><strong>模型:</strong> Light D-LinkNet + 通道注意力</p>
            <p><strong>损失函数:</strong> BCE + Dice (70%权重)</p>
            <p><strong>后处理:</strong> 道路补全算法</p>
            <div class="features">
                <span class="feature-tag">形态学闭运算</span>
                <span class="feature-tag">骨架提取</span>
                <span class="feature-tag">连通域分析</span>
                <span class="feature-tag">道路连接</span>
            </div>
        </div>
        
        <div class="upload-area">
            <div class="upload-box" id="uploadBox">
                <div style="font-size: 48px; margin-bottom: 10px;">📷</div>
                <p>点击或拖拽上传遥感影像</p>
                <p style="color:#999; font-size:14px; margin-top:5px;">支持: JPG, PNG, BMP</p>
            </div>
            <input type="file" id="fileInput" accept="image/*">
            <div style="text-align:center; margin-top:20px;">
                <button class="btn" id="processBtn" disabled>🚀 提取道路</button>
            </div>
        </div>
        
        <div id="resultsContainer" class="results" style="display:none;">
            <div class="result-card">
                <h3>🖼️ 原图</h3>
                <img id="originalImg" />
            </div>
            <div class="result-card">
                <h3>🎯 道路掩膜</h3>
                <img id="maskImg" />
            </div>
            <div class="result-card">
                <h3>✨ 叠加可视化</h3>
                <img id="overlayImg" />
            </div>
        </div>
    </div>

    <script>
        const uploadBox = document.getElementById('uploadBox');
        const fileInput = document.getElementById('fileInput');
        const processBtn = document.getElementById('processBtn');
        const resultsContainer = document.getElementById('resultsContainer');
        
        let selectedFile = null;
        
        uploadBox.addEventListener('click', () => fileInput.click());
        uploadBox.addEventListener('dragover', (e) => { e.preventDefault(); uploadBox.style.borderColor = '#667eea'; });
        uploadBox.addEventListener('dragleave', () => uploadBox.style.borderColor = '#ddd');
        uploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadBox.style.borderColor = '#ddd';
            const files = e.dataTransfer.files;
            if (files.length > 0) handleFile(files[0]);
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) handleFile(e.target.files[0]);
        });
        
        function handleFile(file) {
            selectedFile = file;
            uploadBox.innerHTML = '<div style=\"font-size: 48px; margin-bottom: 10px;\">✅</div><p>已选择: ' + file.name + '</p>';
            processBtn.disabled = false;
            resultsContainer.style.display = 'none';
        }
        
        processBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            const response = await fetch('/upload', { method: 'POST', body: formData });
            const data = await response.json();
            
            document.getElementById('originalImg').src = '/uploads/' + selectedFile.name + '?t=' + Date.now();
            document.getElementById('maskImg').src = data.mask + '?t=' + Date.now();
            document.getElementById('overlayImg').src = data.overlay + '?t=' + Date.now();
            
            resultsContainer.style.display = 'grid';
        });
    </script>
</body>
</html>
    '''


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    filename = file.filename
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)
    
    original, mask, overlay = segment_road(upload_path)
    
    if original is None:
        return jsonify({'error': 'Failed to process image'})
    
    base_name = os.path.splitext(filename)[0]
    mask_path = f'{base_name}_mask.png'
    overlay_path = f'{base_name}_overlay.png'
    
    cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], mask_path), mask)
    cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], overlay_path), overlay)
    
    return jsonify({
        'mask': f'/results/{mask_path}',
        'overlay': f'/results/{overlay_path}'
    })


@app.route('/results/<filename>')
def get_result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


@app.route('/uploads/<filename>')
def get_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    print('[INFO] Web server starting at http://localhost:5001')
    app.run(host='0.0.0.0', port=5001, debug=False)