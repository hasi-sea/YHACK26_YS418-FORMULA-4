import io
import cv2
import json
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI()

# Allow CORS for local testing with the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# STAGE 01: Adaptive Preprocessing Pipeline
# ==========================================
def preprocess_image(image_bytes):
    """
    Apply Otsu's Thresholding (skull stripping sim) and CLAHE (contrast enhancement).
    Returns the processed image as a BGR numpy array and a base64 encoded string.
    """
    # Load image using OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file")

    # Convert to grayscale for thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Otsu's Thresholding to create a mask (simulated skull-stripping)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # 2. CLAHE on the luminance channel
    lab = cv2.cvtColor(masked_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Convert to base64 for the frontend
    _, buffer = cv2.imencode('.jpg', enhanced_img)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    b64_src = f"data:image/jpeg;base64,{b64_str}"

    return enhanced_img, b64_src

# ==========================================
# STAGE 02: Attention-Guided Core Model
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

class NeuroScopeModel(nn.Module):
    def __init__(self):
        super(NeuroScopeModel, self).__init__()
        # Load EfficientNet-B0 backbone
        self.backbone = models.efficientnet_b0(weights=None)
        
        # Inject CBAM before the final classification head
        in_features = self.backbone.classifier[1].in_features
        self.cbam = CBAM(in_planes=1280)  # 1280 is the output channel of effnet b0 features
        
        # Binary classification head (Tumor Present / No Tumor)
        self.backbone.classifier[1] = nn.Linear(in_features, 1)

    def forward(self, x):
        # We need to hook into the features to apply CBAM
        # EfficientNet-B0 features:
        features = self.backbone.features(x)
        attended_features = self.cbam(features)
        
        # Pooling and classification
        out = self.backbone.avgpool(attended_features)
        out = torch.flatten(out, 1)
        out = self.backbone.classifier(F.dropout(out, p=0.2, training=self.training))
        
        return out, attended_features

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuroScopeModel().to(device)
model.eval() # Set to eval mode by default

def load_image_tensor(cv_img):
    """Convert OpenCV BGR image to PyTorch tensor for EfficientNet."""
    img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_img = pil_img.resize((224, 224))
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    # Normalize with ImageNet stats
    img_np = (img_np - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    return img_tensor

# ==========================================
# STAGE 03: Uncertainty Triage (MC Dropout)
# ==========================================
def run_mc_dropout(model, img_tensor, passes=10):
    """
    Run the forward pass 10 separate times on the same input image with Dropout active.
    Returns the mean prediction, variance, and uncertainty flag.
    """
    model.train() # Force dropout to remain active
    
    predictions = []
    with torch.no_grad():
        for _ in range(passes):
            logits, _ = model(img_tensor)
            prob = torch.sigmoid(logits).item()
            predictions.append(prob)
            
    mean_prob = np.mean(predictions)
    variance = np.var(predictions)
    
    # If variance exceeds 0.05, require senior review
    requires_senior_review = bool(variance > 0.05)
    
    return mean_prob, variance, requires_senior_review

# ==========================================
# STAGE 04: Explainable AI (Grad-CAM)
# ==========================================
def generate_gradcam(model, img_tensor, original_cv_img):
    """
    Hook into the final convolutional layer of the EfficientNet-B0 backbone.
    Calculate gradients to generate a spatial activation map.
    """
    model.eval() # Ensure dropout is disabled for Grad-CAM
    
    # Enable gradient calculation on the tensor
    img_tensor.requires_grad = True
    
    # Forward pass
    logits, features = model(img_tensor)
    
    # Target class (since binary, we just backprop the single logit)
    model.zero_grad()
    logits.backward(retain_graph=True)
    
    # Get the gradients of the features
    # Since we didn't attach a hook, we can manually compute gradients or use a simpler approximation 
    # For a robust implementation, we would register a hook on `attended_features`.
    # To keep it completely functional without complex hooks in this simple script, 
    # we can use the weights of the final linear layer as a CAM approximation, 
    # OR properly register a hook. Let's do the hook.
    
    gradients = None
    activations = None
    
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]
        
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    # Register hooks on the CBAM module (which is the last conv feature block)
    h1 = model.cbam.register_forward_hook(forward_hook)
    h2 = model.cbam.register_backward_hook(backward_hook)
    
    # Redo forward and backward to trigger hooks
    logits, _ = model(img_tensor)
    model.zero_grad()
    logits.backward()
    
    h1.remove()
    h2.remove()
    
    if gradients is not None and activations is not None:
        # Global average pooling on gradients
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Weight the channels by corresponding gradients
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()
        
        # ReLU on top of the heatmap
        heatmap = F.relu(heatmap)
        
        # Normalize heatmap
        heatmap /= torch.max(heatmap) + 1e-8
        
        # Convert to numpy
        heatmap = heatmap.cpu().detach().numpy()
    else:
        # Fallback dummy heatmap if hooks fail
        heatmap = np.zeros((7, 7), dtype=np.float32)
        heatmap[3, 3] = 1.0

    # Resize heatmap to original image size
    heatmap = cv2.resize(heatmap, (original_cv_img.shape[1], original_cv_img.shape[0]))
    
    # Convert to uint8 and apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap)
    colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Overlay onto original scan with 0.4 alpha
    overlay = cv2.addWeighted(colormap, 0.4, original_cv_img, 0.6, 0)
    
    # Convert to base64
    _, buffer = cv2.imencode('.jpg', overlay)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    b64_src = f"data:image/jpeg;base64,{b64_str}"
    
    return b64_src

# ==========================================
# API ENDPOINT ROUTING
# ==========================================
@app.post("/analyze-scan")
async def analyze_scan(file: UploadFile = File(...)):
    """
    Executes all 4 stages sequentially and returns the JSON response.
    """
    try:
        # Read uploaded image bytes
        image_bytes = await file.read()
        
        # Stage 1: Adaptive Preprocessing
        enhanced_img, preprocessed_b64 = preprocess_image(image_bytes)
        
        # Stage 2: Convert to Tensor for PyTorch
        img_tensor = load_image_tensor(enhanced_img)
        
        # Stage 3: Uncertainty Triage (MC Dropout)
        mean_prob, variance, uncertainty_flag = run_mc_dropout(model, img_tensor, passes=10)
        
        # Format prediction
        prediction = "Tumor Present" if mean_prob > 0.5 else "No Tumor"
        confidence = float(mean_prob * 100) if mean_prob > 0.5 else float((1 - mean_prob) * 100)
        
        # Stage 4: Explainable AI (Grad-CAM)
        heatmap_b64 = generate_gradcam(model, img_tensor, enhanced_img)
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "uncertainty_flag": uncertainty_flag,
            "preprocessed_b64": preprocessed_b64,
            "heatmap_b64": heatmap_b64
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
