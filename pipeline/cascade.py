import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import SegformerForSemanticSegmentation
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# WELD DEFECT DETECTION CASCADE (v2 + Grad-CAM++)
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Pipeline running on: {DEVICE}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRECLASSIFIER_PATH = os.path.join(BASE_DIR, "models", "preclassifier", "preclassifier_best.pth")
SEGFORMER_PATH     = os.path.join(BASE_DIR, "models", "unet", "segformer_best_v2.pth")

PRECLASSIFIER_THRESHOLD = 0.5
ACCEPT_THRESHOLD        = 0.7
REVIEW_THRESHOLD        = 0.4
MIN_DEFECT_AREA         = 20

# ─────────────────────────────────────────────
# LOAD MODELS & TRANSFORMS
# ─────────────────────────────────────────────
def load_preclassifier(path):
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    return model

def load_segformer(path):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2",
        num_labels=2,
        id2label={0: "background", 1: "defect"},
        label2id={"background": 0, "defect": 1},
        ignore_mismatched_sizes=True,
    )
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    return model

preclassifier_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])
segformer_transform = transforms.Compose([
    transforms.Resize((512, 512)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

# ─────────────────────────────────────────────
# PIPELINE LOGIC
# ─────────────────────────────────────────────
class WeldDefectPipeline:
    def __init__(self):
        print("\n" + "="*50)
        print("Initializing Weld Defect Detection Pipeline")
        print("="*50)
        self.preclassifier = load_preclassifier(PRECLASSIFIER_PATH)
        self.segformer     = load_segformer(SEGFORMER_PATH)

    def predict(self, image_path):
        image_pil = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image_pil.size

        result = {"image_path": image_path, "is_weld": False, "mask": None, "prob_map": None,
                  "defect_area": 0, "status": "NO_DEFECT", "confidence": 0.0, "stages": {}}

        # Stage 1
        tensor_pre = preclassifier_transform(image_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pre_conf = torch.sigmoid(self.preclassifier(tensor_pre)).item()
        
        if pre_conf < PRECLASSIFIER_THRESHOLD:
            result["status"] = "REJECTED_NON_WELD"
            return result
        result["is_weld"] = True

        # Stage 2
        tensor_seg = segformer_transform(image_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = self.segformer(pixel_values=tensor_seg)
            logits = outputs.logits
            logits_up = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            prob_map = F.softmax(logits_up, dim=1)[0, 1].cpu().numpy()
            mask = (prob_map > 0.5).astype(np.uint8) * 255
            
        result["prob_map"] = prob_map 

        # Stage 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask_clean = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_clean)
        cleaned = np.zeros_like(mask_clean)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= MIN_DEFECT_AREA:
                cleaned[labels == i] = 255
        
        defect_area = int((cleaned > 0).sum())
        result["mask"] = cleaned
        result["defect_area"] = defect_area

        # Stage 4
        if defect_area > 0:
            region_conf = float(prob_map[cleaned > 0].mean())
            result["confidence"] = region_conf
            result["status"] = "ACCEPT" if region_conf >= ACCEPT_THRESHOLD else "REVIEW"

        return result

    def _save_annotated(self, img_path, result, output_dir, fname):
        img_cv = cv2.imread(img_path)
        h, w = img_cv.shape[:2]
        base_name = os.path.splitext(fname)[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        status_color = {"ACCEPT":(0,255,0),"REVIEW":(0,165,255),"NO_DEFECT":(255,255,255),"REJECTED_NON_WELD":(128,128,128)}
        s_color = status_color.get(result["status"], (255,255,255))
        conf = result["confidence"]

        def draw_status(img):
            cv2.rectangle(img, (0,0), (280,44), (0,0,0), -1)
            cv2.putText(img, f"{result['status']} {conf:.2f}", (8,32), font, 0.9, s_color, 2)

        blob_boxes = []
        if result["mask"] is not None:
            mask_rs = cv2.resize(result["mask"], (w,h))
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_rs, connectivity=8)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < 30: continue
                bx,by,bw,bh = stats[i,cv2.CC_STAT_LEFT],stats[i,cv2.CC_STAT_TOP],stats[i,cv2.CC_STAT_WIDTH],stats[i,cv2.CC_STAT_HEIGHT]
                blob_boxes.append((bx,by,bx+bw,by+bh))

        # 1. Bounding Box Image
        bbox_img = img_cv.copy()
        for idx,(x1,y1,x2,y2) in enumerate(blob_boxes):
            cv2.rectangle(bbox_img,(x1,y1),(x2,y2),(0,255,255),2)
            cv2.putText(bbox_img,f"R{idx+1}",(x1,max(12,y1-4)),font,0.45,(0,255,255),1)
        draw_status(bbox_img)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_bbox.png"), bbox_img)
        result["img_bbox"] = cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB)

        # 2. Defect Overlay Image
        mask_layer = np.zeros_like(img_cv)
        if result["mask"] is not None:
            mask_rs = cv2.resize(result["mask"], (w,h))
            mask_layer[mask_rs>0] = [0,0,255]
        overlay_out = cv2.addWeighted(img_cv,0.55,mask_layer,0.45,0)
        for idx,(x1,y1,x2,y2) in enumerate(blob_boxes):
            cv2.rectangle(overlay_out,(x1,y1),(x2,y2),(0,255,255),2)
        draw_status(overlay_out)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_overlay.png"), overlay_out)
        result["img_overlay"] = cv2.cvtColor(overlay_out, cv2.COLOR_BGR2RGB)

        # 3. Binary Mask Image
        mask_out = np.zeros((h,w,3), dtype=np.uint8)
        if result["mask"] is not None:
            mask_rs = cv2.resize(result["mask"], (w,h))
            mask_out[mask_rs>0] = [255,255,255]
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_mask.png"), mask_out)
        result["img_mask"] = cv2.cvtColor(mask_out, cv2.COLOR_BGR2RGB)

        # 4. RAW GRAD-CAM++ ACADEMIC HEATMAP
        try:
            img_pil = Image.open(img_path).convert("RGB")
            tensor  = segformer_transform(img_pil).unsqueeze(0).to(DEVICE)
            tensor.requires_grad_(True)

            activations = {}
            gradients   = {}
            
            target_layer = self.segformer.segformer.encoder.block[-1][-1]

            def fwd_hook(module, inp, out):
                activations["feat"] = (out[0] if isinstance(out, tuple) else out).detach()

            def bwd_hook(module, gin, gout):
                gradients["feat"] = (gout[0] if isinstance(gout, tuple) else gout).detach()

            fh = target_layer.register_forward_hook(fwd_hook)
            bh = target_layer.register_full_backward_hook(bwd_hook)

            self.segformer.eval()
            outputs = self.segformer(pixel_values=tensor)
            defect_score = outputs.logits[0, 1].mean()
            self.segformer.zero_grad()
            defect_score.backward()
            fh.remove()
            bh.remove()

            feat = activations.get("feat")
            grad = gradients.get("feat")

            if feat is not None and grad is not None:
                if feat.dim() == 3:
                    B, N, C = feat.shape
                    sp   = int(N ** 0.5)
                    feat = feat.reshape(B, sp, sp, C).permute(0,3,1,2)
                    grad = grad.reshape(B, sp, sp, C).permute(0,3,1,2)

                # Grad-CAM++ Alpha Weighting Formula
                grad_2 = grad ** 2
                grad_3 = grad_2 * grad
                sum_activations = feat.sum(dim=(2, 3), keepdim=True)
                alpha = grad_2 / (2 * grad_2 + sum_activations * grad_3 + 1e-8)
                weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
                
                cam = (weights * feat).sum(dim=1, keepdim=True)
                cam = F.relu(cam).squeeze().cpu().numpy()
                
                cam_resized = cv2.resize(cam, (w, h))
                cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
                
                gradcam_pp_out = cv2.applyColorMap((cam_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

                draw_status(gradcam_pp_out)
                cv2.putText(gradcam_pp_out, "Grad-CAM++ Heatmap", (w-200, h-10), font, 0.5, (255,255,255), 1)

                cv2.imwrite(os.path.join(output_dir, f"{base_name}_gradcam.png"), gradcam_pp_out)
                print("  Grad-CAM++ saved")
                result["img_xai"] = cv2.cvtColor(gradcam_pp_out, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError("Hook capture failed")

        except Exception as e:
            print(f"  Grad-CAM++ error: {e} — using probability map fallback")
            if result["prob_map"] is not None:
                prob_resized = cv2.resize(result["prob_map"], (w, h))
                prob_norm = (prob_resized - prob_resized.min()) / (prob_resized.max() - prob_resized.min() + 1e-8)
                gradcam_pp_out = cv2.applyColorMap((prob_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                draw_status(gradcam_pp_out)
                cv2.imwrite(os.path.join(output_dir, f"{base_name}_gradcam.png"), gradcam_pp_out)
                result["img_xai"] = cv2.cvtColor(gradcam_pp_out, cv2.COLOR_BGR2RGB)