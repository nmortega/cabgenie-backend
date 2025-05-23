import os
import tempfile
import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO

from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from django.http import HttpResponse
from django.conf import settings

import torch
from monai.networks.nets import DynUNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstD, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, Resized, ToTensord
)
from monai.networks.layers import Norm
import urllib.request
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import traceback
from monai.transforms import Orientation
import cloudinary.uploader


# Path where you want to store the downloaded model
MODEL_LOCAL_PATH = os.path.join("models", "best_metric_model.pth")
MODEL_REMOTE_URL = "https://github.com/nmortega/cabgenie-dynunet-best-weight/raw/refs/heads/main/best_metric_model.pth"

# Create model directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Download only if file doesn't exist
if not os.path.exists(MODEL_LOCAL_PATH):
    print("🔽 Downloading model checkpoint...")
    urllib.request.urlretrieve(MODEL_REMOTE_URL, MODEL_LOCAL_PATH)
    print("✅ Downloaded checkpoint to:", MODEL_LOCAL_PATH)

# Constants
pixdim = (1.5, 1.5, 1.0)
a_min, a_max = -200, 200
spatial_size = (128, 128, 64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Setup
print("🧠 Initializing DynUNet model...")

model = DynUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    kernel_size=[(3, 3, 3)] * 5,
    strides=[(1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
    upsample_kernel_size=[(2, 2, 2)] * 4,
    dropout=0.2,
    norm_name="BATCH"
).to(device)

print("✅ DynUNet architecture created.")
print(f"📁 Loading weights from: {MODEL_LOCAL_PATH}")

checkpoint = torch.load(MODEL_LOCAL_PATH, map_location=device)
cleaned_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
model.load_state_dict(cleaned_state_dict)
model.eval()

print("✅ Model weights loaded successfully.")
print(f"🧠 Model class: {model.__class__.__name__}")

# Preprocessing
test_transforms = Compose([
    LoadImaged(keys=["vol"]),
    EnsureChannelFirstD(keys=["vol"]),
    Spacingd(keys=["vol"], pixdim=pixdim, mode="bilinear"),
    Orientationd(keys=["vol"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["vol"], source_key="vol"),
    Resized(keys=["vol"], spatial_size=spatial_size),
    ToTensord(keys=["vol"]),
])

@method_decorator(csrf_exempt, name='dispatch')
class ImageUploadView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, *args, **kwargs):
        uploaded_file = request.FILES.get('image')
        if not uploaded_file:
            return Response({"error": "No file uploaded."}, status=400)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz", dir="/tmp") as tmp:
            for chunk in uploaded_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            # Preprocess
            data = test_transforms({"vol": tmp_path})
            image_tensor = data["vol"].unsqueeze(0).to(device)
            volume = data["vol"].cpu().numpy()[0]

            # Predict
            with torch.no_grad():
                output = sliding_window_inference(
                    image_tensor,
                    roi_size=spatial_size,
                    sw_batch_size=1,
                    predictor=model
                )
                prediction = torch.argmax(output, dim=1).cpu().numpy()[0]
                prediction_tensor = torch.argmax(output, dim=1)

            # Re-orient back to LAS (original orientation)
            reverse_orient = Orientation(axcodes="LAS")
            vol_tensor = torch.from_numpy(volume).unsqueeze(0)
            pred_tensor = prediction_tensor

            vol_las = reverse_orient(vol_tensor)[0].numpy()
            pred_las = reverse_orient(pred_tensor)[0].numpy()

            # Save volume and mask separately
            glance_dir = os.path.join(settings.MEDIA_ROOT, "glance_data")
            os.makedirs(glance_dir, exist_ok=True)

            affine = nib.load(tmp_path).affine
            volume_path = os.path.join(glance_dir, "volume.nii.gz")
            mask_path = os.path.join(glance_dir, "mask.nii.gz")
            nib.save(nib.Nifti1Image(vol_las.astype(np.float32), affine), volume_path)
            nib.save(nib.Nifti1Image(pred_las.astype(np.uint8), affine), mask_path)

            # Matplotlib preview
            start, end = 16, 48
            fig, axes = plt.subplots(end - start, 3, figsize=(15, (end - start) * 3.5))

            for idx, i in enumerate(range(start, end)):
                axes[idx, 0].imshow(volume[:, :, i], cmap='gray', interpolation='none')
                axes[idx, 0].axis('off')
                axes[idx, 0].set_title(f"Image Slice {i}", fontsize=12)

                axes[idx, 1].imshow(volume[:, :, i], cmap='gray', interpolation='none')
                axes[idx, 1].imshow(prediction[:, :, i], cmap='hot', alpha=0.5, interpolation='none')
                axes[idx, 1].axis('off')
                axes[idx, 1].set_title(f"Overlay Slice {i}", fontsize=12)

                axes[idx, 2].imshow(prediction[:, :, i], cmap='hot', interpolation='none')
                axes[idx, 2].axis('off')
                axes[idx, 2].set_title(f"Prediction Slice {i}", fontsize=12)

            plt.tight_layout(pad=2.0)


            preview_path = os.path.join(glance_dir, "preview.png")
            plt.savefig(preview_path, format='png', bbox_inches='tight')
            plt.close()

            # Build response URLs
            base_url = request.build_absolute_uri('/')[:-1]
            volume_url = f"{base_url}/media/glance_data/volume.nii.gz"
            mask_url = f"{base_url}/media/glance_data/mask.nii.gz"
            preview_url = f"{base_url}/media/glance_data/preview.png"

            response_payload = {
                "volume_url": volume_url,
                "mask_url": mask_url,
                "preview_url": preview_url,
            }

            print("📦 Response payload:", response_payload)
            return Response(response_payload, content_type="application/json")


        except Exception as e:
            traceback.print_exc()
            return Response({"error": str(e)}, status=500)

        finally:
            os.remove(tmp_path)