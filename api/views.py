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
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstD, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, Resized, ToTensord
)
from monai.networks.layers import Norm

# Constants
pixdim = (1.5, 1.5, 1.0)
a_min, a_max = -200, 200
spatial_size = (128, 128, 64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Setup
model = SegResNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    init_filters=16,
    blocks_down=(1, 2, 2, 4),
    blocks_up=(1, 1, 1),
    dropout_prob=0.2,
    norm=Norm.BATCH,
).to(device)

checkpoint = torch.load(
    "/Users/nothanilo/Desktop/Vast.ai Training Weights/SegResNet Epochs 200 Dropout 0.2/best_metric_model.pth",
    map_location=device
)
cleaned_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
model.load_state_dict(cleaned_state_dict)
model.eval()

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

class ImageUploadView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, *args, **kwargs):
        uploaded_file = request.FILES.get('image')
        if not uploaded_file:
            return Response({"error": "No file uploaded."}, status=400)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
            for chunk in uploaded_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            # Preprocess
            data = test_transforms({"vol": tmp_path})
            image_tensor = data["vol"].unsqueeze(0).to(device)
            volume = data["vol"].cpu().numpy()[0]  # shape: (128, 128, 64)

            # Predict
            with torch.no_grad():
                output = sliding_window_inference(image_tensor, roi_size=spatial_size, sw_batch_size=1, predictor=model)
                prediction = torch.argmax(output, dim=1).cpu().numpy()[0]  # shape: (128, 128, 64)

            # Save to media/glance_data
            glance_dir = os.path.join(settings.MEDIA_ROOT, "glance_data")
            os.makedirs(glance_dir, exist_ok=True)

            affine = nib.load(tmp_path).affine

            volume_path = os.path.join(glance_dir, "volume.nii.gz")
            mask_path = os.path.join(glance_dir, "mask.nii.gz")

            nib.save(nib.Nifti1Image(volume.astype(np.float32), affine), volume_path)
            nib.save(nib.Nifti1Image(prediction.astype(np.uint8), affine), mask_path)

            # Generate full public URLs
            base_url = request.build_absolute_uri("/media/glance_data/")
            volume_url = base_url + "volume.nii.gz"
            mask_url = base_url + "mask.nii.gz"

            # Plot fewer middle slices to avoid crowding
            start, end = 22, 42
            fig, axes = plt.subplots(end - start, 3, figsize=(12, (end - start) * 2.5))

            for idx, i in enumerate(range(start, end)):
                axes[idx, 0].imshow(volume[:, :, i], cmap='gray')
                axes[idx, 0].axis('off')
                axes[idx, 0].set_title(f"Image Slice {i}")

                axes[idx, 1].imshow(volume[:, :, i], cmap='gray')
                axes[idx, 1].imshow(prediction[:, :, i], cmap='hot', alpha=0.5)
                axes[idx, 1].axis('off')
                axes[idx, 1].set_title(f"Overlay Slice {i}")

                axes[idx, 2].imshow(prediction[:, :, i], cmap='hot')
                axes[idx, 2].axis('off')
                axes[idx, 2].set_title(f"Prediction Slice {i}")

            plt.tight_layout()
            img_buf = BytesIO()
            plt.savefig(img_buf, format='png', bbox_inches='tight')
            img_buf.seek(0)
            plt.close()

            return HttpResponse(img_buf, content_type='image/png')

        except Exception as e:
            return Response({"error": str(e)}, status=500)

        finally:
            os.remove(tmp_path)
