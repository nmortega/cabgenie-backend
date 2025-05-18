from django.urls import path
from .views import ImageUploadView

urlpatterns = [
    path('upload_image/', ImageUploadView.as_view(), name='upload_image'),
]