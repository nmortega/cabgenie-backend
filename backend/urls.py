# backend/urls.py

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static  # ✅ Needed for media

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),  # ✅ This loads your app-level routes
]

# ✅ Add this at the bottom (only in the project-level urls.py)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
