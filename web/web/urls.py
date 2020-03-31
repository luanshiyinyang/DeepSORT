from django.contrib import admin
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.urls import path
from . import views
from . import settings
from django.conf.urls.static import static
from django.conf.urls import url

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.upload),
    url('video/', views.stream_video, name="video")
]

urlpatterns += staticfiles_urlpatterns()
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
