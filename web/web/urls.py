from django.contrib import admin
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.urls import path
from . import views
from . import settings


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.upload),

]

urlpatterns += staticfiles_urlpatterns()
