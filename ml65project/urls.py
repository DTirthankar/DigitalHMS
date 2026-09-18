from django.contrib import admin
from django.urls import path
from ml65app import views

urlpatterns = [
    path('admin/', admin.site.urls),   # Django admin panel
    path('', views.home, name='home'),
    path('sos/', views.sos, name='sos'),
    path('about/', views.about, name='about'),
    path('predict/', views.predict, name='predict'),
    path('contact/', views.contact, name='contact'),
    path('history/', views.history, name='history'),
    path('dashboard/', views.dashboard, name='dashboard'),

    # Patient Login/Register
    path('patient/login/', views.patient_login, name='patient_login'),
    path('patient/register/', views.patient_register, name='patient_register'),

    # Members Login
    path('members/login/', views.members_login, name='members_login'),
    path('members/register/', views.members_register, name='members_register'),
]