from django.contrib import admin
from .models import PatientPrediction, Hospital

admin.site.register(PatientPrediction)
admin.site.register(Hospital)