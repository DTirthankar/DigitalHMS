from django.contrib import admin

from .models import PatientPrediction, Hospital, EmergencyRequest


# Register patient disease predictions
admin.site.register(PatientPrediction)


# Register hospitals
admin.site.register(Hospital)


# Register SOS emergency requests
admin.site.register(EmergencyRequest)