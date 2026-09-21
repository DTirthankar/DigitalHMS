from django.db import models
from django.utils import timezone

class PatientPrediction(models.Model):
    patient_name = models.CharField(max_length=100)
    patient = models.ForeignKey(
        'PatientProfile',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='predictions'
    )
    patient_age = models.IntegerField(null=True, blank=True)
    patient_gender = models.CharField(
        max_length=10,
        choices=[('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other')],
        default='Other'
    )
    symptom1 = models.CharField(max_length=100, blank=True)
    symptom2 = models.CharField(max_length=100, blank=True)
    symptom3 = models.CharField(max_length=100, blank=True)
    symptom4 = models.CharField(max_length=100, blank=True)
    symptom5 = models.CharField(max_length=100, blank=True)

    predicted_disease = models.CharField(max_length=100)
    confidence = models.FloatField(default=0.0)
    top3_predictions = models.CharField(max_length=255, blank=True)
    
    urgency = models.CharField(max_length=20, blank=True, default='Moderate')   # NEW
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.patient_name} - {self.predicted_disease} ({self.created_at.strftime('%d-%b-%Y')})"

class Hospital(models.Model):
    name = models.CharField(max_length=150)
    address = models.CharField(max_length=255)
    latitude = models.FloatField()
    longitude = models.FloatField()
    phone = models.CharField(max_length=20)
    emergency_contact = models.CharField(max_length=20, blank=True)

    total_beds = models.IntegerField(default=0)
    available_beds = models.IntegerField(default=0)

    doctor_name = models.CharField(max_length=100, blank=True)
    doctor_specialization = models.CharField(max_length=100, blank=True)
    doctor_available = models.BooleanField(default=False)

    specializations = models.CharField(max_length=255, blank=True)

    rating = models.FloatField(default=3.0)
    def __str__(self):
        return self.name

class UserProfile(models.Model):
    ROLE_CHOICES = [
    ('PATIENT', 'Patient'),
    ('HOSPITAL', 'Hospital'),
    ('DOCTOR', 'Doctor'),
    ('HEALTH_WORKER', 'Health Worker'),
    ('MEDICAL_SHOP', 'Medical Shop'),
    ('BLOOD_BANK', 'Blood Bank'),
]
    user = models.OneToOneField(
        'auth.User',
        on_delete=models.CASCADE
    )

    role = models.CharField(
        max_length=20,
        choices=ROLE_CHOICES
    )

    mobile = models.CharField(max_length=15, blank=True, default='')

    def __str__(self):
        return f"{self.user.username} - {self.role}"


# =========================
# Patient Profile
# =========================

class PatientProfile(models.Model):

    user = models.OneToOneField(
        'auth.User',
        on_delete=models.CASCADE
    )

    full_name = models.CharField(max_length=150)

    date_of_birth = models.DateField(
        null=True,
        blank=True
    )

    gender = models.CharField(
        max_length=10
    )

    blood_group = models.CharField(
        max_length=5,
        blank=True
    )

    government_id_type = models.CharField(
        max_length=30,
        blank=True
    )

    government_id = models.CharField(
        max_length=100,
        blank=True
    )

    mobile = models.CharField(
        max_length=15
    )

    email = models.EmailField(blank=True)

    address = models.TextField(
        blank=True
    )

    emergency_contact_name = models.CharField(
        max_length=150,
        blank=True
    )

    emergency_contact_number = models.CharField(
        max_length=15,
        blank=True
    )

    # Medical information is optional
    symptoms = models.TextField(
        blank=True
    )

    medical_history = models.TextField(
        blank=True
    )

    allergies = models.TextField(
        blank=True
    )

    current_medications = models.TextField(
        blank=True
    )

    insurance_company = models.CharField(
        max_length=150,
        blank=True
    )

    policy_number = models.CharField(
        max_length=100,
        blank=True
    )

    payment_preference = models.CharField(
        max_length=20,
        blank=True
    )

    created_at = models.DateTimeField(
        default=timezone.now
    )

    def __str__(self):
        return self.full_name

class EmergencyRequest(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('ACCEPTED', 'Accepted'),
        ('REJECTED', 'Rejected'),
        ('COMPLETED', 'Completed'),
    ]

    patient_name = models.CharField(
        max_length=100,
        default='Emergency Patient'
    )

    latitude = models.FloatField()
    longitude = models.FloatField()

    hospital = models.ForeignKey(
        Hospital,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )

    doctor_name = models.CharField(
        max_length=100,
        blank=True
    )

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='PENDING'
    )

    eta_minutes = models.IntegerField(
        null=True,
        blank=True
    )

    created_at = models.DateTimeField(
        default=timezone.now
    )

    updated_at = models.DateTimeField(
        auto_now=True
    )

    def __str__(self):
        return f"SOS - {self.patient_name} - {self.status}"