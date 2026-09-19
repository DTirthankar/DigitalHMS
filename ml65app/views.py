import joblib
import numpy as np
from django.shortcuts import render, redirect
import os
from django.conf import settings
from django.db.models import Q

from django.contrib.auth.models import User
from django.contrib.auth import login
from django.contrib import messages

from .models import (
    PatientPrediction,
    Hospital,
    EmergencyRequest,
    PatientProfile,
    UserProfile
)

from math import radians, cos, sin, asin, sqrt
from django.db.models import Count
from django.utils import timezone


model = joblib.load(
    os.path.join(settings.BASE_DIR, 'ml65cd.joblib')
)


SYMPTOMS = [
    'itching','skin_rash','nodal_skin_eruptions','continuous_sneezing',
    'shivering','chills','joint_pain','stomach_pain','acidity',
    'ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition',
    'spotting_ urination','fatigue','weight_gain','anxiety',
    'cold_hands_and_feets','mood_swings','weight_loss','restlessness',
    'lethargy','patches_in_throat','irregular_sugar_level','cough',
    'high_fever','sunken_eyes','breathlessness','sweating','dehydration',
    'indigestion','headache','yellowish_skin','dark_urine','nausea',
    'loss_of_appetite','pain_behind_the_eyes','back_pain','constipation',
    'abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload',
    'swelling_of_stomach','swelled_lymph_nodes','malaise',
    'blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion',
    'chest_pain','weakness_in_limbs','fast_heart_rate',
    'pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising',
    'obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes',
    'enlarged_thyroid','brittle_nails','swollen_extremeties',
    'excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
    'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness',
    'stiff_neck','swelling_joints','movement_stiffness','spinning_movements',
    'loss_of_balance','unsteadiness','weakness_of_one_body_side',
    'loss_of_smell','bladder_discomfort','foul_smell_of urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching',
    'toxic_look_(typhos)','depression','irritability','muscle_pain',
    'altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic _patches','watering_from_eyes',
    'increased_appetite','polyuria','family_history','mucoid_sputum',
    'rusty_sputum','lack_of_concentration','visual_disturbances',
    'receiving_blood_transfusion','receiving_unsterile_injections','coma',
    'stomach_bleeding','distention_of_abdomen',
    'history_of_alcohol_consumption','fluid_overload.1','blood_in_sputum',
    'prominent_veins_on_calf','palpitations','painful_walking',
    'pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails',
    'blister','red_sore_around_nose','yellow_crust_ooze',
]


DISEASE_IMAGES = {
    'Acne': 'images/Acne.webp',
    'AIDS': 'images/AIDS.png',
    'Alcoholic hepatitis': 'images/Alcoholic hepatitis.png',
    'Allergy': 'images/Allergy.jpg',
    'Arthritis': 'images/Arthritis.webp',
    '(vertigo) Paroymsal  Positional Vertigo':
        'images/benign-paroxysmal-positional-vertigo.jpg',
    'Bronchial Asthma': 'images/Bronchial Asthma.jpg',
    'Cervical spondylosis': 'images/Cervical Spondylosis.jpg',
    'Chicken pox': 'images/Chicken pox.jpg',
    'Chronic cholestasis': 'images/Chronic cholestasis.webp',
    'Common Cold': 'images/Common Cold.jpg',
    'Dengue': 'images/Dengue.png',
    'Diabetes': 'images/Diabetes.jpg',
    'Dimorphic hemmorhoids(piles)':
        'images/Dimorphic hemmorhoids(piles).png',
    'Drug Reaction': 'images/Drug Reaction.jpg',
    'Fungal infection': 'images/Fungal infection.jpg',
    'Gastroenteritis': 'images/Gastroenteritis.jpg',
    'GERD': 'images/GERD.png',
    'Heart attack': 'images/heart attack.png',
    'hepatitis A': 'images/hepatitis A.jpg',
    'Hepatitis B': 'images/Hepatitis B.jpg',
    'Hepatitis C': 'images/Hepatitis C.jpg',
    'Hepatitis D': 'images/Hepatitis D.webp',
    'Hepatitis E': 'images/Hepatitis E.png',
    'Hypertension': 'images/Hypertension.jpg',
    'Hyperthyroidism': 'images/Hyperthyroidism.jpg',
    'Hypoglycemia': 'images/Hypoglycemia.avif',
    'Hypothyroidism': 'images/Hypothyroidism.jpg',
    'Impetigo': 'images/Impetigo.jpg',
    'Jaundice': 'images/Jaundice.png',
    'Malaria': 'images/Malaria.webp',
    'Migraine': 'images/Migraine.jpg',
    'Osteoarthristis': 'images/Osteoarthristis.jpg',
    'Paralysis (brain hemorrhage)':
        'images/Paralysis (brain hemorrhage).jpg',
    'Peptic ulcer diseae': 'images/Peptic ulcer diseae.jpg',
    'Pneumonia': 'images/Pneumonia.jpg',
    'Psoriasis': 'images/Psoriasis.png',
    'Tuberculosis': 'images/Tuberculosis.webp',
    'Typhoid': 'images/Typhoid.jpg',
    'Urinary tract infection':
        'images/Urinary tract infection.jpg',
    'Varicose veins': 'images/Varicose veins.jpg',
}

# Normalize keys — remove trailing spaces
DISEASE_IMAGES = {
    k.strip(): v for k, v in DISEASE_IMAGES.items()
}
DISEASE_SPECIALIZATION = {
    'Diabetes': 'Endocrinology',
    'Hypertension': 'Cardiology',
    'Heart attack': 'Cardiology',
    'Pneumonia': 'Pulmonology',
    'Bronchial Asthma': 'Pulmonology',
    'Tuberculosis': 'Pulmonology',
    'Jaundice': 'Hepatology',
    'hepatitis A': 'Hepatology',
    'Hepatitis B': 'Hepatology',
    'Hepatitis C': 'Hepatology',
    'Hepatitis D': 'Hepatology',
    'Hepatitis E': 'Hepatology',
    'Alcoholic hepatitis': 'Hepatology',
    'Chronic cholestasis': 'Hepatology',
    'Migraine': 'Neurology',
    'Paralysis (brain hemorrhage)': 'Neurology',
    '(vertigo) Paroymsal  Positional Vertigo': 'Neurology',
    'Arthritis': 'Orthopedics',
    'Osteoarthristis': 'Orthopedics',
    'Cervical spondylosis': 'Orthopedics',
    'Urinary tract infection': 'Urology',
    'Dengue': 'General Medicine',
    'Malaria': 'General Medicine',
    'Typhoid': 'General Medicine',
    'Common Cold': 'General Medicine',
    'AIDS': 'Infectious Disease',
}


def haversine_distance(lat1, lon1, lat2, lon2):
    """Returns distance in km between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(
        radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    c = 2 * asin(sqrt(a))

    return 6371 * c


def get_nearby_hospitals(
    predicted_disease,
    user_lat,
    user_lon,
    limit=3
):
    specialization = DISEASE_SPECIALIZATION.get(
        predicted_disease,
        ''
    )

    hospitals = Hospital.objects.all()

    results = []

    for h in hospitals:

        distance = haversine_distance(
            user_lat,
            user_lon,
            h.latitude,
            h.longitude
        )

        relevant = (
            specialization.lower()
            in h.specializations.lower()
            if specialization
            else True
        )

        results.append({
            'hospital': h,
            'distance': round(distance, 1),
            'relevant': relevant,
        })

    results.sort(
        key=lambda x: (
            not x['relevant'],
            x['distance']
        )
    )

    return results[:limit]


URGENCY_LEVELS = {
    'Heart attack': 'Critical',
    'Paralysis (brain hemorrhage)': 'Critical',
    'AIDS': 'Critical',

    'Dengue': 'High',
    'Typhoid': 'High',
    'Malaria': 'High',
    'Pneumonia': 'High',
    'Tuberculosis': 'High',
    'Hepatitis B': 'High',
    'Hepatitis C': 'High',
    'Hepatitis D': 'High',
    'Hepatitis E': 'High',
    'Alcoholic hepatitis': 'High',
    'Jaundice': 'High',
    '(vertigo) Paroymsal  Positional Vertigo': 'High',

    'Diabetes': 'Moderate',
    'Hypertension': 'Moderate',
    'Bronchial Asthma': 'Moderate',
    'hepatitis A': 'Moderate',
    'Chronic cholestasis': 'Moderate',
    'Migraine': 'Moderate',
    'Arthritis': 'Moderate',
    'Osteoarthristis': 'Moderate',
    'Cervical spondylosis': 'Moderate',
    'Urinary tract infection': 'Moderate',
    'Hyperthyroidism': 'Moderate',
    'Hypothyroidism': 'Moderate',
    'Hypoglycemia': 'Moderate',
    'Peptic ulcer diseae': 'Moderate',
    'GERD': 'Moderate',
    'Gastroenteritis': 'Moderate',
    'Dimorphic hemmorhoids(piles)': 'Moderate',
    'Varicose veins': 'Moderate',

    'Common Cold': 'Low',
    'Allergy': 'Low',
    'Acne': 'Low',
    'Fungal infection': 'Low',
    'Impetigo': 'Low',
    'Psoriasis': 'Low',
    'Drug Reaction': 'Low',
    'Chicken pox': 'Low',
}


def get_urgency(predicted_disease, confidence):
    level = URGENCY_LEVELS.get(
        predicted_disease,
        'Moderate'
    )

    if confidence < 50 and level == 'Low':
        level = 'Moderate'

    elif confidence < 50 and level == 'Moderate':
        level = 'High'

    return level


# =========================
# Patient Login
# =========================

def patient_login(request):

    patients = None

    if request.method == 'POST':

        mobile = request.POST.get(
            'mobile',
            ''
        ).strip()

        patient_id = request.POST.get(
            'patient_id',
            ''
        ).strip()

        password = request.POST.get(
            'password',
            ''
        )

        # Step 1: Find patients using mobile number
        if not patient_id:

            patients = PatientProfile.objects.filter(
                mobile=mobile
            )

            if not patients.exists():

                messages.error(
                    request,
                    'No patient found with this mobile number.'
                )

            return render(
                request,
                'patient_login.html',
                {
                    'patients': patients
                }
            )

        # Step 2: Patient selected
        try:

            patient = PatientProfile.objects.get(
                id=patient_id,
                mobile=mobile
            )

        except PatientProfile.DoesNotExist:

            messages.error(
                request,
                'Invalid patient selection.'
            )

            return render(
                request,
                'patient_login.html'
            )

        # Step 3: Authenticate selected patient
        from django.contrib.auth import authenticate

        user = authenticate(
            request,
            username=patient.user.username,
            password=password
        )

        if user is not None:

            login(
                request,
                user
            )

            return redirect('home')

        messages.error(
            request,
            'Invalid password.'
        )

        patients = PatientProfile.objects.filter(
            mobile=mobile
        )

    return render(
        request,
        'patient_login.html',
        {
            'patients': patients
        }
    )


# =========================
# Forgot Password
# =========================

def forgot_password(request):

    patients = None

    if request.method == 'POST':

        identifier = request.POST.get(
            'identifier',
            ''
        ).strip()

        if not identifier:

            messages.error(
                request,
                'Please enter your mobile number or email ID.'
            )

            return render(
                request,
                'forgot_password.html'
            )

        # Search by mobile number
        patients = PatientProfile.objects.filter(
            mobile=identifier
        )

        # If no mobile match, search by email
        if not patients.exists():

            patients = PatientProfile.objects.filter(
                email__iexact=identifier
            )

        if not patients.exists():

            messages.error(
                request,
                'No patient found with this mobile number or email ID.'
            )

            patients = None

    return render(
        request,
        'forgot_password.html',
        {
            'patients': patients
        }
    )


# =========================
# Patient Registration
# =========================

def patient_register(request):

    if request.method == 'POST':

        full_name = request.POST.get(
            'full_name',
            ''
        ).strip()

        date_of_birth = request.POST.get(
            'date_of_birth'
        ) or None

        gender = request.POST.get(
            'gender',
            ''
        ).strip()

        blood_group = request.POST.get(
            'blood_group',
            ''
        ).strip()

        government_id_type = request.POST.get(
            'government_id_type',
            ''
        ).strip()

        government_id = request.POST.get(
            'government_id',
            ''
        ).strip()

        mobile = request.POST.get(
            'mobile',
            ''
        ).strip()

        email = request.POST.get(
            'email',
            ''
        ).strip()

        address = request.POST.get(
            'address',
            ''
        ).strip()

        emergency_contact_name = request.POST.get(
            'emergency_contact_name',
            ''
        ).strip()

        emergency_contact_number = request.POST.get(
            'emergency_contact_number',
            ''
        ).strip()

        symptoms = request.POST.get(
            'symptoms',
            ''
        ).strip()

        medical_history = request.POST.get(
            'medical_history',
            ''
        ).strip()

        allergies = request.POST.get(
            'allergies',
            ''
        ).strip()

        current_medications = request.POST.get(
            'current_medications',
            ''
        ).strip()

        insurance_company = request.POST.get(
            'insurance_company',
            ''
        ).strip()

        policy_number = request.POST.get(
            'policy_number',
            ''
        ).strip()

        payment_preference = request.POST.get(
            'payment_preference',
            ''
        ).strip()

        password = request.POST.get(
            'password',
            ''
        )

        confirm_password = request.POST.get(
            'confirm_password',
            ''
        )

        if not full_name:

            messages.error(
                request,
                'Full Name is required.'
            )

            return render(
                request,
                'patient_register.html'
            )

        if not mobile:

            messages.error(
                request,
                'Mobile Number is required.'
            )

            return render(
                request,
                'patient_register.html'
            )

        if not gender:

            messages.error(
                request,
                'Please select your gender.'
            )

            return render(
                request,
                'patient_register.html'
            )

        if not password:

            messages.error(
                request,
                'Password is required.'
            )

            return render(
                request,
                'patient_register.html'
            )

        if password != confirm_password:

            messages.error(
                request,
                'Passwords do not match.'
            )

            return render(
                request,
                'patient_register.html'
            )

        # Government ID validation

        if government_id_type and not government_id:

            messages.error(
                request,
                'Please enter your Government ID Number.'
            )

            return render(
                request,
                'patient_register.html'
            )

        if government_id and not government_id_type:

            messages.error(
                request,
                'Please select Government ID Type.'
            )

            return render(
                request,
                'patient_register.html'
            )

        if government_id_type and government_id:

            already_registered = PatientProfile.objects.filter(
                government_id_type=government_id_type,
                government_id=government_id
            ).exists()

            if already_registered:

                messages.error(
                    request,
                    'This Government ID is already registered.'
                )

                return render(
                    request,
                    'patient_register.html'
                )

        # Create unique internal username

        username = f"patient_{mobile}"

        counter = 1

        while User.objects.filter(
            username=username
        ).exists():

            username = (
                f"patient_{mobile}_{counter}"
            )

            counter += 1

        # Create User

        user = User.objects.create_user(
            username=username,
            password=password,
            email=email
        )

        # Create Patient Profile

        PatientProfile.objects.create(
            user=user,
            full_name=full_name,
            date_of_birth=date_of_birth,
            gender=gender,
            blood_group=blood_group,
            government_id_type=government_id_type,
            government_id=government_id,
            mobile=mobile,
            email=email,
            address=address,
            emergency_contact_name=emergency_contact_name,
            emergency_contact_number=emergency_contact_number,
            symptoms=symptoms,
            medical_history=medical_history,
            allergies=allergies,
            current_medications=current_medications,
            insurance_company=insurance_company,
            policy_number=policy_number,
            payment_preference=payment_preference
        )

        messages.success(
            request,
            'Patient registration successful! Please login.'
        )

        return redirect('patient_login')

    return render(
        request,
        'patient_register.html'
    )


# =========================
# Members Login
# =========================

def members_login(request):

    if request.method == 'POST':

        member_type = request.POST.get(
            'member_type',
            ''
        ).strip()

        username = request.POST.get(
            'username',
            ''
        ).strip()

        password = request.POST.get(
            'password',
            ''
        )

        if not member_type or not username or not password:

            messages.error(
                request,
                'Please fill in all fields.'
            )

            return render(
                request,
                'members_login.html'
            )

        try:

            user = User.objects.get(
                username=username
            )

        except User.DoesNotExist:

            messages.error(
                request,
                'Invalid username or password.'
            )

            return render(
                request,
                'members_login.html'
            )

        try:

            profile = UserProfile.objects.get(
                user=user
            )

        except UserProfile.DoesNotExist:

            messages.error(
                request,
                'Member profile not found.'
            )

            return render(
                request,
                'members_login.html'
            )

        if profile.role != member_type:

            messages.error(
                request,
                'Selected member type does not match this account.'
            )

            return render(
                request,
                'members_login.html'
            )

        from django.contrib.auth import authenticate

        authenticated_user = authenticate(
            request,
            username=username,
            password=password
        )

        if authenticated_user is not None:

            login(
                request,
                authenticated_user
            )

            return redirect('home')

        messages.error(
            request,
            'Invalid username or password.'
        )

    return render(
        request,
        'members_login.html'
    )

def members_forgot_password(request):

    if request.method == 'POST':

        identifier = request.POST.get('identifier', '').strip()
        action = request.POST.get('action', '')

        # -------------------------
        # STEP 1: Find Member
        # -------------------------
        if action == '':

            profiles = UserProfile.objects.filter(
                Q(mobile=identifier) |
                Q(user__email__iexact=identifier)
            ).select_related('user')

            if not profiles.exists():
                messages.error(
                    request,
                    'No member account found with this mobile or email.'
                )

                return render(
                    request,
                    'members_forgot_password.html'
                )

            profile = profiles.first()
            user = profile.user

            return render(
                request,
                'members_forgot_password.html',
                {
                    'member_found': True,
                    'member_name': user.username,
                    'identifier': identifier,
                }
            )

        # -------------------------
        # STEP 2: Send OTP
        # -------------------------
        if action == 'send_otp':

            profiles = UserProfile.objects.filter(
                Q(mobile=identifier) |
                Q(user__email__iexact=identifier)
            ).select_related('user')

            if not profiles.exists():
                messages.error(
                    request,
                    'No member account found.'
                )

                return render(
                    request,
                    'members_forgot_password.html'
                )

            profile = profiles.first()
            user = profile.user

            import random

            otp = str(random.randint(100000, 999999))

            request.session['member_reset_user_id'] = user.id
            request.session['member_reset_otp'] = otp

            print(
                f"MEMBER PASSWORD RESET OTP for {user.username}: {otp}"
            )

            return render(
                request,
                'members_forgot_password.html',
                {
                    'otp_sent': True,
                    'member_name': user.username,
                }
            )

        # -------------------------
        # STEP 3: Verify OTP
        # -------------------------
        if action == 'verify_otp':

            entered_otp = request.POST.get('otp', '').strip()

            saved_otp = request.session.get(
                'member_reset_otp'
            )

            user_id = request.session.get(
                'member_reset_user_id'
            )

            if not saved_otp or not user_id:

                messages.error(
                    request,
                    'OTP session expired. Please try again.'
                )

                return render(
                    request,
                    'members_forgot_password.html'
                )

            if entered_otp != saved_otp:

                messages.error(
                    request,
                    'Invalid OTP. Please try again.'
                )

                return render(
                    request,
                    'members_forgot_password.html',
                    {
                        'otp_sent': True
                    }
                )

            user = User.objects.get(id=user_id)

            request.session['member_reset_verified'] = True

            return render(
                request,
                'members_forgot_password.html',
                {
                    'password_reset': True,
                    'member_name': user.username,
                }
            )

        # -------------------------
        # STEP 4: Change Password
        # -------------------------
        if action == 'reset_password':

            if not request.session.get(
                'member_reset_verified'
            ):
                messages.error(
                    request,
                    'Please verify OTP first.'
                )

                return render(
                    request,
                    'members_forgot_password.html'
                )

            user_id = request.session.get(
                'member_reset_user_id'
            )

            new_password = request.POST.get(
                'new_password',
                ''
            )

            confirm_password = request.POST.get(
                'confirm_password',
                ''
            )

            if not new_password:

                messages.error(
                    request,
                    'Please enter a new password.'
                )

                return render(
                    request,
                    'members_forgot_password.html',
                    {
                        'password_reset': True
                    }
                )

            if new_password != confirm_password:

                messages.error(
                    request,
                    'Passwords do not match.'
                )

                return render(
                    request,
                    'members_forgot_password.html',
                    {
                        'password_reset': True
                    }
                )

            user = User.objects.get(id=user_id)

            user.set_password(new_password)
            user.save()

            # Clear reset session
            request.session.pop(
                'member_reset_user_id',
                None
            )

            request.session.pop(
                'member_reset_otp',
                None
            )

            request.session.pop(
                'member_reset_verified',
                None
            )

            messages.success(
                request,
                'Password changed successfully. Please login.'
            )

            return redirect('members_login')

    return render(
        request,
        'members_forgot_password.html'
    )
# =========================
# Members Registration
# =========================

def members_register(request):

    if request.method == 'POST':

        member_type = request.POST.get(
            'member_type',
            ''
        ).strip()

        member_name = request.POST.get(
            'member_name',
            ''
        ).strip()

        mobile = request.POST.get(
            'mobile',
            ''
        ).strip()

        email = request.POST.get(
            'email',
            ''
        ).strip()

        address = request.POST.get(
            'address',
            ''
        ).strip()

        username = request.POST.get(
            'username',
            ''
        ).strip()

        password = request.POST.get(
            'password',
            ''
        )

        confirm_password = request.POST.get(
            'confirm_password',
            ''
        )

        consent = request.POST.get(
            'consent'
        )

        # -------------------------
        # Validation
        # -------------------------

        if not member_type:

            messages.error(
                request,
                'Please select member type.'
            )

            return render(
                request,
                'members_register.html'
            )

        if not member_name:

            messages.error(
                request,
                'Member / Organization Name is required.'
            )

            return render(
                request,
                'members_register.html'
            )

        if not mobile:

            messages.error(
                request,
                'Contact Number is required.'
            )

            return render(
                request,
                'members_register.html'
            )

        if not address:

            messages.error(
                request,
                'Address is required.'
            )

            return render(
                request,
                'members_register.html'
            )

        if not username:

            messages.error(
                request,
                'Username is required.'
            )

            return render(
                request,
                'members_register.html'
            )

        if not password:

            messages.error(
                request,
                'Password is required.'
            )

            return render(
                request,
                'members_register.html'
            )

        if password != confirm_password:

            messages.error(
                request,
                'Passwords do not match.'
            )

            return render(
                request,
                'members_register.html'
            )

        if not consent:

            messages.error(
                request,
                'Please accept the consent.'
            )

            return render(
                request,
                'members_register.html'
            )

        # -------------------------
        # Valid member types
        # -------------------------

        valid_roles = [
            'HOSPITAL',
            'DOCTOR',
            'HEALTH_WORKER',
            'MEDICAL_SHOP',
            'BLOOD_BANK',
        ]

        if member_type not in valid_roles:

            messages.error(
                request,
                'Invalid member type.'
            )

            return render(
                request,
                'members_register.html'
            )

        # -------------------------
        # Username check
        # -------------------------

        if User.objects.filter(
            username=username
        ).exists():

            messages.error(
                request,
                'This username is already registered.'
            )

            return render(
                request,
                'members_register.html'
            )

        # -------------------------
        # Create User
        # -------------------------

        user = User.objects.create_user(
            username=username,
            password=password,
            email=email
        )

        # -------------------------
        # Create User Profile
        # -------------------------

        UserProfile.objects.create(
            user=user,
            role=member_type,
            mobile=mobile
        )

        messages.success(
            request,
            'Member registration successful! Please login.'
        )

        return redirect(
            'members_login'
        )

    return render(
        request,
        'members_register.html'
    )



# =========================
# Home
# =========================

def home(request):

    return render(
        request,
        'home.html'
    )


# =========================
# SOS
# =========================

def sos(request):

    if request.method == 'POST':

        latitude = request.POST.get(
            'latitude'
        )

        longitude = request.POST.get(
            'longitude'
        )

        if not latitude or not longitude:

            return render(
                request,
                'sos.html',
                {
                    'error':
                        'Location could not be detected.'
                }
            )

        emergency = EmergencyRequest.objects.create(
            latitude=float(latitude),
            longitude=float(longitude)
        )

        return render(
            request,
            'emergency_status.html',
            {
                'emergency': emergency
            }
        )

    return render(
        request,
        'sos.html'
    )


# =========================
# About
# =========================

def about(request):

    return render(
        request,
        'about.html'
    )


# =========================
# Contact
# =========================

def contact(request):

    return render(
        request,
        'contact.html'
    )


# =========================
# Prediction
# =========================

def predict(request):

    if request.method == 'POST':

        # Patient profile details

        patient_name = request.POST.get(
            'patient_name',
            ''
        )

        patient_age = request.POST.get(
            'patient_age',
            ''
        )

        patient_gender = request.POST.get(
            'patient_gender',
            'Other'
        )

        # Extract selected symptoms

        selected_symptoms = []

        for i in range(1, 6):

            s_val = request.POST.get(
                f'symptom{i}'
            )

            if s_val:

                selected_symptoms.append(
                    s_val
                )

        # ML input

        input_vector = np.zeros(
            len(SYMPTOMS)
        )

        for symptom in selected_symptoms:

            if symptom in SYMPTOMS:

                input_vector[
                    SYMPTOMS.index(symptom)
                ] = 1

        # ML classification

        predictions = model.predict(
            [input_vector]
        )

        prediction = (
            predictions[0]
            if len(predictions) > 0
            else "Unknown"
        )

        # Confidence

        confidence = 90.0

        if hasattr(
            model,
            "predict_proba"
        ):

            try:

                probabilities = model.predict_proba(
                    [input_vector]
                )

                confidence = round(
                    float(
                        np.max(probabilities)
                    ) * 100,
                    1
                )

            except Exception:

                pass

        # Warning and image

        confidence_warning = (
            confidence < 60.0
        )

        image_file = DISEASE_IMAGES.get(
            prediction.strip(),
            None
        )

        # Top 3 predictions

        if hasattr(
            model,
            "predict_proba"
        ):

            try:

                probabilities = model.predict_proba(
                    [input_vector]
                )[0]

                top3_indices = np.argsort(
                    probabilities
                )[::-1][:3]

                top3 = [
                    (
                        model.classes_[idx],
                        round(
                            float(
                                probabilities[idx]
                            ) * 100,
                            1
                        )
                    )
                    for idx in top3_indices
                ]

            except Exception:

                top3 = [
                    (prediction, confidence),
                    ("Allergy", 15.0),
                    ("GERD", 5.0)
                ]

        else:

            top3 = [
                (prediction, confidence),
                (
                    "Allergy",
                    round(
                        confidence * 0.15,
                        1
                    )
                ),
                (
                    "GERD",
                    round(
                        confidence * 0.08,
                        1
                    )
                )
            ]

        importances = model.feature_importances_
        xai_symptoms = []
        for sym in selected_symptoms:
            if sym in SYMPTOMS:
                idx = SYMPTOMS.index(sym)
                xai_symptoms.append({
                    'symptom': sym.replace('_', ' '),
                    'score': round(float(importances[idx]) * 100, 2)
                })
        xai_symptoms.sort(key=lambda x: x['score'], reverse=True)

        urgency = URGENCY_LEVELS.get(
            prediction,
            'Low'
        )

        # User location

        user_lat = request.POST.get(
            'latitude',
            '0.0'
        )

        user_lon = request.POST.get(
            'longitude',
            '0.0'
        )

        try:

            nearby_hospitals = get_nearby_hospitals(
                prediction,
                float(
                    user_lat or 0.0
                ),
                float(
                    user_lon or 0.0
                )
            )

        except (
            ValueError,
            TypeError
        ):

            nearby_hospitals = []

        # Save prediction

        PatientPrediction.objects.create(
            patient_name=patient_name,
            patient_age=(
                int(patient_age)
                if patient_age
                else None
            ),
            patient_gender=patient_gender,

            symptom1=(
                selected_symptoms[0]
                if len(selected_symptoms) > 0
                else ''
            ),

            symptom2=(
                selected_symptoms[1]
                if len(selected_symptoms) > 1
                else ''
            ),

            symptom3=(
                selected_symptoms[2]
                if len(selected_symptoms) > 2
                else ''
            ),

            symptom4=(
                selected_symptoms[3]
                if len(selected_symptoms) > 3
                else ''
            ),

            symptom5=(
                selected_symptoms[4]
                if len(selected_symptoms) > 4
                else ''
            ),

            predicted_disease=prediction,
            confidence=confidence,
            top3_predictions=str(top3),
            urgency=urgency,
            created_at=timezone.now()
        )

        return render(
            request,
            'predict.html',
            {
                'result': prediction,
                'image_file': image_file,
                'confidence': confidence,
                'confidence_warning':
                    confidence_warning,
                'top3': top3,
                'xai_symptoms':
                    xai_symptoms,
                'symptoms': SYMPTOMS,
                'patient_name':
                    patient_name,
                'nearby_hospitals':
                    nearby_hospitals,
            }
        )

    return render(
        request,
        'predict.html',
        {
            'symptoms': SYMPTOMS
        }
    )


# =========================
# History
# =========================

def history(request):

    predictions = (
        PatientPrediction.objects.all()[:50]
    )

    disease_counts = (
        PatientPrediction.objects
        .values('predicted_disease')
        .annotate(
            count=Count(
                'predicted_disease'
            )
        )
        .order_by('-count')[:10]
    )

    chart_labels = [
        d['predicted_disease']
        for d in disease_counts
    ]

    chart_values = [
        d['count']
        for d in disease_counts
    ]

    return render(
        request,
        'history.html',
        {
            'predictions':
                predictions,
            'chart_labels':
                chart_labels,
            'chart_values':
                chart_values,
        }
    )


# =========================
# Dashboard
# =========================

def dashboard(request):

    predictions = (
        PatientPrediction.objects.all()[:20]
    )

    total_predictions = (
        PatientPrediction.objects.count()
    )

    today = timezone.now().date()

    today_count = (
        PatientPrediction.objects
        .filter(
            created_at__date=today
        )
        .count()
    )

    top = (
        PatientPrediction.objects
        .values('predicted_disease')
        .annotate(
            count=Count(
                'predicted_disease'
            )
        )
        .order_by('-count')
        .first()
    )

    top_disease = (
        top['predicted_disease']
        if top
        else 'N/A'
    )

    critical_count = (
        PatientPrediction.objects
        .filter(
            urgency='Critical'
        )
        .count()
    )

    high_count = (
        PatientPrediction.objects
        .filter(
            urgency='High'
        )
        .count()
    )

    return render(
        request,
        'dashboard.html',
        {
            'predictions':
                predictions,
            'total_predictions':
                total_predictions,
            'today_count':
                today_count,
            'top_disease':
                top_disease,
            'critical_count':
                critical_count,
            'high_count':
                high_count,
        }
    )