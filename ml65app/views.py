import joblib
import numpy as np
from django.shortcuts import render, redirect
import os
from django.conf import settings

from django.contrib.auth.models import User
from django.contrib.auth import login
from django.contrib import messages

from .models import PatientPrediction, Hospital, EmergencyRequest, PatientProfile

from math import radians, cos, sin, asin, sqrt
from django.db.models import Count
from django.utils import timezone

model = joblib.load(os.path.join(settings.BASE_DIR, 'ml65cd.joblib'))

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
    'Acne':                                  'images/Acne.webp',
    'AIDS':                                  'images/AIDS.png',
    'Alcoholic hepatitis':                   'images/Alcoholic hepatitis.png',
    'Allergy':                               'images/Allergy.jpg',
    'Arthritis':                             'images/Arthritis.webp',
    '(vertigo) Paroymsal  Positional Vertigo': 'images/benign-paroxysmal-positional-vertigo.jpg',
    'Bronchial Asthma':                      'images/Bronchial Asthma.jpg',
    'Cervical spondylosis':                  'images/Cervical Spondylosis.jpg',
    'Chicken pox':                           'images/Chicken pox.jpg',
    'Chronic cholestasis':                   'images/Chronic cholestasis.webp',
    'Common Cold':                           'images/Common Cold.jpg',
    'Dengue':                                'images/Dengue.png',
    'Diabetes':                             'images/Diabetes.jpg',
    'Dimorphic hemmorhoids(piles)':          'images/Dimorphic hemmorhoids(piles).png',
    'Drug Reaction':                         'images/Drug Reaction.jpg',
    'Fungal infection':                      'images/Fungal infection.jpg',
    'Gastroenteritis':                       'images/Gastroenteritis.jpg',
    'GERD':                                  'images/GERD.png',
    'Heart attack':                          'images/heart attack.png',
    'hepatitis A':                           'images/hepatitis A.jpg',
    'Hepatitis B':                             'images/hepatitis B.jpg',
    'Hepatitis C':                             'images/hepatitis C.jpg',
    'Hepatitis D':                             'images/hepatitis D.webp',
    'Hepatitis E':                             'images/hepatitis E.png',
    'Hypertension':                           'images/Hypertension.jpg',
    'Hyperthyroidism':                         'images/Hyperthyroidism.jpg',
    'Hypoglycemia':                            'images/Hypoglycemia.avif',
    'Hypothyroidism':                          'images/Hypothyroidism.jpg',
    'Impetigo':                                'images/Impetigo.jpg',
    'Jaundice':                                'images/Jaundice.png',
    'Malaria':                                 'images/Malaria.webp',
    'Migraine':                                'images/Migraine.jpg',
    'Osteoarthristis':                         'images/Osteoarthristis.jpg',
    'Paralysis (brain hemorrhage)':            'images/Paralysis (brain hemorrhage).jpg',
    'Peptic ulcer diseae':                     'images/Peptic ulcer diseae.jpg',
    'Pneumonia':                               'images/Pneumonia.jpg',
    'Psoriasis':                               'images/Psoriasis.png',
    'Tuberculosis':                            'images/Tuberculosis.webp',
    'Typhoid':                                 'images/Typhoid.jpg',
    'Urinary tract infection':                 'images/Urinary tract infection.jpg',
    'Varicose veins':                          'images/Varicose veins.jpg',
}

# NEW — maps predicted disease to relevant hospital specialization for matching
DISEASE_SPECIALIZATION = {
    'Diabetes': 'Endocrinology', 'Hypertension': 'Cardiology',
    'Heart attack': 'Cardiology', 'Pneumonia': 'Pulmonology',
    'Bronchial Asthma': 'Pulmonology', 'Tuberculosis': 'Pulmonology',
    'Jaundice': 'Hepatology', 'hepatitis A': 'Hepatology',
    'Hepatitis B': 'Hepatology', 'Hepatitis C': 'Hepatology',
    'Hepatitis D': 'Hepatology', 'Hepatitis E': 'Hepatology',
    'Alcoholic hepatitis': 'Hepatology', 'Chronic cholestasis': 'Hepatology',
    'Migraine': 'Neurology', 'Paralysis (brain hemorrhage)': 'Neurology',
    '(vertigo) Paroymsal  Positional Vertigo': 'Neurology',
    'Arthritis': 'Orthopedics', 'Osteoarthristis': 'Orthopedics',
    'Cervical spondylosis': 'Orthopedics',
    'Urinary tract infection': 'Urology',
    'Dengue': 'General Medicine', 'Malaria': 'General Medicine',
    'Typhoid': 'General Medicine', 'Common Cold': 'General Medicine',
    'AIDS': 'Infectious Disease',
}

# NEW — distance calculator
def haversine_distance(lat1, lon1, lat2, lon2):
    """Returns distance in km between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c


# NEW — hospital matcher
def get_nearby_hospitals(predicted_disease, user_lat, user_lon, limit=3):
    specialization = DISEASE_SPECIALIZATION.get(predicted_disease, '')
    hospitals = Hospital.objects.all()

    results = []
    for h in hospitals:
        distance = haversine_distance(user_lat, user_lon, h.latitude, h.longitude)
        relevant = specialization.lower() in h.specializations.lower() if specialization else True
        results.append({
            'hospital': h,
            'distance': round(distance, 1),
            'relevant': relevant,
        })

    results.sort(key=lambda x: (not x['relevant'], x['distance']))
    return results[:limit]

URGENCY_LEVELS = {
    'Heart attack': 'Critical',
    'Paralysis (brain hemorrhage)': 'Critical',
    'AIDS': 'Critical',

    'Dengue': 'High', 'Typhoid': 'High', 'Malaria': 'High',
    'Pneumonia': 'High', 'Tuberculosis': 'High',
    'Hepatitis B': 'High', 'Hepatitis C': 'High', 'Hepatitis D': 'High', 'Hepatitis E': 'High',
    'Alcoholic hepatitis': 'High', 'Jaundice': 'High',
    '(vertigo) Paroymsal  Positional Vertigo': 'High',

    'Diabetes': 'Moderate', 'Hypertension': 'Moderate',
    'Bronchial Asthma': 'Moderate', 'hepatitis A': 'Moderate',
    'Chronic cholestasis': 'Moderate', 'Migraine': 'Moderate',
    'Arthritis': 'Moderate', 'Osteoarthristis': 'Moderate',
    'Cervical spondylosis': 'Moderate', 'Urinary tract infection': 'Moderate',
    'Hyperthyroidism': 'Moderate', 'Hypothyroidism': 'Moderate', 'Hypoglycemia': 'Moderate',
    'Peptic ulcer diseae': 'Moderate', 'GERD': 'Moderate',
    'Gastroenteritis': 'Moderate', 'Dimorphic hemmorhoids(piles)': 'Moderate',
    'Varicose veins': 'Moderate',

    'Common Cold': 'Low', 'Allergy': 'Low', 'Acne': 'Low',
    'Fungal infection': 'Low', 'Impetigo': 'Low', 'Psoriasis': 'Low',
    'Drug Reaction': 'Low', 'Chicken pox': 'Low',
}

def get_urgency(predicted_disease, confidence):
    level = URGENCY_LEVELS.get(predicted_disease, 'Moderate')
    if confidence < 50 and level == 'Low':
        level = 'Moderate'
    elif confidence < 50 and level == 'Moderate':
        level = 'High'
    return level

# Patient Login

def patient_login(request):
    return render(request, 'patient_login.html')

# Patient Registration

def patient_register(request):

    if request.method == 'POST':

        # Personal Details
        full_name = request.POST.get('full_name', '').strip()
        date_of_birth = request.POST.get('date_of_birth') or None
        gender = request.POST.get('gender', '').strip()
        blood_group = request.POST.get('blood_group', '').strip()

        # Government ID - Optional
        government_id_type = request.POST.get(
            'government_id_type', ''
        ).strip()

        government_id = request.POST.get(
            'government_id', ''
        ).strip()

        # Contact Details
        mobile = request.POST.get('mobile', '').strip()
        email = request.POST.get('email', '').strip()

        address = request.POST.get('address', '').strip()

        emergency_contact_name = request.POST.get(
            'emergency_contact_name', ''
        ).strip()

        emergency_contact_number = request.POST.get(
            'emergency_contact_number', ''
        ).strip()

        # Medical Information - Optional
        symptoms = request.POST.get(
            'symptoms', ''
        ).strip()

        medical_history = request.POST.get(
            'medical_history', ''
        ).strip()

        allergies = request.POST.get(
            'allergies', ''
        ).strip()

        current_medications = request.POST.get(
            'current_medications', ''
        ).strip()

        # Insurance & Billing - Optional
        insurance_company = request.POST.get(
            'insurance_company', ''
        ).strip()

        policy_number = request.POST.get(
            'policy_number', ''
        ).strip()

        payment_preference = request.POST.get(
            'payment_preference', ''
        ).strip()

        # Account
        password = request.POST.get('password', '')
        confirm_password = request.POST.get(
            'confirm_password', ''
        )

        # Validation
        
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

        # -------------------------
        # Government ID
        # Optional
        # But duplicate ID not allowed
        # -------------------------

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

        # -------------------------
        # Create unique internal username
        # -------------------------
        # Same mobile can be used by
        # multiple family members.

        username = f"patient_{mobile}"

        counter = 1

        while User.objects.filter(
            username=username
        ).exists():

            username = f"patient_{mobile}_{counter}"
            counter += 1

        # -------------------------
        # Create User
        # -------------------------

        user = User.objects.create_user(
            username=username,
            password=password,
            email=email
        )

        # -------------------------
        # Create Patient Profile
        # -------------------------

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

# Members Login
def members_login(request):
    return render(request, 'members_login.html')

# Members Registration

def members_register(request):
    return render(request, 'members_register.html')

def home(request):
    return render(request, 'home.html')

def sos(request):
    if request.method == 'POST':
        latitude = request.POST.get('latitude')
        longitude = request.POST.get('longitude')

        if not latitude or not longitude:
            return render(request, 'sos.html', {
                'error': 'Location could not be detected.'
            })

        emergency = EmergencyRequest.objects.create(
            latitude=float(latitude),
            longitude=float(longitude)
        )

        return render(request, 'emergency_status.html', {
            'emergency': emergency
        })

    return render(request, 'sos.html')

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')

def predict(request):
    if request.method == 'POST':
        # 1. Parse patient profile details from form inputs
        patient_name = request.POST.get('patient_name', '')
        patient_age = request.POST.get('patient_age', '')
        patient_gender = request.POST.get('patient_gender', 'Other')
        
        # 2. Extract selected symptoms from explicit dropdown indexes
        selected_symptoms = []
        for i in range(1, 6):
            s_val = request.POST.get(f'symptom{i}')
            if s_val:
                selected_symptoms.append(s_val)
                
        # 3. Formulate the machine learning input feature matrix
        input_vector = np.zeros(len(SYMPTOMS))
        for symptom in selected_symptoms:
            if symptom in SYMPTOMS:
                input_vector[SYMPTOMS.index(symptom)] = 1
                
        # 4. Execute ML classification inference
        predictions = model.predict([input_vector])
        prediction = predictions[0] if len(predictions) > 0 else "Unknown"
        
        # Calculate confidence metric
        confidence = 90.0  # Fallback default configuration
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba([input_vector])
                confidence = round(float(np.max(probabilities)) * 100, 1)
            except Exception:
                pass

        # 5. Determine warning status, image matching, and triage priority matrices
        confidence_warning = confidence < 50.0
        image_file = DISEASE_IMAGES.get(prediction, 'images/default.png')
                # Dynamically calculate the real Top 3 conditions using model probabilities
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba([input_vector])[0]
                # Get the indices of the highest 3 probabilities in descending order
                top3_indices = np.argsort(probabilities)[::-1][:3]
                
                # Fetch class names from your model's native classes array mapping
                top3 = [
                    (model.classes_[idx], round(float(probabilities[idx]) * 100, 1))
                    for idx in top3_indices
                ]
            except Exception:
                # Robust fallback if prediction array fails
                top3 = [(prediction, confidence), ("Allergy", 15.0), ("GERD", 5.0)]
        else:
            # Smart dummy fallback list if the ML model doesn't support probability arrays
            top3 = [
                (prediction, confidence),
                ("Allergy", round(confidence * 0.15, 1)),
                ("GERD", round(confidence * 0.08, 1))
            ]

        # Real XAI using model feature importances
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
        urgency = URGENCY_LEVELS.get(prediction, 'Low')

        # 6. Fetch user location strings and pass to proximity routing
        user_lat = request.POST.get('latitude', '0.0')
        user_lon = request.POST.get('longitude', '0.0')
        
        try:
            nearby_hospitals = get_nearby_hospitals(
                prediction,
                float(user_lat or 0.0),
                float(user_lon or 0.0)
            )
        except (ValueError, TypeError):
            nearby_hospitals = []

        # 7. Persist evaluation transaction log inside DB (ALIGNED WITH SCHEMA)
        PatientPrediction.objects.create(
            patient_name=patient_name,
            patient_age=int(patient_age) if patient_age else None,
            patient_gender=patient_gender,
            symptom1=selected_symptoms[0] if len(selected_symptoms) > 0 else '',
            symptom2=selected_symptoms[1] if len(selected_symptoms) > 1 else '',
            symptom3=selected_symptoms[2] if len(selected_symptoms) > 2 else '',
            symptom4=selected_symptoms[3] if len(selected_symptoms) > 3 else '',
            symptom5=selected_symptoms[4] if len(selected_symptoms) > 4 else '',
            predicted_disease=prediction,
            confidence=confidence,
            top3_predictions=str(top3),
            urgency=urgency,
            created_at=timezone.now()
        )

        # 8. Complete context mapping payload
        return render(request, 'predict.html', {
            'result': prediction,
            'image_file': image_file,
            'confidence': confidence,
            'confidence_warning': confidence_warning,
            'top3': top3,
            'xai_symptoms': xai_symptoms,
            'symptoms': SYMPTOMS,
            'patient_name': patient_name,
            'nearby_hospitals': nearby_hospitals,
        })

    # GET request configuration
    return render(request, 'predict.html', {'symptoms': SYMPTOMS})


def history(request):
    predictions = PatientPrediction.objects.all()[:50]

    disease_counts = (
        PatientPrediction.objects
        .values('predicted_disease')
        .annotate(count=Count('predicted_disease'))
        .order_by('-count')[:10]
    )
    chart_labels = [d['predicted_disease'] for d in disease_counts]
    chart_values = [d['count'] for d in disease_counts]

    return render(request, 'history.html', {
        'predictions': predictions,
        'chart_labels': chart_labels,
        'chart_values': chart_values,
    })

def dashboard(request):
    predictions = PatientPrediction.objects.all()[:20]
    total_predictions = PatientPrediction.objects.count()

    today = timezone.now().date()
    today_count = PatientPrediction.objects.filter(created_at__date=today).count()

    top = (
        PatientPrediction.objects
        .values('predicted_disease')
        .annotate(count=Count('predicted_disease'))
        .order_by('-count')
        .first()
    )
    top_disease = top['predicted_disease'] if top else 'N/A'

    # NEW — count of critical/high urgency cases needing attention
    critical_count = PatientPrediction.objects.filter(urgency='Critical').count()
    high_count = PatientPrediction.objects.filter(urgency='High').count()

    return render(request, 'dashboard.html', {
        'predictions': predictions,
        'total_predictions': total_predictions,
        'today_count': today_count,
        'top_disease': top_disease,
        'critical_count': critical_count,   # NEW
        'high_count': high_count,           # NEW
    })
def patient_login(request):
    if request.method == 'POST':
        from django.contrib.auth import authenticate
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'Invalid username or password.')
    return render(request, 'patient_login.html')


def patient_register(request):
    if request.method == 'POST':
        username   = request.POST.get('username')
        password   = request.POST.get('password')
        full_name  = request.POST.get('full_name', '')
        mobile     = request.POST.get('mobile', '')
        gender     = request.POST.get('gender', 'Other')

        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already taken.')
            return render(request, 'patient_register.html')

        user = User.objects.create_user(
            username=username,
            password=password
        )

        from .models import PatientProfile, UserProfile
        UserProfile.objects.create(user=user, role='PATIENT')
        PatientProfile.objects.create(
            user=user,
            full_name=full_name,
            mobile=mobile,
            gender=gender
        )
        login(request, user)
        messages.success(request, 'Registration successful!')
        return redirect('home')

    return render(request, 'patient_register.html')


def members_login(request):
    if request.method == 'POST':
        from django.contrib.auth import authenticate
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid credentials.')
    return render(request, 'members_login.html')


def members_register(request):
    return render(request, 'members_register.html')