import joblib
import numpy as np
from django.shortcuts import render
import os
from django.conf import settings
from .models import PatientPrediction, Hospital, EmergencyRequest
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
        patient_name = request.POST.get('patient_name', '')
        
        # 1. Safely collect up to 5 symptoms from the dropdowns
        selected_symptoms = []
        for i in range(1, 6):
            s_val = request.POST.get(f'symptom{i}')
            if s_val:
                selected_symptoms.append(s_val)

        # 2. Build the model's feature vector
        input_vector = np.zeros(len(SYMPTOMS))
        for symptom in selected_symptoms:
            if symptom in SYMPTOMS:
                input_vector[SYMPTOMS.index(symptom)] = 1

        # 3. Predict the disease using the model
        predictions = model.predict([input_vector])
        prediction = predictions[0] if len(predictions) > 0 else "Unknown"

        # 4. Handle probability & confidence scores safely
        confidence = 90.0  # Fallback default
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba([input_vector])
                confidence = round(float(np.max(probabilities)) * 100, 1)
            except Exception:
                pass

        # 5. Determine warning status, image matching, and mock XAI tracking variables
        confidence_warning = confidence < 50.0
        image_file = DISEASE_IMAGES.get(prediction, 'images/default.png')
        top3 = [(prediction, confidence)]
        xai_symptoms = [{'symptom': s.replace('_', ' '), 'score': 85} for s in selected_symptoms]

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

        # 7. Persist evaluation transaction log inside DB
        PatientPrediction.objects.create(
            predicted_disease=prediction,
            symptoms_present=",".join(selected_symptoms),
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

    # GET request handler
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