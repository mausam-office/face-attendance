from django.urls import path
from face_hajiri import views


urlpatterns = [
    path('registration/', views.RegistrationView.as_view()),
    path('verification/', views.VerificationView.as_view()),
    path('userdetails/', views.UserDetailsView.as_view()),
]