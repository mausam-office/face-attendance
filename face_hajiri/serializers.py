
from rest_framework import serializers
from .models import Registration, FaceVerification, Attendance

class RegistrationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Registration
        fields = ['attendee_name', 'attendee_id', 'registration_device', 'department', 'image_base64', 'face_embedding']


class FaceVerificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = FaceVerification
        fields = "__all__"


class AttendanceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Attendance
        fields = "__all__"

