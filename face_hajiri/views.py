from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Registration, Attendance
from .serializers import RegistrationSerializer, FaceVerificationSerializer, AttendanceSerializer
from django.http import QueryDict

import io
import base64
from PIL import Image
import numpy as np
import cv2
import face_recognition
from datetime import datetime
from core.utils import gen_encoding, automatic_brightness_and_contrast


stored_encodings = None
attendee_names = None
attendee_ids = None


def base64_img(img_str):
    image = base64.b64decode((img_str))
    image = Image.open(io.BytesIO(image))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def img_preprocessing(img_str):
    image = base64_img(img_str)
    # cv2.imwrite('temp.jpg', image)
    # print(type(image), image.shape)
    return gen_encoding([image])

def get_user_data():
    reg = Registration.objects.all()#filter(attendee_id='f0b67154-4bb7-11ed-bdc3-0242ac120002')
    reg_serializer = RegistrationSerializer(reg, many=True)
    stored_encodings = []
    attendee_names = []
    attendee_ids = []
    for row in reg_serializer.data:
        embedding = np.array(row['face_embedding'][0])
        attendee_name = row['attendee_name']
        attendee_id = row['attendee_id']
        stored_encodings.append(embedding)
        attendee_names.append(attendee_name)
        attendee_ids.append(attendee_id)
    return stored_encodings, attendee_names, attendee_ids

def construct_dict(name, id, device, current_time, state):
    query_dict = QueryDict('', mutable=True)
    if state == 'in':
        data = {
            'attendee_name' : name,
            'attendee_id' : id,
            'device' : device,
            'in_time' : current_time
        }
        query_dict.update(data)
        return query_dict
    else:
        print("None of either states")
        return query_dict


def store_in_time(name, id, device, current_time, state):
    query_dict = construct_dict(name, id, device, current_time, state)
    if query_dict:
        serializer_attendance = AttendanceSerializer(data = query_dict)
        if serializer_attendance.is_valid():
            print('attendance serializer is valid in else-if')
            serializer_attendance.save()
        else:
            print('attendance serializer is invalid in else-else')

class RegistrationView(APIView):
    def post(self, request):
        global stored_encodings
        global attendee_names
        global attendee_ids
        # data grabbing
        try:
            data = request.data
            print("In try")
        except:
            data = request.POST
            print("in except")
        
        # generation of face_encoding
        if data['image_base64']:
            try:
                face_encoding = img_preprocessing(data['image_base64'])
                print(f"face encoding type: {type(face_encoding)}")
                print('face encoding completed')
            except:
                return Response({'Acknowledge':'invalid image data'})
        
        # dict used in generation of query dict
        data_ = {
            'attendee_name' : data['attendee_name'],
            'attendee_id' : data['attendee_id'],
            'registration_device' : data['registration_device'],
            'department' : data['department'],
            'image_base64' : data['image_base64'],
            'face_embedding' : face_encoding,
        }
        query_dict = QueryDict('', mutable=True)
        query_dict.update(data_)

        serializer = RegistrationSerializer(data=query_dict)
        if serializer.is_valid():
            print("Valid")
            serializer.save()
            stored_encodings, attendee_names, attendee_ids = get_user_data()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class VerificationView(APIView):
    def post(self, request):
        global stored_encodings
        global attendee_names
        global attendee_ids
        
        if stored_encodings is None or attendee_names is None or attendee_ids is None:
            stored_encodings, attendee_names, attendee_ids = get_user_data()

        try:
            data = request.data
            print("In try")
        except:
            data = request.POST
            print("in except")
        # face_encoding = img_preprocessing(data['image_base64'])

        serializer = FaceVerificationSerializer(data=data)
        if serializer.is_valid():
            current_time = datetime.now()
            serializer_data = serializer.data
            # print(f"device: {serializer_data['device']}")
            img_str = serializer_data['image_base64']
            image = base64_img(img_str)

            face_cropped = face_recognition.face_locations(image, model = "cnn")
            encoded_face_in_frame = face_recognition.face_encodings(image, face_cropped)
            recognized_faces = []
            for encode_face, face_loc in zip(encoded_face_in_frame, face_cropped):
                matches = face_recognition.compare_faces(stored_encodings, encode_face)
                face_dist = face_recognition.face_distance(stored_encodings, encode_face)
                match_index = np.argmin(face_dist)
                # matched_face_distance = face_dist[match_index]
                if matches[match_index]:
                    # aggregate matched user data
                    name = attendee_names[match_index].upper()
                    id = attendee_ids[match_index]
                    recognized_faces.append({'name':name, 'id':id})

                    # write data to database
                    rows = Attendance.objects.filter(date=datetime.now().date()).filter(attendee_name=name, attendee_id=id)

                    if rows:
                        '''If any rocord is found'''
                        row = rows.order_by('-in_time')[:1].get()
                        if row.out_time is None:
                            row.out_time = current_time
                            row.save()
                        elif isinstance(row.in_time, datetime):
                            print('Here')
                            store_in_time(name, id, serializer_data['device'], current_time, state='in')
                           
                    else:
                        '''If no rocord is found'''
                        store_in_time(name, id, serializer_data['device'], current_time, state='in')

            #         y1, x2, y2, x1 = face_loc
            #         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #         cv2.putText(image, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            # cv2.imwrite('detections/detection.jpg', image)            

            return Response({'recognized':recognized_faces})
        return Response({'Acknowledge':'error'})




