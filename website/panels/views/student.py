from django.conf import settings

from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.utils.html import escape
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User

from panels.models import Student, Studies, Subject, Teacher
from datetime import datetime

from panels.camera.streamer import Camera
import cv2
from recognise.class_net import *

# # Create your views here.

people = ['abdul_karim', 'abdul_wajid', 'abhishek_bhatnagar', 'abhishek_joshi', 'aditya', 'ahsan',
          'akshat', 'aly', 'aman', 'ameen', 'antriksh', 'anzal', 'ashar', 'asif', 'avishkar', 'bushra',
          'chaitanya', 'dhawal', 'farhan', 'farheen', 'ghalib', 'habib', 'harsh', 'irfan_ansari',
          'jeevan', 'manaff', 'manish', 'maria', 'mehrab', 'mohib', 'naeem', 'nikhil_mittal', 'nikhil_raman',
          'prerit', 'raghib_ahsan', 'rahul', 'ravi', 'rehan', 'rezwan', 'rubab', 'sachin', 'sahil', 'saif',
          'saifur', 'sajjad', 'sana', 'sapna', 'sarah_khan', 'sarah_masud', 'sarthak', 'shadab', 'shafiya',
          'shahbaz', 'shahjahan', 'sharan', 'shivam', 'shoaib', 'shoib', 'shruti', 'suhani', 'sultana',
          'sunny', 'sushmita', 'tushar', 'umar', 'zeya', 'zishan']

# recognition = VGG()
# camera = Camera()
# recognition.VGGNet(people)

class StudentBackend(object):
    """
    Authenticate Student Backend. Students can view using their enrollment number and date of birth.
    """

    def authenticate(self, enrollno=None, dob=None):
        
        login_valid = Student.objects.filter(dob=dob, enrollno=enrollno).exists()
        
        if login_valid:
            try:
                student = Student.objects.get(dob=dob, enrollno=enrollno)
            except Student.DoesNotExist:
                return None
            return student
        
        return None

    def get_user(self, user_id):
        
        try:
            return Student.objects.get(pk=user_id)
        except Student.DoesNotExist:
            return None

def student_required(function):
    """Check that the student is NOT logged in.

    """
    def _dec(view_func):
        def _view(request, *args, **kwargs):
            user = request.user

            login_valid = Student.objects.filter(dob=user.dob, enrollno=user.enrollno).exists()            
            if login_valid:
                return view_func(request, *args, **kwargs)
            else:
                return render(request, 'index.html')
                

        _view.__name__ = view_func.__name__
        _view.__dict__ = view_func.__dict__
        _view.__doc__ = view_func.__doc__

        return _view

    if function is None:
        return _dec
    else:
        return _dec(function)


def student_login(request):
    # if request.user.is_authenticated():
    #     return HttpResponseRedirect('/dashboard')

    if request.method == 'POST':
        enrollno = escape(request.POST.get('enrollno', None).strip())
        dob = escape(request.POST.get('dob', None).strip())

        if (enrollno and dob):
            user = authenticate(enrollno=enrollno, dob=dob)
                
            if user.is_active:
                login(request, user)
                student = Student.objects.get(dob=dob, enrollno=enrollno)
                return render(request, 'student/dashboard.html', {'student': student})
            else:
                return render(request, 'student/login.html', {'error': 1})

    return render(request, 'student/login.html', {'error': 0})


@student_required
def student(request):

    attendance = Studies.objects.all()
    total = float(float(len([i for i in attendance])) / float(3)) * 100
    print total * 100
    lab = Subject.objects.filter(name='Major Project')

    student = Student.objects.get(username=request.user.username)

    return render(request, 'student/dashboard.html', {'student': student, 'total': total, 'lab': lab})
    # return render(request, 'student_dashboard.html')


@student_required
def student_tables(request):

    student = Student.objects.get(enrollno=request.user.enrollno, dob=request.user.dob)
    attendance = Studies.objects.filter(student=student)
    # attendance = Studies.objects.all()
    return render(request, 'student/tables.html', {'student': student, 'attendance': attendance})
