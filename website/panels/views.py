from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.utils.html import escape
from panels.camera.streamer import Camera
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from models import Student

# import cv2

# Create your views here.


def index(request):
    if request.user.is_authenticated():
        return HttpResponseRedirect('/dashboard')

    if request.method == 'POST':
        username = escape(request.POST.get('username', None).strip())
        password = escape(request.POST.get('password', None).strip())

        if (username and password):
            user = authenticate(username=username, password=password)
            if user:
                if user.is_active:
                    login(request, user)
                    return HttpResponseRedirect('/dashboard')
                else:
                    return HttpResponse('Your account is disabled')
            else:
                return render(request, 'index.html', {'error': 1})
    return render(request, 'index.html')

@login_required
def dashboard(request):
    return render(request, 'dashboard.html')

@login_required
def forms(request):
    return render(request, 'forms.html')

@login_required
def add_student(request):
    from uuid import uuid4

    if request.method == 'POST':
        name = escape(request.POST.get('name', None).strip())
        username = escape(request.POST.get('username', None).strip())
        password = escape(request.POST.get('password', None).strip())
        rno = escape(request.POST.get('roll', None).strip())
        dob = escape(request.POST.get('dob', None).strip())
        course = escape(request.POST.get('course', None).strip())
        year = escape(request.POST.get('year', None).strip())
        semester = escape(request.POST.get('semester', None).strip())
        image1 = request.FILES.get('image1')
        image2 = request.FILES.get('image2')
        image3 = request.FILES.get('image3')
        image4 = request.FILES.get('image4')
        image1.name = '{}{}'.format(uuid4().hex, image1.name[image1.name.rfind('.'):])
        image2.name = '{}{}'.format(uuid4().hex, image2.name[image1.name.rfind('.'):])
        image3.name = '{}{}'.format(uuid4().hex, image3.name[image1.name.rfind('.'):])
        image4.name = '{}{}'.format(uuid4().hex, image4.name[image1.name.rfind('.'):])
        student = Student(name=name, username=username, password=password, rno=rno, dob=dob,
                          course=course, year=year, semester=semester, image1=image1, image2=image2,
                          image3=image3, image4=image4)
        student.save()

        return HttpResponseRedirect('/dashboard')

    return HttpResponseRedirect('/forms')

def tables(request):
    return render(request, 'tables.html')


def forgot_password(request):
    return render(request, 'forgot_password.html')


def surveillance(request):

    camera = Camera()
    camera.surveillance()
    return render(request, 'index.html')

def user_logout(request):
    logout(request)
    return HttpResponseRedirect('/')
