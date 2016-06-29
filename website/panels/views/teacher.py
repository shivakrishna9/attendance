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
from recognise.get_input import *
from collections import Counter
import time
import pandas as pd

# # Create your views here.

i=[]
for image in glob.glob("media/images/*/*.*"):
        person = image.split('/')[2]
        i.append(person)

x = Counter(i)
people = list(set(i))
people = sorted(people)

print "People: ", people

recognition = VGG()
camera = Camera()
recognition.VGGNet(people, len(people))

recognition.train(batch_size=4, epochs=4, lr=1.5e-4, nb_epoch=1)


def index(request):
    return render(request, 'index.html')


def admin_login(request):

    if request.method == 'POST':
        username = escape(request.POST.get('username', None).strip())
        password = escape(request.POST.get('password', None).strip())

        if (username and password):
            user = authenticate(username=username, password=password)
            if user:
                if user.is_active:
                    login(request, user)
                    admin = Teacher.objects.get(user=request.user)
                    return render(request, 'admin/dashboard.html', {'admin': admin})
                else:
                    return HttpResponse('Your account is disabled')
            else:
                return render(request, 'admin/login.html', {'error': 1})

    return render(request, 'admin/login.html', {'error': 0})


@login_required
def teacher(request):

    attendance = Studies.objects.all()
    total = float(float(len([i for i in attendance])) / float(3)) * 100
    print total * 100
    lab = Subject.objects.filter(name='Major Project')

    # admin = Teacher.objects.get(user=request.user)

    return render(request, 'admin/dashboard.html', {'total': total, 'lab': lab})

    # return render(request, 'dashboard.html', {})


@login_required
def forms(request):

    # admin = Teacher.objects.get(user=request.user)

    return render(request, 'admin/forms.html')


@login_required
def add_student(request):
    from uuid import uuid4

    if request.method == 'POST':
        name = escape(request.POST.get('name', None).strip())
        user = escape(request.POST.get('user', None).strip())
        # password = escape(request.POST.get('password', None).strip())
        rno = escape(request.POST.get('rno', None).strip())
        enrollno = escape(request.POST.get('enrollno', None).strip())
        dob = escape(request.POST.get('dob', None).strip())
        course = escape(request.POST.get('course', None).strip())
        year = escape(request.POST.get('year', None).strip())
        semester = escape(request.POST.get('semester', None).strip())
        image1 = request.FILES.get('image1')
        if not os.path.exists('media/images/'+rno+'/'):
            os.mkdir('media/images/'+rno+'/')
        image1.name = '{}{}'.format(
            rno + '_1', image1.name[image1.name.rfind('.'):])
        student = Student(name=name, user=user, password=password, rollno=rno, dob=dob,
                          course=course, year=year, semester=semester, image1=image1)
        student.save()

        return HttpResponseRedirect('/dashboard')

    return HttpResponseRedirect('/forms')


@login_required
def add_teacher(request):
    # from uuid import uuid4

    if request.method == 'POST':
        username = escape(request.POST.get('username', None).strip())
        password = escape(request.POST.get('password', None).strip())
        teacher = User(username=username, password=password)
        teacher.save()

        return HttpResponseRedirect('/dashboard')

    return HttpResponseRedirect('/forms')


@login_required
def admin_tables(request, low=None, mid=None, high=None):

    if request.method == 'POST':
        pk = escape(request.POST.get('attendance', None).strip())
        Studies.objects.filter(pk=pk).delete()
# <<<<<<< HEAD

    else:

        start = time.time()
        while camera.read_cam() != None :
            img = camera.read_cam()
            low, mid, high = recognition.run(img, batch_size=2)
            if high != None:
                for i in high:
                    # print i
                    if i != 'not_face':
                        st = Student.objects.get(rollno=i)
                        lab = Subject.objects.get(name='lab')
                        h = Studies.objects.filter(student__rollno=i, subject__name='lab',
                                                   date=time.strftime('%Y-%m-%d'), confidence=2).exists()
                        m = Studies.objects.filter(student__rollno=i, subject__name='lab',
                                                   date=time.strftime('%Y-%m-%d'), confidence=1).exists()
                        l = Studies.objects.filter(student__rollno=i, subject__name='lab',
                                                   date=time.strftime('%Y-%m-%d'), confidence=0).exists()
                        if not h:
                            attendance = Studies(
                                student=st, subject=lab, confidence=2)
                            attendance.save()

                        if m:
                            print i
                            Studies.objects.filter(student__rollno=i,
                                                   subject__name='lab', date=time.strftime('%Y-%m-%d'), confidence=1).delete()
                        if l:
                            print i
                            Studies.objects.filter(student__rollno=i,
                                                   subject__name='lab', date=time.strftime('%Y-%m-%d'), confidence=1).delete()
            if mid != None:
                for i in mid:
                    if i != 'not_face':
                        st = Student.objects.get(rollno=i)
                        lab = Subject.objects.get(name='lab')
                        h = Studies.objects.filter(student__rollno=i, subject__name='lab',
                                                   date=time.strftime('%Y-%m-%d'), confidence=2).exists()
                        m = Studies.objects.filter(student__rollno=i, subject__name='lab',
                                                   date=time.strftime('%Y-%m-%d'), confidence=1).exists()
                        l = Studies.objects.filter(student__rollno=i, subject__name='lab',
                                                   date=time.strftime('%Y-%m-%d'), confidence=0).exists()
                        if not (h or m):
                            attendance = Studies(
                                student=st, subject=lab, confidence=1)
                            attendance.save()
                        if l:
                            Studies.objects.filter(student__rollno=i,
                                                   subject__name='lab', date=time.strftime('%Y-%m-%d'), confidence=1).delete()

            if low != None:
                for i in low:
                    if i != 'not_face':
                        st = Student.objects.get(rollno=i)
                        lab = Subject.objects.get(name='lab')
                        h = Studies.objects.filter(student__rollno=i, subject__name='lab',
                                                   date=time.strftime('%Y-%m-%d'), confidence=2).exists()
                        m = Studies.objects.filter(student__rollno=i, subject__name='lab',
                                                   date=time.strftime('%Y-%m-%d'), confidence=1).exists()
                        l = Studies.objects.filter(student__rollno=i, subject__name='lab',
                                                   date=time.strftime('%Y-%m-%d'), confidence=0).exists()

                        if not (h or m or l):

                            attendance = Studies(
                                student=st, subject=lab, confidence=0)
                            attendance.save()

        print 'Time for total:', time.time() - start

    attendance = Studies.objects.filter(subject__name='lab', date=time.strftime('%Y-%m-%d'))
    # attendance = Studies.objects.all()
    # admin = Teacher.objects.get(user=request.user)
    return render(request, 'admin/tables.html', {'attendance': attendance})


def forgot_password(request):
    return render(request, 'forgot_password.html')


@login_required
def surveillance():

    camera = Camera()
    camera.surveillance()
    # return render(request, 'index.html')


@login_required
def user_logout(request):
    logout(request)
    return HttpResponseRedirect('/')


def train(request):
    recognition.train(batch_size=4, epochs=4, lr=1.5e-4, nb_epoch=1)