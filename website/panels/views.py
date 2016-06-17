from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.utils.html import escape
from panels.camera.streamer import Camera
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from models import Student, Studies, Subject
from datetime import datetime
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


def index(request):
    return render(request, 'index.html')


def admin_login(request):
    if request.user.is_authenticated():
        return HttpResponseRedirect('/dashboard')

    if request.method == 'POST':
        username = escape(request.POST.get('username', None).strip())
        password = escape(request.POST.get('password', None).strip())

        if (username and password):
            user = authenticate(user=username, password=password)
            if user:
                if user.is_active:
                    login(request, user)
                    return HttpResponseRedirect('/dashboard')
                else:
                    return HttpResponse('Your account is disabled')
            else:
                return render(request, 'admin_login.html', {'error': 1})

    return render(request, 'admin_login.html', {'error': 0})


def student_login(request):
    if request.user.is_authenticated():
        return HttpResponseRedirect('/dashboard')

    if request.method == 'POST':
        enrollno = escape(request.POST.get('enrollno', None).strip())
        dob = escape(request.POST.get('dob', None).strip())

        if (username and password):
            user = authenticate(enrollno=enrollno, dob=dob)
            if user:
                if user.is_active:
                    login(request, user)
                    return HttpResponseRedirect('/dashboard')
                else:
                    return HttpResponse('Your account is disabled')
            else:
                return render(request, 'student_login.html', {'error': 1})

    return render(request, 'student_login.html', {'error': 0})


@login_required
def dashboard(request):

    attendance = Studies.objects.all()
    total = float(float(len([i for i in attendance])) / float(3)) * 100
    print total * 100
    lab = Subject.objects.filter(name='Major Project')

    return render(request, 'dashboard.html', {'total': total, 'lab': lab})


@login_required
def forms(request):
    return render(request, 'forms.html')


@login_required
def add_student(request):
    from uuid import uuid4

    if request.method == 'POST':
        name = escape(request.POST.get('name', None).strip())
        username = escape(request.POST.get('username', None).strip())
        # password = escape(request.POST.get('password', None).strip())
        rno = escape(request.POST.get('rno', None).strip())
        dob = escape(request.POST.get('dob', None).strip())
        course = escape(request.POST.get('course', None).strip())
        year = escape(request.POST.get('year', None).strip())
        semester = escape(request.POST.get('semester', None).strip())
        image1 = request.FILES.get('image1')
        image2 = request.FILES.get('image2')
        image3 = request.FILES.get('image3')
        image4 = request.FILES.get('image4')
        image1.name = '{}{}'.format(
            rno + '_1', image1.name[image1.name.rfind('.'):])
        image2.name = '{}{}'.format(
            rno + '_2', image2.name[image2.name.rfind('.'):])
        image3.name = '{}{}'.format(
            rno + '_3', image3.name[image3.name.rfind('.'):])
        image4.name = '{}{}'.format(
            rno + '_4', image4.name[image4.name.rfind('.'):])
        student = Student(name=name, username=username, password=password, rollno=rno, dob=dob,
                          course=course, year=year, semester=semester, image1=image1, image2=image2,
                          image3=image3, image4=image4)
        student.save()

        return HttpResponseRedirect('/dashboard')

    return HttpResponseRedirect('/forms')


@login_required
def add_teacher(request):
    from uuid import uuid4

    if request.method == 'POST':
        user = escape(request.POST.get('username', None).strip())
        password = escape(request.POST.get('password', None).strip())
        teacher = Teacher(user=username, password=password)
        teacher.save()

    return


@login_required
def tables(request, low=None, mid=None, high=None):

    # images = ['demo/DSC_1666.JPG','demo/DSC_1663.JPG']
    # images = ['demo/DSC_1663.JPG']
    # images = ['DSC_1663.JPG','DSC_1666.JPG']
    images = ['']
    for img in images:
        # frame = camera.read_cam()
        # low, mid, high = recognition.run(people, img, batch_size=4)
        high = ['ashar', 'shafiya', 'sapna']
        mid = []
        low = ['nikhil_mittal']
        if high != None:
            for i in high:
                # print i
                st = Student.objects.get(username=i)
                lab = Subject.objects.get(name='lab')
                h = Studies.objects.filter(student__username=i, subject__name='lab',
                                           date=time.strftime('%Y-%m-%d'), confidence=2).exists()
                m = Studies.objects.filter(student__username=i, subject__name='lab',
                                           date=time.strftime('%Y-%m-%d'), confidence=1).exists()
                l = Studies.objects.filter(student__username=i, subject__name='lab',
                                           date=time.strftime('%Y-%m-%d'), confidence=0).exists()
                if not h:
                    attendance = Studies(student=st, subject=lab, confidence=2)
                    attendance.save()
                # if h:
                #     log

                if m:
                    print i
                    attendance = Studies.objects.filter(student__name=i,
                                                        subject__name='lab', date=time.strftime('%Y-%m-%d'), confidence=1).delete()
                if l:
                    print i
                    attendance = Studies.objects.filter(student__name=i,
                                                        subject__name='lab', date=time.strftime('%Y-%m-%d'), confidence=1).delete()
        if mid != None:
            for i in mid:
                st = Student.objects.get(username=i)
                lab = Subject.objects.get(name='lab')
                h = Studies.objects.filter(student__username=i, subject__name='lab',
                                           date=time.strftime('%Y-%m-%d'), confidence=2).exists()
                m = Studies.objects.filter(student__username=i, subject__name='lab',
                                           date=time.strftime('%Y-%m-%d'), confidence=1).exists()
                l = Studies.objects.filter(student__username=i, subject__name='lab',
                                           date=time.strftime('%Y-%m-%d'), confidence=0).exists()
                if not (h or m):
                    attendance = Studies(student=st, subject=lab, confidence=1)
                    attendance.save()
                if l:
                    attendance = Studies.objects.filter(student__name=i,
                                                        subject__name='lab', date=time.strftime('%Y-%m-%d'), confidence=1).delete()
        if low != None:
            for i in low:
                st = Student.objects.get(username=i)
                lab = Subject.objects.get(name='lab')
                h = Studies.objects.filter(student__username=i, subject__name='lab',
                                           date=time.strftime('%Y-%m-%d'), confidence=2).exists()
                m = Studies.objects.filter(student__username=i, subject__name='lab',
                                           date=time.strftime('%Y-%m-%d'), confidence=1).exists()
                l = Studies.objects.filter(student__username=i, subject__name='lab',
                                           date=time.strftime('%Y-%m-%d'), confidence=0).exists()

                if not (h or m or l):

                    attendance = Studies(student=st, subject=lab, confidence=0)
                    attendance.save()

    attendance = Studies.objects.filter(date=time.strftime('%Y-%m-%d'))
    # attendance = Studies.objects.all()
    return render(request, 'tables.html', {'attendance': attendance})


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
