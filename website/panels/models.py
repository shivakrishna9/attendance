from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User, AbstractBaseUser
from datetime import datetime
import time

# Create your models here.


class Student(models.Model):
    name = models.CharField(max_length=50, blank=True, null=True)
    course = models.CharField(
        max_length=30, default="BTECH (Computer Engineering)")
    year = models.PositiveSmallIntegerField(default=2016)
    semester = models.SmallIntegerField(default=1)
    rollno = models.CharField(max_length=10, blank=True, null=True)
    enrollno = models.CharField(max_length=10, blank=False, null=True)
    dob = models.DateField(default=datetime(1993, 01, 02))

    user = models.CharField(max_length=20, blank=True, null=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    has_module_perms = models.BooleanField(default=False, null=False)
    last_login = models.DateTimeField(default=timezone.now, null=False)

    image1 = models.ImageField(upload_to='images', default='no-image.png')
    # image2 = models.ImageField(upload_to='images', default='no-image.png')
    # image3 = models.ImageField(upload_to='images', default='no-image.png')
    # image4 = models.ImageField(upload_to='images', default='no-image.png')

    added_on = models.DateTimeField(default=datetime.now)

    # def has_module_perms(self, app_label):
    #     return False

    def is_authenticated(self):
        return True

    def __str__(self):
        return self.name


class Teacher(models.Model):

    user = models.OneToOneField(User, blank=True, null=True)
    # password = models.CharField(max_length=20)
    added_on = models.DateTimeField(default=datetime.now)

    def __str__(self):
        return self.user.username


class Subject(models.Model):
    code = models.CharField(max_length=10, blank=True, null=True)
    name = models.CharField(max_length=30, blank=True, null=True)
    teacher = models.ForeignKey(Teacher)

    def __str__(self):
        return self.name


class Studies(models.Model):
    student = models.ForeignKey(Student)
    subject = models.ForeignKey(Subject)
    attendance = models.IntegerField(default=0)
    timestamp = models.DateTimeField(default=datetime.now)
    confidence = models.IntegerField(default=0)
    date = models.DateField(default=time.strftime('%Y-%m-%d'))

    def __str__(self):
        return str(self.subject) + " <-> " + str(self.student)


class Logs(models.Model):
    subject = models.ForeignKey(Subject)
    student = models.ForeignKey(Student)
    timestamp = models.DateTimeField(default=datetime.now)
    # False for coming in. True for going out
    entry_in_out = models.BooleanField(default=True)

    def __str__(self):
        return str(self.id) + ": " + str(self.timestamp)
