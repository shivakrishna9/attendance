from __future__ import unicode_literals

from django.db import models
import time

# Create your models here.


class student(models.Model):
    student_id = models.CharField(max_length=200)
    name = models.IntegerField(default=1)
    roll_no = models.IntegerField(default=0)
    year = models.IntegerField(default=1)
    dob = models.IntegerField(default=0)
    image1 = models.BooleanField(default=0)
    image2 = models.CharField(default='', max_length=500)
    image3 = models.DateTimeField(default=time.strftime('%Y-%m-%d'))
    image4 = models.DateTimeField(default=time.strftime('%Y-%m-%d %H:%M:%S'))

    def __unicode__(self):
        return self.name


class teacher(models.Model):
    teacher_id = models.CharField(max_length=200)
    name = models.IntegerField(default=1)
    username = models.BooleanField(default=0)
    password = models.IntegerField(default=0)

    def __unicode__(self):
        return self.name


class lab(models.Model):
	lab_code = models.IntegerField(default=0)
    name = models.IntegerField(default=0)
    semester = models.IntegerField(default=0)
    teacher_id = models.IntegerField(default=0)
    
    def __unicode__(self):
        return self.name

class attendance(models.Model):
    timestamp = models.DateTimeField(default=time.strftime('%Y-%m-%d %H:%M:%S'))
    student_id = models.IntegerField(default=0)
    lab_code = models.IntegerField(default=0)
    teacher_id = models.IntegerField(default=0)
    onlydate = models.DateTimeField(default=time.strftime('%Y-%m-%d'))
    
    def __unicode__(self):
        return self.student_id

class admin(models.Model):
    admin_id = models.IntegerField(default=0)
    username = models.IntegerField(default=0)
    name = models.IntegerField(default=0)
    password = models.IntegerField(default=0)
    
    def __unicode__(self):
        return self.name

class logs(models.Model):
    timestamp = models.DateTimeField(default=time.strftime('%Y-%m-%d %H:%M:%S'))
    student_id = models.IntegerField(default=0)
    lab_code = models.IntegerField(default=0)
    enter_exit = models.IntegerField(default=0)

    def __unicode__(self):
        return self.timestamp


class session(models.Model):
    session_id = models.IntegerField(default=0)
    ip_address = models.IntegerField(default=0)
    user_agent = models.IntegerField(default=0)
    last_activity = models.IntegerField(default=0)
    onlydate = models.DateTimeField(default=time.strftime('%Y-%m-%d'))
    timestamp = models.DateTimeField(default=time.strftime('%Y-%m-%d %H:%M:%S'))

    def __unicode__(self):
        return self.user_agent
