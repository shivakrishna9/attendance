from __future__ import unicode_literals

from django.db import models

# Create your models here.
class Student(models.Model):
    name = models.CharField(max_length=20)
    course = models.CharField(max_length=20)
    year = models.IntegerField(max_length=4)
    semester = models.SmallIntegerField(default=1)

    def __str__(self):
        return self.name

class Teacher(models.Model):
    name = models.CharField(max_length=20)

    def __str__(self):
        return self.name

class Subject(models.Model):
    name = models.CharField(max_length=30)
    teacher = models.ForeignKey(Teacher)

    def __str__(self):
        return self.name

class Studies(models.Model):
    student_name = models.ForeignKey(Student)
    subject_name = models.ForeignKey(Subject)
    attendance = models.IntegerField(default=0)

    def __str__(self):
        return self.subject_name + " <-> " + self.student_name