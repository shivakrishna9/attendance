
from django.contrib.auth.models import User
from datetime import datetime

# Create your models here.


class Student(models.Model):
    name = models.CharField(max_length=20, blank=True, null=True)
    course = models.CharField(
        max_length=20, default="BTECH (Computer Engineering)")
    year = models.PositiveSmallIntegerField(default=2016)
    semester = models.SmallIntegerField(default=1)
    rollno = models.CharField(max_length=10, blank=True, null=True)
    dob = models.DateField(default=datetime(1993, 01, 02))

    username = models.CharField(max_length=20, blank=True, null=True)
    password = models.CharField(max_length=20, blank=True, null=True)

    image1 = models.ImageField(upload_to='images', default='no-image.png')
    image2 = models.ImageField(upload_to='images', default='no-image.png')
    image3 = models.ImageField(upload_to='images', default='no-image.png')
    image4 = models.ImageField(upload_to='images', default='no-image.png')

    added_on = models.DateTimeField(default=datetime.now)

    def __str__(self):
        return self.name


class Teacher(models.Model):
    user = models.OneToOneField(User, blank=True, null=True)
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

    def __str__(self):
        return self.subject_name + " <-> " + self.student_name


class Logs(models.Model):
    subject = models.ForeignKey(Subject)
    student = models.ForeignKey(Student)
    timestamp = models.DateTimeField(default=datetime.now)
    # true for coming in. false for going out
    entry_in_out = models.BooleanField(default=False)

    def __str__(self):
        return str(self.id) + ": " + str(self.timestamp)
