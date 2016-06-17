from django.conf.urls import url, patterns
from panels import views

urlpatterns = patterns('',
    url(r'^$', views.index, name='index'),
    url(r'^forgot_password[/]?$', views.forgot_password, name='forgot_password'),
    url(r'^dashboard[/]?$', views.dashboard, name='dashboard'),
    url(r'^forms[/]?$', views.forms, name='forms'),
    url(r'^tables[/]?$', views.tables, name='tables'),
    url(r'^surveillance[/]?$', views.surveillance, name='surveillance'),
    url(r'^logout[/]?$', views.user_logout, name='logout'),
    url(r'^add_student[/]?$', views.add_student, name='add_student'),
    url(r'^add_teacher[/]?$', views.add_teacher, name='add_teacher'),
    url(r'^admin_login[/]?$', views.admin_login, name='admin_login'),
    url(r'^student_login[/]?$', views.student_login, name='student_login'),
)