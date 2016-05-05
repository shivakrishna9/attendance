# -*- coding: utf-8 -*-
# Generated by Django 1.9.5 on 2016-05-05 11:22
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Student',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=20)),
                ('course', models.CharField(max_length=20)),
                ('year', models.IntegerField(max_length=4)),
                ('semester', models.SmallIntegerField(default=1)),
            ],
        ),
        migrations.CreateModel(
            name='Studies',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('attendance', models.IntegerField(default=0)),
                ('student_name', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='panels.Student')),
            ],
        ),
        migrations.CreateModel(
            name='Subject',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=30)),
            ],
        ),
        migrations.CreateModel(
            name='Teacher',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=20)),
            ],
        ),
        migrations.AddField(
            model_name='subject',
            name='teacher',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='panels.Teacher'),
        ),
        migrations.AddField(
            model_name='studies',
            name='subject_name',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='panels.Subject'),
        ),
    ]
