# Generated by Django 4.0 on 2022-10-18 10:52

import django.contrib.postgres.fields
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Registration',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('attendee_name', models.CharField(max_length=64)),
                ('attendee_id', models.UUIDField(unique=True)),
                ('registration_device', models.CharField(max_length=64)),
                ('department', models.CharField(max_length=64)),
                ('image_base64', models.TextField()),
                ('face_embedding', django.contrib.postgres.fields.ArrayField(base_field=django.contrib.postgres.fields.ArrayField(base_field=models.FloatField(), size=None), size=None)),
                ('created_on', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]