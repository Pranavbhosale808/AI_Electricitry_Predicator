from django.urls import path
from . import views

urlpatterns = [
     path('hourly/', views.api_hourly, name='api_hourly'),
    path('weekly/', views.api_weekly, name='api_weekly'),
    path('monthly/', views.api_monthly, name='api_monthly'),
    path('max_peak_load/', views.api_monthly, name='max_peak_load')
]