from django.urls import path
from . import views_metadata, views

urlpatterns = [
    path('available_experiments', views_metadata.available_experiments, name='available_experiments'),
    path('request_setting', views_metadata.request_setting, name='request_setting'),
    path('get_setting', views_metadata.get_setting, name='get_setting'),
    path('direction_initialization', views.direction_hierarchy_initialization, name='direction_initialization'),
    path('direction_selection', views.direction_hierarchy_selection, name='direction_selection'),
    path('direction_recluster', views.direction_hierarchy_recluster, name='direction_recluster'),
    path('code_initialization', views.code_hierarchy_initialization, name='code_initialization'),
    path('code_selection', views.code_hierarchy_selection, name='code_selection')
]