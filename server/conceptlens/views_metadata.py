import os
import json

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import natsort
from django.conf import settings

SERVED_DATA_ROOT = settings.SERVED_DATA_ROOT

@csrf_exempt
def get_setting(request):
    """

    Returns:

    """
    data = json.loads(request.body)
    experiment_name = data['experiment_name']

    # Read Data
    root = os.path.join('served_data', experiment_name)

    # Read config
    with open(os.path.join(root, 'setting.json'), 'r') as file:
        setting = json.load(file)

    response = json.dumps(setting, indent=2)
    return JsonResponse(response, safe=False)


@csrf_exempt
def available_experiments(request):
    # List the experiments in the served data directory
    found_experiments = natsort.natsorted(os.listdir(SERVED_DATA_ROOT))
    # Remove unwanted files (like .DS_Store on macOS)
    if '.DS_Store' in found_experiments:
        found_experiments.remove('.DS_Store')

    # Create a list of experiment dictionaries following the original EXPERIMENT structure
    experiments = [{"name": exp} for exp in found_experiments]

    # Return the experiments under the key "EXPERIMENT"
    return JsonResponse({"EXPERIMENT": experiments})


@csrf_exempt
def request_setting(request):
    data = json.loads(request.body)
    experiment_name = data['experiment_name']

    # Read Data
    root = os.path.join('served_data', experiment_name)

    with open(os.path.join(root, 'setting.json')) as file:
        config = json.load(file)

    experiment_name = config['experiment_name']
    n_code = config['n_code']
    n_method = config['n_method']
    n_direction_per_method = config['n_direction']

    return JsonResponse({
        'data': {
            'experiment_name': experiment_name,
            'feature_space': config['feature_space'],
            'n_code': n_code,
            'n_method': n_method,
            'n_direction_per_method': n_direction_per_method
        }
    })
