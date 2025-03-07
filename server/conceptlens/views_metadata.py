import os
import json

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse


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
    found_experiments = sorted(os.listdir('served_data'))
    try:
        found_experiments.remove('.DS_Store')
    except:
        pass

    resp = json.dumps({
        'experiment_names': [v for v in found_experiments]
    })
    return JsonResponse(resp, safe=False)


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
