from utils.file_handling import get_absolute_path


def test_get_absolute_path():
    entropy_file = get_absolute_path('data', directory_file_hint='cartpole', ignore_files=False)
    print(entropy_file)