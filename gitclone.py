from git import Repo, RemoteProgress

class CloneProgress(RemoteProgress):
    def update(self, op_code, cur_count, max_count=None, message=''):
        if message:
            print(message)

def gitclone(url, path):
    print('Cloning ' + url)
    Repo.clone_from(url, path, progress=CloneProgress())