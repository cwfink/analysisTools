import os
import git


repo = git.Repo(os.path.dirname(os.path.abspath(__file__)))
repo.git.add("git_test2.py")
repo.git.commit(m = "Sam's test of GitPython")
#print(repo.git.status())

repo.git.pull('origin', 'master')
repo.git.push('origin', 'master')
