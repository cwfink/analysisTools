import os
import git


repo = git.Repo('/scratch/cwfink/repositories/analysisTools')
repo.git.add("git_test2.py")
repo.git.commit(m = "trying to test GitPython")
#print(repo.git.status())

repo.git.pull('origin', 'master')
repo.git.push('origin', 'master')
