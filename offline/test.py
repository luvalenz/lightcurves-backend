__author__ = 'lucas'
import glob, yaml

print(glob.glob('*'))
print(yaml.safe_load(open('../config.yaml')))