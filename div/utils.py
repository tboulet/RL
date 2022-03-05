import sys

def s():
    sys.exit()    

def pr_and_raise(*args):
    for arg in args:
        print(arg)
    raise

def pr(*args):
    for arg in args:
        print(arg)

def pr_shape(*args):
    for arg in args:
        try:
            print(arg.shape)
        except:
            print('No shape')