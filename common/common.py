import time

def print_and_write(string, file_name):
    print(string)
    with open(file_name,'a') as f:
        f.write(string+'\n')


def print_and_write_with_time(string, file_name='log.txt'):
    print(current_time()+'    '+string)
    with open(file_name,'a') as f:
        f.write(current_time()+'    '+string+'\n')

        
def current_time():
    return time.strftime('%Y.%m.%d %H:%M:%S ',time.localtime(time.time()))
