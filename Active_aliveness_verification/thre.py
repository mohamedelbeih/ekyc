# Python program to illustrate the concept
# of threading
# importing the threading module
import threading
import time


def print_cube(num):
	# function to print cube of given num
	time.sleep(5)
	print("Cube: {}" .format(num * num * num))


def print_square(num):
	# function to print square of given num
	t2 = threading.Thread(target=print_cube, args=(10,))
	t2.start()
	print("Square: {}" .format(num * num))



t1 = threading.Thread(target=print_square, args=(10,))
t1.start()



#t1.join()

#t2.join()


print("Done!")

