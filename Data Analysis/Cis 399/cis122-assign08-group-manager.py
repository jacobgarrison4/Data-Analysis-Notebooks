def create_group(d):
	pass

def list_groups(d):
	pass

def add_group_data(d):
	pass

def list_group_data(d):
	pass

myGroup = {}

while True:
	user = input("Command (empty or X to quit, ? for help): ")
	user = user.upper()
	if(user == "?"):
		print("?: list commands\nC: Create a new group\nG: List Groups\nA: Add data to a group\nL: List data for a group\nX: Exit")
	elif(user == "X" or user == ""):
		break
	elif(user == "C"):
		create_group(myGroup)
	elif(user == "G"):
		list_groups(myGroup)
	elif(user == "A"):
		add_group_data(myGroup)
	elif(user == "L"):
		list_group_data(myGroup)
	else:
		print("I do not understand")